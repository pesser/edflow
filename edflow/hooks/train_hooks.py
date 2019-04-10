import tensorflow as tf
import os
import time

from edflow.hooks.hook import Hook
from edflow.hooks.evaluation_hooks import get_checkpoint_files
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch

import signal
import sys

"""TensorFlow hooks useful during training."""


class CheckpointHook(Hook):
    '''Does that checkpoint thingy where it stores everything in a
    checkpoint.'''

    def __init__(self,
                 root_path,
                 variables,
                 modelname='model',
                 session=None,
                 step=None,
                 interval=None,
                 max_to_keep=5):
        '''Args:
            root_path (str): Path to where the checkpoints are stored.
            variables (list): List of all variables to keep track of.
            session (tf.Session): Session instance for saver.
            modelname (str): Used to name the checkpoint.
            step (tf.Tensor or callable): Step op, that can be evaluated
                (i,.e. a tf.Tensor or a python callable returning the step as
                an integer).
            interval (int): Number of iterations after which a checkpoint is
                saved. If None, a checkpoint is saved after each epoch.
            max_to_keep (int): Maximum number of checkpoints to keep on
                disk. Use 0 or None to never delete any checkpoints.
        '''

        signal.signal(signal.SIGINT, self.at_exception)
        signal.signal(signal.SIGTERM, self.at_exception)

        self.root = root_path
        self.interval = interval
        self.step = step if step is not None else tf.train.get_global_step()

        self.saver = tf.train.Saver(variables, max_to_keep=max_to_keep)
        self.logger = get_logger(self)

        os.makedirs(root_path, exist_ok=True)
        self.savename = os.path.join(root_path, '{}.ckpt'.format(modelname))

        self.session = session

    def before_epoch(self, ep):
        if self.session is None:
            self.session = tf.get_default_session()

    def after_epoch(self, epoch):
        if self.interval is None:
            self.save()

    def after_step(self, step, last_results):
        if self.interval is not None \
                and self.global_step() % self.interval == 0:
            self.save()

    def at_exception(self, *args, **kwargs):
        self.save()

        sys.exit()

    def save(self):
        global_step = self.global_step()
        self.saver.save(self.session, self.savename, global_step=global_step)
        self.logger.info("Saved model to {}".format(self.savename))

    def global_step(self):
        if isinstance(
                self.step, tf.Tensor) or isinstance(
                self.step, tf.Variable):
            global_step = self.step
        else:
            global_step = self.step()
        return global_step


class LoggingHook(Hook):
    '''Supply and evaluate logging ops at an intervall of training steps.'''

    def __init__(self,
                 scalars={},
                 histograms={},
                 images={},
                 logs={},
                 graph=None,
                 interval=100,
                 root_path='logs'):
        '''Args:
            scalars (dict): Scalar ops.
            histograms (dict): Histogram ops.
            images (dict): Image ops. Note that for these no
                tensorboard logging ist used but a custom image saver.
            logs (dict): Logs to std out via logger.
            graph (tf.Graph): Current graph.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        '''

        scalars = [tf.summary.scalar(n, s) for n, s in scalars.items()]
        histograms = [tf.summary.histogram(n, h)
                      for n, h in histograms.items()]

        self._has_summary = len(scalars + histograms) > 0
        if self._has_summary:
            summary_op = tf.summary.merge(scalars + histograms)
        else:
            summary_op = tf.no_op()

        self.fetch_dict = {'summaries': summary_op,
                           'logs': logs,
                           'images': images}

        self.interval = interval

        self.graph = graph
        self.root = root_path
        self.logger = get_logger(self)

    def before_epoch(self, ep):
        if ep == 0:
            if self.graph is None:
                self.graph = tf.get_default_graph()

            self.writer = tf.summary.FileWriter(self.root, self.graph)

    def before_step(self, batch_index, fetches, feeds, batch):
        if batch_index % self.interval == 0:
            fetches['logging'] = self.fetch_dict

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results['global_step']
            last_results = last_results['logging']
            if self._has_summary:
                summary = last_results['summaries']
                self.writer.add_summary(summary, step)

            logs = last_results['logs']
            for name in sorted(logs.keys()):
                self.logger.info('{}: {}'.format(name, logs[name]))

            for name, image_batch in last_results['images'].items():
                full_name = name + "_{:07}.png".format(step)
                save_path = os.path.join(self.root, full_name)
                plot_batch(image_batch, save_path)

            self.logger.info("project root: {}".format(self.root))


class RetrainHook(Hook):
    '''Restes the global step at the beginning of training.'''

    def __init__(self, global_step=None):
        '''Args:
            global_step (tf.Variable): Variable tracking the training step.
        '''

        self.global_step = global_step
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        self.epoch = epoch

    def before_step(self, batch_index, fetches, feeds, batch):
        if self.epoch == 0 and batch_index == 0:
            fetches['reset_step'] = tf.assign(self.global_step, 0)

    def after_step(self, step, *args, **kwargs):
        if step == 0 and self.epoch == 0:
            self.logger.info("Reset global_step")


class WaitForManager(Hook):
    '''Wait to make sure checkpoints are not overflowing.'''

    def __init__(self,
                 checkpoint_root,
                 max_n,
                 interval=5):
        '''Args:
            checkpoint_root (str): Path to look for checkpoints.
            max_n (int): Wait as long as there are more than max_n ckpts.
            interval (float): Number of seconds after which to check for number
                of checkpoints again.
        '''

        self.root = checkpoint_root
        self.max_n = max_n
        self.sleep_interval = interval

        self.logger = get_logger(self)

    def wait(self):
        '''Loop until the number of checkpoints got reduced.'''
        while True:
            n_ckpts = len(get_checkpoint_files(self.root))
            if n_ckpts <= self.max_n:
                break
            self.logger.info(
                "Found {} checkpoints.".format(n_ckpts)
                + "Waiting until one is removed.")
            time.sleep(self.sleep_interval)

    def before_epoch(self, ep):
        self.wait()

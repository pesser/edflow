import tensorflow as tf
import os

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch


class CheckpointHook(Hook):
    '''Does that checkpoint thingy where it stores everything in a
    checkpoint.'''

    def __init__(self,
                 root_path,
                 variables,
                 modelname='model',
                 session=None,
                 step=None,
                 interval=None):
        '''Args:
            root_path (str): Path to where the checkpoints are stored.
            variables (list): List of all variables to keep track of.
            session (tf.Session): Session instance for saver.
            modelname (str): Used to name the checkpoint.
            step (tf.Tensor): Step op, that can be evaluated.
            interval (int): Number of iterations after which a checkpoint is
                saved. If None, a checkpoint is saved after each epoch.
        '''

        self.root = root_path
        self.interval = interval
        self.step = step if step is not None else tf.train.get_global_step()

        self.saver = tf.train.Saver(variables)
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
                and step % self.interval == 0:
            self.save()

    def save(self):
        self.saver.save(self.session, self.savename, global_step=self.step)
        self.logger.info("Saved model to {}".format(self.savename))


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

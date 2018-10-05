import torch
import os

from tensorboardX import SummaryWriter

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch
from edflow.util import retrieve


"""PyTorch hooks useful during training."""


class PyCheckpointHook(Hook):
    '''Does that checkpoint thingy where it stores everything in a
    checkpoint.'''

    def __init__(self,
                 root_path,
                 model,
                 modelname='model',
                 interval=None):
        '''Args:
            root_path (str): Path to where the checkpoints are stored.
            model (nn.Module): Model to checkpoint.
            modelname (str): Prefix for checkpoint files.
            interval (int): Number of iterations after which a checkpoint is
                saved. If None, a checkpoint is saved after each epoch.
        '''

        self.root = root_path
        self.interval = interval
        self.model = model

        self.logger = get_logger(self)

        os.makedirs(root_path, exist_ok=True)
        self.savename = os.path.join(root_path,
                                     '{{}}-{{}}_{}.ckpt'.format(modelname))

    def before_epoch(self, epoch):
        self.epoch = epoch

    def after_epoch(self, epoch):
        if self.interval is None:
            self.save()

    def after_step(self, step, last_results):
        self.step = retrieve('global_step', last_results)
        if self.interval is not None \
                and step % self.interval == 0:
            self.save()

    def save(self):
        e = self.epoch
        s = self.step

        savename = self.savename.format(e, s)
        torch.save(self.model.state_dict(), savename)
        self.logger.info("Saved model to {}".format(savename))


class PyLoggingHook(Hook):
    '''Supply and evaluate logging ops at an intervall of training steps.'''

    def __init__(self,
                 log_ops=[],
                 scalar_keys=[],
                 histogram_keys=[],
                 image_keys=[],
                 log_keys=[],
                 graph=None,
                 interval=100,
                 root_path='logs'):
        '''Args:
            log_ops (list): Ops to run at logging time.
            scalars (dict): Scalar ops.
            histograms (dict): Histogram ops.
            images (dict): Image ops. Note that for these no
                tensorboard logging ist used but a custom image saver.
            logs (dict): Logs to std out via logger.
            graph (tf.Graph): Current graph.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        '''

        self.log_ops = log_ops

        self.scalar_keys = scalar_keys
        self.histogram_keys = histogram_keys
        self.image_keys = image_keys
        self.log_keys = log_keys

        self.interval = interval

        self.tb_logger = SummaryWriter(root_path)

        self.graph = graph
        self.root = root_path
        self.logger = get_logger(self)

    def before_step(self, batch_index, fetches, feeds, batch):
        if batch_index % self.interval == 0:
            fetches['logging'] = self.log_ops

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results['global_step']

            for key in self.scalar_keys:
                value = retrieve(key, last_results)
                self.tb_logger.add_scalar(key, value, step)

            for key in self.histogram_keys:
                value = retrieve(key, last_results)
                self.tb_logger.add_histogram(key, value, step)

            for key in self.image_keys:
                value = retrieve(key, last_results)
                self.tb_logger.add_image(key, value, step)

            for key in self.log_keys:
                value = retrieve(key, last_results)
                self.logger.info('{}: {}'.format(key, value))


class ToNumpyHook(Hook):
    '''Converts all pytorch Variables and Tensors in the results to numpy
    arrays and leaves the rest as is.'''

    def after_step(self, step, results):
        def convert(var_or_tens):
            if hasattr(var_or_tens, 'cpu'):
                var_or_tens = var_or_tens.cpu()

            if isinstance(var_or_tens, torch.autograd.Variable):
                return var_or_tens.data.numpy()
            elif isinstance(var_or_tens, torch.Tensor):
                return var_or_tens.numpy()
            else:
                return var_or_tens

        walk(results, convert, inplace=True)

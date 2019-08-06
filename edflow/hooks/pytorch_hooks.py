import os
import sys

import torch
import numpy as np
from tensorboardX import SummaryWriter

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.util import retrieve
from edflow.util import walk
from edflow.iterators.batches import plot_batch

"""PyTorch hooks useful during training."""


class PyCheckpointHook(Hook):
    """Does that checkpoint thingy where it stores everything in a
    checkpoint."""

    def __init__(self, root_path, model, modelname="model", interval=None):
        """Args:
            root_path (str): Path to where the checkpoints are stored.
            model (nn.Module): Model to checkpoint.
            modelname (str): Prefix for checkpoint files.
            interval (int): Number of iterations after which a checkpoint is
                saved. In any case a checkpoint is savead after each epoch.
        """

        self.root = root_path
        self.interval = interval
        self.model = model

        self.logger = get_logger(self)

        os.makedirs(root_path, exist_ok=True)
        self.savename = os.path.join(root_path, "{{}}-{{}}_{}.ckpt".format(modelname))

        # Init to save even before first step... More of a debug statement
        self.step = 0
        self.epoch = 0

    def before_epoch(self, epoch):
        self.epoch = epoch

    def after_epoch(self, epoch):
        self.save()

    def after_step(self, step, last_results):
        self.step = retrieve(last_results, "global_step")
        if self.interval is not None and step % self.interval == 0:
            self.save()

    def at_exception(self, *args, **kwargs):
        self.save()

    def save(self):
        e = self.epoch
        s = self.step

        savename = self.savename.format(e, s)
        torch.save(self.model.state_dict(), savename)
        self.logger.info("Saved model to {}".format(savename))


class PyLoggingHook(Hook):
    """Supply and evaluate logging ops at an intervall of training steps."""

    def __init__(
        self,
        log_ops=[],
        scalar_keys=[],
        histogram_keys=[],
        image_keys=[],
        log_keys=[],
        graph=None,
        interval=100,
        root_path="logs",
    ):
        """Args:
            log_ops (list): Ops to run at logging time.
            scalars (dict): Scalar ops.
            histograms (dict): Histogram ops.
            images (dict): Image ops. Note that for these no
                tensorboard logging ist used but a custom image saver.
            logs (dict): Logs to std out via logger.
            graph (tf.Graph): Current graph.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        """

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
            fetches["logging"] = self.log_ops

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results["global_step"]

            for key in self.scalar_keys:
                value = retrieve(last_results, key)
                self.tb_logger.add_scalar(key, value, step)

            for key in self.histogram_keys:
                value = retrieve(last_results, key)
                self.tb_logger.add_histogram(key, value, step)

            for key in self.image_keys:
                value = retrieve(last_results, key)

                name = key.split("/")[-1]
                full_name = name + "_{:07}.png".format(step)
                save_path = os.path.join(self.root, full_name)
                plot_batch(value, save_path)

            for key in self.log_keys:
                value = retrieve(last_results, key)
                self.logger.info("{}: {}".format(key, value))


class ToNumpyHook(Hook):
    """Converts all pytorch Variables and Tensors in the results to numpy
    arrays and leaves the rest as is."""

    def after_step(self, step, results):
        def convert(var_or_tens):
            if hasattr(var_or_tens, "cpu"):
                var_or_tens = var_or_tens.cpu()

            if isinstance(var_or_tens, torch.autograd.Variable):
                return var_or_tens.data.numpy()
            elif isinstance(var_or_tens, torch.Tensor):
                return var_or_tens.numpy()
            else:
                return var_or_tens

        walk(results, convert, inplace=True)


class ToTorchHook(Hook):
    """Converts all numpy arrays in the batch to torch.Tensor
    arrays and leaves the rest as is."""

    def __init__(self, push_to_gpu=True, dtype=torch.float):
        self.use_gpu = push_to_gpu
        self.dtype = dtype
        self.logger = get_logger(self)

    def before_step(self, step, fetches, feeds, batch):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                try:
                    obj = torch.tensor(obj)
                    obj = obj.to(self.dtype)
                    if self.use_gpu:
                        obj = obj.cuda()
                    return obj
                except Exception:
                    return obj
            else:
                return obj

        walk(feeds, convert, inplace=True)


class ToFromTorchHook(ToNumpyHook, ToTorchHook):
    def __init__(self, *args, **kwargs):
        ToTorchHook.__init__(self, *args, **kwargs)


class DataPrepHook(ToFromTorchHook):
    """
    The hook is needed in order to convert the input appropriately. Here, we have
    to reshape the input i.e. append 1 to the shape (for the number of channels
    of  the image). Plus, it converts to data to Pytorch tensors, and back.
    """

    def before_step(self, step, fetches, feeds, batch):
        """
        Steps taken before the training step.
        :param step: Training step.
        :param fetches: Fetches for the next session.run call.
        :param feeds: Feeds for the next session.run call.
        :param batch: The batch to be iterated over.
        :return:
        """

        def to_image(obj):
            if isinstance(obj, np.ndarray) and len(obj.shape) == 3:
                batches, height, width = obj.shape
                obj = obj.reshape(batches, height, width, 1)
            if isinstance(obj, np.ndarray) and len(obj.shape) == 4:
                return obj.transpose(0, 3, 1, 2)
            else:
                return obj

        walk(feeds, to_image, inplace=True)

        super().before_step(step, fetches, feeds, batch)

    def after_step(self, step, results):
        """
        Steps taken after the training step.
        :param step: Training step.
        :param results: Result of the session.
        :return:
        """
        super().after_step(step, results)

        def to_image(k, obj):
            if (
                "weights" not in k
                and isinstance(obj, np.ndarray)
                and len(obj.shape) == 4
            ):
                return obj.transpose(0, 2, 3, 1)
            else:
                return obj

        walk(results, to_image, inplace=True, pass_key=True)

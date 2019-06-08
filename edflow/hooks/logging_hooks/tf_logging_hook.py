import tensorflow as tf
import os
import time

from edflow.hooks.hook import Hook
from edflow.hooks.checkpoint_hooks.common import get_checkpoint_files
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch

import sys

"""TensorFlow hooks useful during training."""


class LoggingHook(Hook):
    """Supply and evaluate logging ops at an intervall of training steps."""

    def __init__(
        self,
        scalars={},
        histograms={},
        images={},
        logs={},
        graph=None,
        interval=100,
        root_path="logs",
    ):
        """Args:
            scalars (dict): Scalar ops.
            histograms (dict): Histogram ops.
            images (dict): Image ops. Note that for these no
                tensorboard logging ist used but a custom image saver.
            logs (dict): Logs to std out via logger.
            graph (tf.Graph): Current graph.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        """

        scalars = [tf.summary.scalar(n, s) for n, s in scalars.items()]
        histograms = [tf.summary.histogram(n, h) for n, h in histograms.items()]

        self._has_summary = len(scalars + histograms) > 0
        if self._has_summary:
            summary_op = tf.summary.merge(scalars + histograms)
        else:
            summary_op = tf.no_op()

        self.fetch_dict = {"summaries": summary_op, "logs": logs, "images": images}

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
            fetches["logging"] = self.fetch_dict

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results["global_step"]
            last_results = last_results["logging"]
            if self._has_summary:
                summary = last_results["summaries"]
                self.writer.add_summary(summary, step)

            logs = last_results["logs"]
            for name in sorted(logs.keys()):
                self.logger.info("{}: {}".format(name, logs[name]))

            for name, image_batch in last_results["images"].items():
                full_name = name + "_{:07}.png".format(step)
                save_path = os.path.join(self.root, full_name)
                plot_batch(image_batch, save_path)

            self.logger.info("project root: {}".format(self.root))


class ImageOverviewHook(Hook):
    def __init__(
        self,
        images={},
        graph=None,
        interval=100,
        root_path="logs",
        log_images_to_tensorboard=False
    ):
        """
        Logs an overview of all image outputs at an intervall of training steps.
        Args:
            scalars (dict): Scalar ops.
            histograms (dict): Histogram ops.
            images (dict): Image ops. Note that for these no
                tensorboard logging ist used but a custom image saver.
            logs (dict): Logs to std out via logger.
            graph (tf.Graph): Current graph.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        """

        summary_op = tf.no_op()
        self.log_images_to_tensorboard = log_images_to_tensorboard
        # TODO: actually implement this functionality

        self.fetch_dict = {"summaries": summary_op, "images": images}

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
            pass

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            step = last_results["global_step"]

            ## dirty hack
            # TODO: fix hard-coded font type
            # TODO: add option to switch between summary only and images only and both
            batches = []
            fnt = ImageFont.truetype('LiberationMono-Regular.ttf', 20) # TODO fontsize
            for name in sorted(last_results['images'].keys()):
                full_name = name + "_{:07}.png".format(step)
                save_path = os.path.join(self.root, full_name)
                im = Image.open(save_path)
                im.thumbnail((512, 512), Image.ANTIALIAS)
                d = ImageDraw.Draw(im)
                d.text((10, 10), name, fill=(255, 0, 0), font=fnt)
                batches.append(im)

            im = Image.new("RGB", batches[0].size, color = (0, 0, 0))
            fnt = ImageFont.truetype('LiberationMono-Regular.ttf', 50)
            d = ImageDraw.Draw(im)
            d.text((10, 10), "epoch\n{:07d}".format(step), fill=(255, 0, 0), font=fnt)
            batches.append(im)
            batch = np.stack(batches, axis=0) / 255.0 * 2 - 1.0
            out_path = os.path.join(self.root, "overview_{:07d}.png".format(step))
            plot_batch(batch, out_path)

            self.logger.info("project root: {}".format(self.root))
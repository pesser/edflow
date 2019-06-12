import os
import tensorflow as tf
from edflow.hooks.hook import Hook
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.custom_logging import get_logger


class RestoreModelHook(Hook):
    """Restores a TensorFlow model from a checkpoint at each epoch. Can also
    be used as a functor."""

    def __init__(
        self,
        variables,
        checkpoint_path,
        filter_cond=lambda c: True,
        global_step_setter=None,
    ):
        """Args:
            variables (list): tf.Variable to be loaded from the checkpoint.
            checkpoint_path (str): Directory in which the checkpoints are
                stored or explicit checkpoint. Ignored if used as functor.
            filter_cond (Callable): A function used to filter files, to only
                get the checkpoints that are wanted. Ignored if used as
                functor.
            global_step_setter (Callable): Callback to set global_step.
        """
        self.root = checkpoint_path
        self.fcond = filter_cond
        self.setstep = global_step_setter

        self.logger = get_logger(self)

        self.saver = tf.train.Saver(variables)

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = tf.get_default_session()
        return self._session

    def before_epoch(self, ep):
        # checkpoint = tf.train.latest_checkpoint(self.root)
        checkpoint = get_latest_checkpoint(self.root, self.fcond)
        self(checkpoint)

    def __call__(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        self.logger.info("Restored model from {}".format(checkpoint))
        global_step = self.parse_global_step(checkpoint)
        self.logger.info("Global step: {}".format(global_step))
        if self.setstep is not None:
            self.setstep(global_step)

    @staticmethod
    def parse_global_step(checkpoint):
        global_step = int(checkpoint.rsplit("-", 1)[1])
        return global_step


# Simple renaming for consistency
# Todo: Make the Restore op part of the model (issue #2)
# https://bitbucket.org/jhaux/edflow/issues/2/make-a-general-model-restore-hook
RestoreTFModelHook = RestoreModelHook


class CheckpointHook(Hook):
    """Does that checkpoint thingy where it stores everything in a
    checkpoint."""

    def __init__(
        self,
        root_path,
        variables,
        modelname="model",
        session=None,
        step=None,
        interval=None,
        max_to_keep=5,
    ):
        """Args:
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
        """

        self.root = root_path
        self.interval = interval
        self.step = step if step is not None else tf.train.get_global_step()

        self.saver = tf.train.Saver(variables, max_to_keep=max_to_keep)
        self.logger = get_logger(self)

        os.makedirs(root_path, exist_ok=True)
        self.savename = os.path.join(root_path, "{}.ckpt".format(modelname))

        self.session = session

    def before_epoch(self, ep):
        if self.session is None:
            self.session = tf.get_default_session()

    def after_epoch(self, epoch):
        if self.interval is None:
            self.save()

    def after_step(self, step, last_results):
        if self.interval is not None and self.global_step() % self.interval == 0:
            self.save()

    def at_exception(self, *args, **kwargs):
        self.save()

    def save(self):
        global_step = self.global_step()
        self.saver.save(self.session, self.savename, global_step=global_step)
        self.logger.info("Saved model to {}".format(self.savename))

    def global_step(self):
        if isinstance(self.step, tf.Tensor) or isinstance(self.step, tf.Variable):
            global_step = self.step
        else:
            global_step = self.step()
        return global_step


class RetrainHook(Hook):
    """Restes the global step at the beginning of training."""

    def __init__(self, global_step=None):
        """Args:
            global_step (tf.Variable): Variable tracking the training step.
        """

        self.global_step = global_step
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        self.epoch = epoch

    def before_step(self, batch_index, fetches, feeds, batch):
        if self.epoch == 0 and batch_index == 0:
            fetches["reset_step"] = tf.assign(self.global_step, 0)

    def after_step(self, step, *args, **kwargs):
        if step == 0 and self.epoch == 0:
            self.logger.info("Reset global_step")


class WaitForManager(Hook):
    """Wait to make sure checkpoints are not overflowing."""

    def __init__(self, checkpoint_root, max_n, interval=5):
        """Args:
            checkpoint_root (str): Path to look for checkpoints.
            max_n (int): Wait as long as there are more than max_n ckpts.
            interval (float): Number of seconds after which to check for number
                of checkpoints again.
        """

        self.root = checkpoint_root
        self.max_n = max_n
        self.sleep_interval = interval

        self.logger = get_logger(self)

    def wait(self):
        """Loop until the number of checkpoints got reduced."""
        while True:
            n_ckpts = len(get_checkpoint_files(self.root))
            if n_ckpts <= self.max_n:
                break
            self.logger.info(
                "Found {} checkpoints.".format(n_ckpts)
                + "Waiting until one is removed."
            )
            time.sleep(self.sleep_interval)

    def before_epoch(self, ep):
        self.wait()


class RestoreCurrentCheckpointHook(RestoreModelHook):
    """Restores a TensorFlow model from a checkpoint at each epoch. Can also
    be used as a functor."""

    def before_epoch(self, ep):
        checkpoint = self.root
        self(checkpoint)

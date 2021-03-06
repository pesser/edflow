import os

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint


class LambdaCheckpointHook(Hook):
    """ """

    def __init__(
        self,
        root_path,
        global_step_getter,
        global_step_setter,
        save,
        restore,
        interval=None,
        ckpt_zero=False,
        modelname="model",
    ):
        """
        Parameters
        ----------
        """
        self.root = root_path
        self.logger = get_logger(self)
        self.global_step_getter = global_step_getter
        self.global_step_setter = global_step_setter
        self._save = save
        self._restore = restore
        self.interval = interval
        self.ckpt_zero = ckpt_zero

        os.makedirs(root_path, exist_ok=True)
        self.savename = os.path.join(root_path, "{}-{{}}.ckpt".format(modelname))
        self._active = False

    def before_epoch(self, epoch):
        """

        Parameters
        ----------
        epoch :


        Returns
        -------

        """
        if self.ckpt_zero and self.global_step_getter() == 0:
            self.save(force_active=True)

    def after_epoch(self, epoch):
        """

        Parameters
        ----------
        epoch :


        Returns
        -------

        """
        self._active = True
        if self.interval is None:
            self.save()

    def after_step(self, step, last_results):
        """

        Parameters
        ----------
        step :

        last_results :


        Returns
        -------

        """
        if step > 0:
            self._active = True
        step = self.global_step_getter()
        if self.interval is not None and step % self.interval == 0:
            self.save()

    def at_exception(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :

        **kwargs :


        Returns
        -------

        """
        self.save()

    def save(self, force_active=False):
        """ """
        if self._active or force_active:
            savename = self.savename.format(self.global_step_getter())
            self._save(savename)
            self.logger.info("Saved model to {}".format(savename))

    def __call__(self, checkpoint):
        """Load checkpoint and set global step."""
        self._restore(checkpoint)
        self.logger.info("Restored model from {}".format(checkpoint))

        step = self.parse_global_step(checkpoint)

        if self.global_step_setter is not None:
            self.global_step_setter(step)
        self.logger.info("Global step: {}".format(step))

    @staticmethod
    def parse_global_step(checkpoint):
        """

        Parameters
        ----------
        checkpoint :


        Returns
        -------

        """
        checkpoint = checkpoint.rsplit(".ckpt", 1)[0]
        global_step = int(checkpoint.rsplit("-", 1)[1])
        return global_step

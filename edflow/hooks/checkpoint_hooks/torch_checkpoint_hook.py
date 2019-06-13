import torch, os

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint


class RestorePytorchModelHook(Hook):
    """Restores a PyTorch model from a checkpoint at each epoch. Can also be
    used as a functor."""

    def __init__(
        self,
        model,
        checkpoint_path,
        filter_cond=lambda c: True,
        global_step_setter=None,
    ):
        """Args:
            model (torch.nn.Module): Model to initialize
            checkpoint_path (str): Directory in which the checkpoints are
                stored or explicit checkpoint. Ignored if used as functor.
            filter_cond (Callable): A function used to filter files, to only
                get the checkpoints that are wanted. Ignored if used as
                functor.
            global_step_setter (Callable): Function, that the retrieved global
                step can be passed to.
        """
        self.root = checkpoint_path
        self.fcond = filter_cond

        self.logger = get_logger(self)

        self.model = model
        self.global_step_setter = global_step_setter

    def before_epoch(self, ep):
        checkpoint = get_latest_checkpoint(self.root, self.fcond)
        self(checkpoint)

    def __call__(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))
        self.logger.info("Restored model from {}".format(checkpoint))

        epoch, step = self.parse_checkpoint(checkpoint)

        if self.global_step_setter is not None:
            self.global_step_setter(step)
        self.logger.info("Epoch: {}, Global step: {}".format(epoch, step))

    @staticmethod
    def parse_global_step(checkpoint):
        return RestorePytorchModelHook.parse_checkpoint(checkpoint)[1]

    @staticmethod
    def parse_checkpoint(checkpoint):
        e_s = os.path.basename(checkpoint).split(".")[0].split("-")
        if len(e_s) > 1:
            epoch = e_s[0]
            step = e_s[1].split("_")[0]
        else:
            epoch = 0
            step = e_s[0].split("_")[0]

        return int(epoch), int(step)

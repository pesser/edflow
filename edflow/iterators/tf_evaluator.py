import tensorflow as tf
import os

from edflow.iterators.tf_iterator import HookedModelIterator, TFHookedModelIterator
from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import (
    RestoreModelHook,
    RestoreTFModelHook,
    RestoreCurrentCheckpointHook,
)
from edflow.project_manager import ProjectManager

P = ProjectManager()


class BaseEvaluator(HookedModelIterator):
    """Runs evaluations on a test model. Adds support for continuous
    evaluation of new checkpoints."""

    def __init__(
        self,
        config,
        root_path,
        model,
        hooks=[],
        hook_freq=100,
        bar_position=0,
        **kwargs
    ):
        """Args:
            config (str): Some config file. (TODO: make this into kwargs)
            root_path (str): Root directory to store all eval outputs.
            model (Model): :class:`Model` to evaluate.
            hooks (list): List of :class:'`Hook`s
            logger (logging.Logger): Logger instance for logging log stuff.
            hook_freq (int): Step frequencey at which hooks are evaluated.
            bar_position (int): Used by tqdm to place bars at the right
                position when using multiple Iterators in parallel.
        """
        super().__init__(model, 1, hooks, hook_freq, bar_position, **kwargs)

        self.config = config
        self.root = root_path

        self.setup()

    def setup(self):
        """Define ops neccessary for step op."""
        restore_variables = [
            v for v in tf.global_variables() if not "__noinit__" in v.name
        ]
        # Prepend those hooks
        check_dir = P.checkpoints
        W = WaitForCheckpointHook(check_dir, self.model.model_name)
        R = RestoreModelHook(restore_variables, check_dir)
        self.hooks = [W, R] + self.hooks

    def step_ops(self):
        """Evaluate the model!!!!"""
        return self.model.outputs


class TFBaseEvaluator(TFHookedModelIterator):
    def __init__(self, *args, desc="Eval", hook_freq=1, num_epochs=1, **kwargs):
        """
        New Base evaluator restores given checkpoint path if provided,
        else scans checkpoint directory for latest checkpoint and uses that

        Parameters
        ----------
        desc : str
            a description for the evaluator. This description will be used during the logging.
        hook_freq : int
            Frequency at which hooks are evaluated.
        num_epochs : int
            Number of times to iterate over the data.
        """
        kwargs.update({"desc": desc, "hook_freq": hook_freq, "num_epochs": num_epochs})
        super().__init__(*args, **kwargs)
        self.define_graph()

    def initialize(self, checkpoint_path=None):
        self.restore_variables = self.model.variables
        # wait for new checkpoint and restore
        if checkpoint_path:
            restorer = RestoreCurrentCheckpointHook(
                variables=self.restore_variables,
                checkpoint_path=checkpoint_path,
                global_step_setter=self.set_global_step,
            )
            self.hooks += [restorer]
        else:
            restorer = RestoreTFModelHook(
                variables=self.restore_variables,
                checkpoint_path=ProjectManager.checkpoints,
                global_step_setter=self.set_global_step,
            )
            waiter = WaitForCheckpointHook(
                checkpoint_root=ProjectManager.checkpoints,
                callback=restorer,
                eval_all=self.config.get("eval_all", False),
            )
            self.hooks += [waiter]

    def define_graph(self):
        pass

    def step_ops(self):
        return self.model.outputs

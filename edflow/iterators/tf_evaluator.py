import tensorflow as tf
import os

from edflow.iterators.tf_iterator import TFHookedModelIterator
from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import (
    RestoreModelHook,
    RestoreTFModelHook,
    RestoreCurrentCheckpointHook,
)
from edflow.project_manager import ProjectManager

P = ProjectManager()


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
                filter_cond=eval(self.config.get("fcond", "lambda c: True")),
            )
            self.hooks += [waiter]

    def define_graph(self):
        pass

    def step_ops(self):
        return self.model.outputs

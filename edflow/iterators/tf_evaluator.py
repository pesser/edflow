import tensorflow as tf
import os

from edflow.iterators.tf_iterator import HookedModelIterator, TFHookedModelIterator
from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import (
    RestoreModelHook,
    RestoreTFModelHook,
    RestoreCurrentCheckpointHook
)
from edflow.project_manager import ProjectManager
from deprecated import deprecated


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
    @deprecated(version="0.2.0", reason="Does not restore checkpoint path when provided. Use TFBaseEvaluator2 instead")
    def __init__(self, *args, **kwargs):
        kwargs["desc"] = "Eval"
        kwargs["hook_freq"] = 1
        kwargs["num_epochs"] = 1
        super().__init__(*args, **kwargs)

        # wait for new checkpoint and restore
        self.restore_variables = self.model.variables
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

    def step_ops(self):
        return self.model.outputs


class TFBaseEvaluator2(TFHookedModelIterator):
    def __init__(self, *args, checkpoint_path=None, **kwargs):
        '''
        New Base evaluator restores given checkpoint path if provided,
        else scans checkpoint directory for latest checkpoint and uses that

        Parameters
        ----------
        args
        checkpoint_path
        kwargs
        '''
        kwargs['desc'] = 'Eval'
        kwargs['hook_freq'] = 1
        kwargs["num_epochs"] = 1
        super().__init__(*args, **kwargs)

        # wait for new checkpoint and restore
        self.restore_variables = self.model.variables
        if checkpoint_path:
            restorer = RestoreCurrentCheckpointHook(variables=self.restore_variables,
                                                    checkpoint_path=checkpoint_path,
                                                    global_step_setter=self.set_global_step)
            self.hooks += [restorer]
        else:
            restorer = RestoreTFModelHook(variables=self.restore_variables,
                                          checkpoint_path=ProjectManager.checkpoints,
                                          global_step_setter=self.set_global_step)
            waiter = WaitForCheckpointHook(checkpoint_root=ProjectManager.checkpoints,
                                           callback=restorer,
                                           eval_all=self.config.get("eval_all", False))
            self.hooks += [waiter]


    def step_ops(self):
        return self.model.outputs
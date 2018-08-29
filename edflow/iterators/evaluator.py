import tensorflow as tf
import os

from edflow.iterators.model_iterator import HookedModelIterator
from edflow.hooks import WaitForCheckpointHook, RestoreModelHook
from edflow.project_manager import ProjectManager


P = ProjectManager()


class BaseEvaluator(HookedModelIterator):
    '''Runs evaluations on a test model. Adds support for continuous
    evaluation of new checkpoints.'''

    def __init__(self,
                 config,
                 root_path,
                 model,
                 hooks=[],
                 hook_freq=100,
                 bar_position=0,
                 **kwargs):
        '''Args:
            config (str): Some config file. (TODO: make this into kwargs)
            root_path (str): Root directory to store all eval outputs.
            model (Model): :class:`Model` to evaluate.
            hooks (list): List of :class:'`Hook`s
            logger (logging.Logger): Logger instance for logging log stuff.
            hook_freq (int): Step frequencey at which hooks are evaluated.
            bar_position (int): Used by tqdm to place bars at the right
                position when using multiple Iterators in parallel.
        '''
        super().__init__(model, 1, hooks, hook_freq, bar_position, **kwargs)

        self.config = config
        self.root = root_path

        self.setup()

    def setup(self):
        '''Define ops neccessary for step op.'''
        restore_variables = [v for v in tf.global_variables() if not "__noinit__" in v.name]
        # Prepend those hooks
        check_dir = P.checkpoints
        W = WaitForCheckpointHook(check_dir, self.model.model_name)
        R = RestoreModelHook(restore_variables, check_dir)
        self.hooks = [W, R] + self.hooks

    def step_ops(self):
        '''Evaluate the model!!!!'''
        return self.model.outputs

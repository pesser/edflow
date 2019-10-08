from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.hooks.logging_hooks.minimal_logging_hook import LoggingHook
from edflow.hooks.util_hooks import IntervalHook
from edflow.eval.pipeline import TemplateEvalHook
from edflow.project_manager import ProjectManager
from edflow.util import retrieve, set_default
from edflow.main import get_obj_from_str


class TemplateIterator(PyHookedModelIterator):
    """A specialization of PyHookedModelIterator which adds reasonable default
    behaviour. Subclasses should implement `save`, `restore` and `step_op`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # wrap save and restore into a LambdaCheckpointHook
        self.ckpthook = LambdaCheckpointHook(
            root_path=ProjectManager.checkpoints,
            global_step_getter=self.get_global_step,
            global_step_setter=self.set_global_step,
            save=self.save,
            restore=self.restore,
            interval=set_default(self.config, "ckpt_freq", None),
        )
        if not self.config.get("test_mode", False):
            # in training, excute train ops and add logginghook
            self._train_ops = set_default(
                self.config, "train_ops", ["step_ops/train_op"]
            )
            self._log_ops = set_default(self.config, "log_ops", ["step_ops/log_op"])
            # logging
            self.loghook = LoggingHook(
                paths=self._log_ops, root_path=ProjectManager.train, interval=1
            )
            # wrap it in interval hook
            self.ihook = IntervalHook(
                [self.loghook],
                interval=set_default(self.config, "start_log_freq", 1),
                modify_each=1,
                max_interval=set_default(self.config, "log_freq", 1000),
                get_step=self.get_global_step,
            )
            self.hooks.append(self.ihook)
            # write checkpoints after epoch or when interrupted
            self.hooks.append(self.ckpthook)
        else:
            # evaluate
            self._eval_op = set_default(
                self.config, "eval_hook/eval_op", "step_ops/eval_op"
            )
            self._eval_callbacks = set_default(
                self.config, "eval_hook/eval_callbacks", dict()
            )
            if not isinstance(self._eval_callbacks, dict):
                self._eval_callbacks = {"cb": self._eval_callbacks}
            for k in self._eval_callbacks:
                self._eval_callbacks[k] = get_obj_from_str(self._eval_callbacks[k])
            label_key = set_default(
                self.config, "eval_hook/label_key", "step_ops/eval_op/labels"
            )
            self.evalhook = TemplateEvalHook(
                dataset=self.dataset,
                step_getter=self.get_global_step,
                keypath=self._eval_op,
                config=self.config,
                callbacks=self._eval_callbacks,
                labels_key=label_key,
            )
            self.hooks.append(self.evalhook)
            self._train_ops = []
            self._log_ops = []

    def initialize(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.ckpthook(checkpoint_path)

    def step_ops(self):
        return self.step_op

    def run(self, fetches, feed_dict):
        results = super().run(fetches, feed_dict)
        for train_op in self._train_ops:
            retrieve(results, train_op)
        return results

    def save(self, checkpoint_path):
        """Save state to checkpoint path."""
        raise NotImplemented()

    def restore(self, checkpoint_path):
        """Restore state from checkpoint path."""
        raise NotImplemented()

    def step_op(self, model, **kwargs):
        """Actual step logic. By default, a dictionary with keys 'train_op',
        'log_op', 'eval_op' and callable values is expected. 'train_op' should
        update the model's state as a side-effect, 'log_op' will be logged to
        the project's train folder. It should be a dictionary with keys
        'images' and 'scalars'. Images are written as png's, scalars are
        written to the log file and stdout. Outputs of 'eval_op' are written
        into the project's eval folder to be evaluated with `edeval`."""
        raise NotImplemented()

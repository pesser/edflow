import os
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.hooks.logging_hooks.minimal_logging_hook import LoggingHook
from edflow.hooks.util_hooks import IntervalHook, ExpandHook
from edflow.eval.pipeline import TemplateEvalHook
from edflow.project_manager import ProjectManager
from edflow.util import (
    retrieve,
    set_default,
    set_value,
    get_obj_from_str,
    get_str_from_obj,
)


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
            ckpt_zero=set_default(self.config, "ckpt_zero", False),
        )
        # write checkpoints after epoch or when interrupted during training
        if not self.config.get("test_mode", False):
            self.hooks.append(self.ckpthook)

        ## hooks - disabled unless -t is specified

        # execute train ops
        self._train_ops = set_default(self.config, "train_ops", ["train/train_op"])
        train_hook = ExpandHook(paths=self._train_ops, interval=1)
        self.hooks.append(train_hook)

        # log train/step_ops/log_ops in increasing intervals
        self._log_ops = set_default(
            self.config, "log_ops", ["train/log_op", "validation/log_op"]
        )
        self.loghook = LoggingHook(
            paths=self._log_ops, root_path=ProjectManager.train, interval=1
        )
        self.ihook = IntervalHook(
            [self.loghook],
            interval=set_default(self.config, "start_log_freq", 1),
            modify_each=1,
            max_interval=set_default(self.config, "log_freq", 1000),
            get_step=self.get_global_step,
        )
        self.hooks.append(self.ihook)

        # setup logging integrations
        if not self.config.get("test_mode", False):
            default_wandb_logging = {"active": False, "handlers": ["scalars", "images"]}
            wandb_logging = set_default(
                self.config, "integrations/wandb", default_wandb_logging
            )
            if wandb_logging["active"]:
                import wandb
                from edflow.hooks.logging_hooks.wandb_handler import (
                    log_wandb,
                    log_wandb_images,
                )

                os.environ["WANDB_RESUME"] = "allow"
                os.environ["WANDB_RUN_ID"] = ProjectManager.root.strip("/").replace(
                    "/", "-"
                )
                wandb_project = set_default(
                    self.config, "integrations/wandb/project", None
                )
                wandb_entity = set_default(
                    self.config, "integrations/wandb/entity", None
                )
                wandb.init(
                    name=ProjectManager.root,
                    config=self.config,
                    project=wandb_project,
                    entity=wandb_entity,
                )

                handlers = set_default(
                    self.config,
                    "integrations/wandb/handlers",
                    default_wandb_logging["handlers"],
                )
                if "scalars" in handlers:
                    self.loghook.handlers["scalars"].append(log_wandb)
                if "images" in handlers:
                    self.loghook.handlers["images"].append(log_wandb_images)

            default_tensorboard_logging = {
                "active": False,
                "handlers": ["scalars", "images", "figures"],
            }
            tensorboard_logging = set_default(
                self.config, "integrations/tensorboard", default_tensorboard_logging
            )
            if tensorboard_logging["active"]:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except:
                    from tensorboardX import SummaryWriter

                from edflow.hooks.logging_hooks.tensorboard_handler import (
                    log_tensorboard_config,
                    log_tensorboard_scalars,
                    log_tensorboard_images,
                    log_tensorboard_figures,
                )

                self.tensorboard_writer = SummaryWriter(ProjectManager.root)
                log_tensorboard_config(
                    self.tensorboard_writer, self.config, self.get_global_step()
                )
                handlers = set_default(
                    self.config,
                    "integrations/tensorboard/handlers",
                    default_tensorboard_logging["handlers"],
                )
                if "scalars" in handlers:
                    self.loghook.handlers["scalars"].append(
                        lambda *args, **kwargs: log_tensorboard_scalars(
                            self.tensorboard_writer, *args, **kwargs
                        )
                    )
                if "images" in handlers:
                    self.loghook.handlers["images"].append(
                        lambda *args, **kwargs: log_tensorboard_images(
                            self.tensorboard_writer, *args, **kwargs
                        )
                    )
                if "figures" in handlers:
                    self.loghook.handlers["figures"].append(
                        lambda *args, **kwargs: log_tensorboard_figures(
                            self.tensorboard_writer, *args, **kwargs
                        )
                    )
        ## epoch hooks

        # evaluate validation/step_ops/eval_op after each epoch
        self._eval_op = set_default(
            self.config, "eval_hook/eval_op", "validation/eval_op"
        )
        _eval_callbacks = set_default(self.config, "eval_hook/eval_callbacks", dict())
        if not isinstance(_eval_callbacks, dict):
            _eval_callbacks = {"cb": _eval_callbacks}
        eval_callbacks = dict()
        for k in _eval_callbacks:
            eval_callbacks[k] = _eval_callbacks[k]
        if hasattr(self, "callbacks"):
            iterator_callbacks = retrieve(self.callbacks, "eval_op", default=dict())
            for k in iterator_callbacks:
                import_path = get_str_from_obj(iterator_callbacks[k])
                set_value(
                    self.config, "eval_hook/eval_callbacks/{}".format(k), import_path
                )
                eval_callbacks[k] = import_path
        if hasattr(self.model, "callbacks"):
            model_callbacks = retrieve(self.model.callbacks, "eval_op", default=dict())
            for k in model_callbacks:
                import_path = get_str_from_obj(model_callbacks[k])
                set_value(
                    self.config, "eval_hook/eval_callbacks/{}".format(k), import_path
                )
                eval_callbacks[k] = import_path
        callback_handler = None
        if not self.config.get("test_mode", False):
            callback_handler = lambda results, paths: self.loghook(
                results=results, step=self.get_global_step(), paths=paths,
            )

        # offer option to run eval functor:
        # overwrite step op to only include the evaluation of the functor and
        # overwrite callbacks to only include the callbacks of the functor
        if self.config.get("test_mode", False) and "eval_functor" in self.config:
            # offer option to use eval functor for evaluation
            eval_functor = get_obj_from_str(self.config["eval_functor"])(
                config=self.config
            )
            self.step_ops = lambda: {"eval_op": eval_functor}
            eval_callbacks = dict()
            if hasattr(eval_functor, "callbacks"):
                for k in eval_functor.callbacks:
                    eval_callbacks[k] = get_str_from_obj(eval_functor.callbacks[k])
            set_value(self.config, "eval_hook/eval_callbacks", eval_callbacks)
        self.evalhook = TemplateEvalHook(
            datasets=self.datasets,
            step_getter=self.get_global_step,
            keypath=self._eval_op,
            config=self.config,
            callbacks=eval_callbacks,
            callback_handler=callback_handler,
        )
        self.epoch_hooks.append(self.evalhook)

    def initialize(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.ckpthook(checkpoint_path)

    def step_ops(self):
        return self.step_op

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

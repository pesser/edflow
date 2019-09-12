import tensorflow as tf
import numpy as np
import copy
from edflow.iterators.tf_iterator import TFHookedModelIterator

from edflow.hooks.logging_hooks.tf_logging_hook import LoggingHook
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import (
    CheckpointHook,
    RetrainHook,
    RestoreTFModelHook,
)
from edflow.tf_util import make_linear_var

from edflow.project_manager import ProjectManager
from edflow.hooks.util_hooks import IntervalHook

P = ProjectManager()


class TFBaseTrainer(TFHookedModelIterator):
    """Same but based on TFHookedModelIterator."""

    def __init__(self, config, root, model, **kwargs):
        super().__init__(config, root, model, **kwargs)
        self.setup()

    def initialize(self, checkpoint_path=None):
        """Initialize from scratch or restore and keep restorer around."""
        init_op = tf.variables_initializer(self.get_init_variables())
        self.session.run(init_op)
        if checkpoint_path is not None:
            restorer = RestoreTFModelHook(
                variables=self.get_restore_variables(),
                checkpoint_path=ProjectManager.checkpoints,
                global_step_setter=self.set_global_step,
            )
            with self.session.as_default():
                restorer(checkpoint_path)

    def step_ops(self):
        return self.train_op

    def make_feeds(self, batch):
        """Put global step into batches and add all extra required placeholders
        from batches."""
        feeds = super().make_feeds(batch)
        batch["global_step"] = self.get_global_step()
        for k, v in self.train_placeholders.items():
            if k in batch:
                feeds[v] = batch[k]
        return feeds

    def setup(self):
        """Init train_placeholders, log_ops and img_ops which can be added
        to."""
        self.train_placeholders = dict()
        self.log_ops = dict()
        self.img_ops = dict()
        self.update_ops = list()
        self.create_train_op()

        ckpt_hook = CheckpointHook(
            root_path=ProjectManager.checkpoints,
            variables=self.get_checkpoint_variables(),
            modelname="model",
            step=self.get_global_step,
            interval=self.config.get("ckpt_freq", None),
            max_to_keep=self.config.get("ckpt_keep", None),
        )
        self.hooks.append(ckpt_hook)

        loghook = LoggingHook(
            logs=self.log_ops,
            scalars=self.log_ops,
            images=self.img_ops,
            root_path=ProjectManager.train,
            interval=1,
            log_images_to_tensorboard=self.config.get(
                "log_images_to_tensorboard", False
            ),
        )
        ihook = IntervalHook(
            [loghook],
            interval=self.config.get("start_log_freq", 1),
            modify_each=1,
            max_interval=self.config.get("log_freq", 1000),
            get_step=self.get_global_step,
        )
        self.hooks.append(ihook)

    def create_train_op(self):
        """Default optimizer + optimize each submodule"""
        self.train_placeholders["global_step"] = tf.placeholder(
            tf.int32, name="global_step"
        )
        # workaround of https://github.com/tensorflow/tensorflow/issues/23316
        self._global_step_variable = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.global_step = tf.assign(
            self._global_step_variable, self.train_placeholders["global_step"]
        )

        # Optimizer
        self.initial_lr = self.config["lr"]
        self.ultimate_lr = self.config.get("lr_end", self.initial_lr)
        self.lr_decay_begin = self.config.get("lr_decay_begin", 0)
        self.lr_decay_end = self.config.get(
            "lr_decay_end", self.config.get("num_steps")
        )

        self.lr = lr = make_linear_var(
            self.global_step,
            self.lr_decay_begin,
            self.lr_decay_end,
            self.initial_lr,
            self.ultimate_lr,
            min(self.ultimate_lr, self.initial_lr),
            max(self.ultimate_lr, self.initial_lr),
        )
        optimizer_name = self.config.get("optimizer", "AdamOptimizer")
        Optimizer = getattr(tf.train, optimizer_name)
        opt_kwargs = {"learning_rate": lr}
        if optimizer_name == "AdamOptimizer":
            opt_kwargs["beta1"] = self.config.get("beta1", 0.5)
            opt_kwargs["beta2"] = self.config.get("beta2", 0.9)

        self.optimizers = dict()

        # Optimization ops
        losses = self.make_loss_ops()
        opt_ops = dict()

        if hasattr(self, "slow_keys"):
            self.slow_lr = self.slow_factor * self.lr
            self.log_ops["slow_lr"] = self.slow_lr

        for k in losses:
            variables = self.get_trainable_variables(k)
            if k in getattr(self, "slow_keys", list()):
                opts = dict(opt_kwargs)
                opts["learning_rate"] = self.slow_lr
                self.optimizers[k] = Optimizer(**opts)
            else:
                self.optimizers[k] = Optimizer(**opt_kwargs)
            opt_ops[k] = self.optimizers[k].minimize(losses[k], var_list=variables)
            print(k)
            print("============================")
            print(
                "\n".join(
                    [
                        "{:>22} {:>22} {}".format(
                            str(v.shape.as_list()),
                            str(np.prod(v.shape.as_list())),
                            v.name,
                        )
                        for v in variables
                    ]
                )
            )
            print(len(variables))
            print("============================")
        self.opt_ops = opt_ops

        opt_op = tf.group(*opt_ops.values())
        with tf.control_dependencies([opt_op] + self.update_ops):
            train_op = tf.no_op()
        self.train_op = train_op

        self.run_once_op = self.make_run_once_op()

        # add log ops
        self.log_ops["global_step"] = self.global_step
        self.log_ops["lr"] = self.lr

    def make_loss_ops(self):
        """Return per submodule loss. Can add tensors to log_ops and img_ops"""
        raise NotImplemented()

    def make_run_once_op(self):
        """Return op to be run at step zero. Used for custom initialization
        etc."""
        return tf.no_op()

    def get_trainable_variables(self, submodule):
        trainable_variables = [
            v for v in tf.trainable_variables() if v in self.model.variables
        ]
        return [v for v in trainable_variables if submodule in v.name]

    def get_init_variables(self):
        return [v for v in tf.global_variables() if "__noinit__" not in v.name]

    def get_restore_variables(self):
        return self.get_init_variables()

    def get_checkpoint_variables(self):
        return self.get_init_variables()

    def run(self, fetches, feed_dict):
        if self.get_global_step() == 0:
            self.logger.info("Running custom initialization op.")
            fetches["step_ops"] = self.run_once_op
        return super().run(fetches, feed_dict)


class TFFrequencyTrainer(TFBaseTrainer):
    def create_train_op(self):
        super().create_train_op()
        for k in self.opt_ops:
            self.opt_ops[k] = tf.group(self.opt_ops[k], self.update_ops)
        self.train_op = None

        self.which_opt = list()
        if type(self.opt_frequency) == dict:
            for k in self.opt_frequency:
                self.which_opt += self.opt_frequency[k] * [k]
        elif type(self.opt_frequency) == list:
            self.which_opt = self.opt_frequency
        else:
            assert False

    def run(self, fetches, feed_dict):
        current_opt = self.which_opt[self.get_global_step() % len(self.which_opt)]
        if type(current_opt) == list:
            fetches["step_ops"] = [self.opt_ops[k] for k in current_opt]
        else:
            fetches["step_ops"] = self.opt_ops[current_opt]
        return super().run(fetches, feed_dict)


class TFListTrainer(TFBaseTrainer):
    def create_train_op(self):
        """Default optimizer + optimize each submodule"""
        self.train_placeholders["global_step"] = tf.placeholder(
            tf.int32, name="global_step"
        )
        # workaround of https://github.com/tensorflow/tensorflow/issues/23316
        self._global_step_variable = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.global_step = tf.assign(
            self._global_step_variable, self.train_placeholders["global_step"]
        )

        # Optimizer
        self.initial_lr = self.config["lr"]
        self.ultimate_lr = self.config.get("lr_end", self.initial_lr)
        self.lr_decay_begin = self.config.get("lr_decay_begin", 0)
        self.lr_decay_end = self.config.get(
            "lr_decay_end", self.config.get("num_steps")
        )

        self.lr = lr = make_linear_var(
            self.global_step,
            self.lr_decay_begin,
            self.lr_decay_end,
            self.initial_lr,
            self.ultimate_lr,
            min(self.ultimate_lr, self.initial_lr),
            max(self.ultimate_lr, self.initial_lr),
        )
        optimizer_name = self.config.get("optimizer", "AdamOptimizer")
        Optimizer = getattr(tf.train, optimizer_name)
        opt_kwargs = {"learning_rate": lr}
        if optimizer_name == "AdamOptimizer":
            opt_kwargs["beta1"] = self.config.get("beta1", 0.5)
            opt_kwargs["beta2"] = self.config.get("beta2", 0.9)

        self.optimizers = dict()
        self.all_train_ops = list()

        # Optimization ops
        list_of_losses = self.make_loss_ops()
        assert type(list_of_losses) == list
        for i, losses in enumerate(list_of_losses):
            opt_ops = dict()
            optimizers = dict()

            for k in losses:
                variables = self.get_trainable_variables(k)
                current_opt_kwargs = dict(opt_kwargs)
                current_opt_kwargs["learning_rate"] = current_opt_kwargs[
                    "learning_rate"
                ] * self.get_learning_rate_multiplier(i)
                optimizers[k] = Optimizer(**current_opt_kwargs)
                opt_ops[k] = optimizers[k].minimize(losses[k], var_list=variables)
                print(i, k, self.get_learning_rate_multiplier(i))
                print("============================")
                print(
                    "\n".join(
                        [
                            "{:>22} {:>22} {}".format(
                                str(v.shape.as_list()),
                                str(np.prod(v.shape.as_list())),
                                v.name,
                            )
                            for v in variables
                        ]
                    )
                )
                print(len(variables))
                print("============================")

            opt_op = tf.group(*opt_ops.values())
            with tf.control_dependencies([opt_op] + self.update_ops):
                train_op = tf.no_op()
            self.all_train_ops.append(train_op)

        self.train_op = None  # dummy, set actual train op in run
        self.run_once_op = self.make_run_once_op()
        # add log ops
        self.log_ops["global_step"] = self.global_step
        self.log_ops["lr"] = self.lr

    def run(self, fetches, feed_dict):
        train_idx = self.get_global_step() % len(self.all_train_ops)
        fetches["step_ops"] = self.all_train_ops[train_idx]
        return super().run(fetches, feed_dict)

    def get_learning_rate_multiplier(self, i):
        return 1.0


class TFMultiStageTrainer(TFBaseTrainer):
    """Adds multistage training to Edflow Trainer

    Stages are defined through the config. For example

        stages:
          1:
            name: pretrain
            end: 10
            losses: []
          2:
            name: retrain
            end: 30
            losses: ["model"]
          3:
            name: train
            losses: ["model"]

    The stages are sorted by their key. It is recommended to keep the simple numeric ordering.
    In each stage, a set of losses can be specified through the `losses : [ "loss1", "loss2", ...]` syntax.
    The duration of each stage is given by the `end : num_steps` value. Note that the end of a stage
    is determined in the order of the stages. A later stage has to have a higher end value then the previous one.

    The model has to implement the `edflowiterators.tf_trainer.TFMultiStageModel` interface. Look at the multistage_trainer example.
    """

    def __init__(self, config, root, model, **kwargs):
        if not isinstance(model, TFMultiStageModel):
            raise ValueError("Model does not support Multistage interface")
        else:
            super(TFMultiStageTrainer, self).__init__(config, root, model, **kwargs)

    def create_train_op(self):
        super(TFMultiStageTrainer, self).create_train_op()
        self.stages = copy.deepcopy(self.config.get("stages"))
        for stage_number, stage_options in self.stages.items():
            stage_opt_ops = [
                self.opt_ops[loss_name] for loss_name in stage_options["losses"]
            ]
            stage_opt_ops = tf.group(*stage_opt_ops)
            with tf.control_dependencies([stage_opt_ops] + self.update_ops):
                stage_train_op = tf.no_op()
            stage_options.update({"train_op": stage_train_op})
        self.log_ops["stage"] = self.model.stage

    def run(self, fetches, feed_dict):
        current_stage = self.determine_current_stage()
        self.session.run(
            [self.model.stage_update_op],
            feed_dict={self.model.stage_placeholder: current_stage},
        )
        current_train_op = self.get_current_train_op(current_stage)
        if self.get_global_step() == 0:
            self.logger.info("Running custom initialization op.")
            fetches["step_ops"] = self.run_once_op
        else:
            fetches["step_ops"] = current_train_op
        get_global_step = fetches.pop("global_step")
        results = self.session.run(fetches, feed_dict=feed_dict)
        results["global_step"] = get_global_step()
        return results

    def get_current_train_op(self, current_stage):
        current_train_op = self.stages[current_stage]["train_op"]
        return current_train_op

    def determine_current_stage(self):
        global_step = self.get_global_step()
        for stage_number in sorted(self.stages.keys()):
            stage_options = self.stages[stage_number]
            end = stage_options.get("end", -1)
            if end < 0:
                return stage_number
            elif global_step < end:
                return stage_number
            else:
                pass


class TFMultiStageModel(object):
    def __init__(self):
        self._stage_update = tf.placeholder(tf.int32, shape=(), name="stage_update")
        self._stage = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._stage_update_op = tf.assign(self._stage, self._stage_update)

    @property
    def stage_update_op(self):
        return self._stage_update_op

    @property
    def stage_placeholder(self):
        return self._stage_update

    @property
    def stage(self):
        return self._stage

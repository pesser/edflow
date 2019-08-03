import functools
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.eval.pipeline import EvalHook
from edflow.project_manager import ProjectManager
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

tf.enable_eager_execution()


# ================ Generic Logging Hook ==================


from edflow.hooks.hook import Hook
from edflow.util import retrieve
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch
import os
class LoggingHook(Hook):
    """Evaluate logging ops at an interval of training steps."""
    def __init__(
        self,
        paths,
        interval,
        root_path,
    ):
        """Args:
            paths (list(str)): List of key-paths to logging outputs. Will be
                expanded so they can be evaluated lazily.
            interval (int): Intervall of training steps before logging.
            root_path (str): Path at which the logs are stored.
        """
        self.paths = paths
        self.interval = interval
        self.root = root_path
        self.logger = get_logger(self)
        self.handlers = {
                "images": self.log_images,
                "scalars": self.log_scalars}

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            self._step = last_results["global_step"]
            self.logger.info("global_step: {}".format(self._step))
            for path in self.paths:
                for k in self.handlers:
                    handler_results = retrieve(last_results, path+"/"+k, default = dict())
                    self.handlers[k](handler_results)
            self.logger.info("project root: {}".format(self.root))

    def log_scalars(self, results):
        for name in sorted(results.keys()):
            self.logger.info("{}: {}".format(name, results[name]))

    def log_images(self, results):
        for name, image_batch in results.items():
            full_name = name + "_{:07}.png".format(self._step)
            save_path = os.path.join(self.root, full_name)
            plot_batch(image_batch, save_path)


# ================ Generic Iterator ==================


from edflow.hooks.util_hooks import IntervalHook

class TemplateIterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # and wrap it into a LambdaCheckpointHook
        self.ckpthook = LambdaCheckpointHook(
            root_path=ProjectManager.checkpoints,
            global_step_getter=self.get_global_step,
            global_step_setter=self.set_global_step,
            save=self.save,
            restore=self.restore,
            interval=self.config.get("ckpt_freq", None),
        )
        if not self.config.get("test_mode", False):
            self._train_ops = self.config.get("train_ops", ["step_ops/train_op"])
            self._log_ops = self.config.get("log_ops", ["step_ops/log_op"])
            # logging
            self.loghook = LoggingHook(
                    paths=self._log_ops,
                    root_path=ProjectManager.train,
                    interval=1,
                    )
            # wrap it in interval hook
            self.ihook = IntervalHook(
                [self.loghook],
                interval=self.config.get("start_log_freq", 1),
                modify_each=1,
                max_interval=self.config.get("log_freq", 1000),
                get_step=self.get_global_step,
            )
            self.hooks.append(self.ihook)
            # write checkpoints after epoch or when interrupted
            self.hooks.append(self.ckpthook)
        else:
            # evaluate
            self._eval_op = self.config.get("eval_op", "step_ops/eval_op")
            self.evalhook = EvalHook(
                    dataset = self.dataset,
                    step_getter = self.get_global_step,
                    keypath = self._eval_op,
                    meta = self.config)
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
            retrieve(results, train_op)()
        return results

    def save(self, checkpoint_path):
        raise NotImplemented()

    def restore(self, checkpoint_path):
        raise NotImplemented()

    def step_op(self, model, **kwargs):
        raise NotImplemented()


# ================ Application Code ==================


class Model(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = tfk.layers.Conv2D(filters=6, kernel_size=5)
        self.pool = tfk.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tfk.layers.Conv2D(filters=16, kernel_size=5)
        self.fc1 = tfk.layers.Dense(units=120)
        self.fc2 = tfk.layers.Dense(units=84)
        self.fc3 = tfk.layers.Dense(units=config["n_classes"])

        input_shape = (config["batch_size"], 28, 28, 1)
        self.build(input_shape)

    def call(self, x):
        x = self.pool(tf.nn.relu(self.conv1(x)))
        x = self.pool(tf.nn.relu(self.conv2(x)))
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.criterion = functools.partial(
            tfk.losses.sparse_categorical_crossentropy, from_logits=True
        )
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        # to save and restore
        self.tfcheckpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer
        )

    def save(self, checkpoint_path):
        self.tfcheckpoint.write(checkpoint_path)

    def restore(self, checkpoint_path):
        self.tfcheckpoint.restore(checkpoint_path)

    def step_op(self, model, **kwargs):
        # get inputs
        inputs, labels = kwargs["image"], kwargs["class"]

        # compute loss
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = self.criterion(y_true=labels, y_pred=outputs)
            mean_loss = tf.reduce_mean(loss)

        def train_op():
            grads = tape.gradient(mean_loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        def log_op():
            acc = np.mean(np.argmax(outputs, axis = 1) == labels)
            min_loss = np.min(loss)
            max_loss = np.max(loss)
            return {"images": {"inputs": inputs},
                    "scalars": {
                        "min_loss": min_loss,
                        "max_loss": max_loss,
                        "mean_loss": mean_loss,
                        "acc": acc}}

        def eval_op():
            return {"outputs": np.array(outputs), "loss": np.array(loss)[:,None]}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


def acc_callback(root, data_in, data_out, config):
    from tqdm import trange
    logger = get_logger("acc_callback")
    correct = 0
    seen = 0
    loss = 0.0
    for i in trange(len(data_in)):
        labels = data_in[i]["class"]
        outputs = data_out[i]["outputs"]
        loss = data_out[i]["loss"].squeeze()

        prediction = np.argmax(outputs, axis = 0)
        correct += labels == prediction
        loss += loss
    logger.info("Loss: {}".format(loss / len(data_in)))
    logger.info("Accuracy: {}".format(correct / len(data_in)))

import functools
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.project_manager import ProjectManager
from edflow.hooks.hook import Hook
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

tf.enable_eager_execution()

class Model(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = tfk.layers.Conv2D(filters = 6, kernel_size = 5)
        self.pool = tfk.layers.MaxPool2D(pool_size = 2, strides = 2)
        self.conv2 = tfk.layers.Conv2D(filters = 16, kernel_size = 5)
        self.fc1 = tfk.layers.Dense(units = 120)
        self.fc2 = tfk.layers.Dense(units = 84)
        self.fc3 = tfk.layers.Dense(units = config["n_classes"])

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

class AccuracyHook(Hook):
    def __init__(self, iterator):
        self.logger = iterator.logger
        self.correct = 0
        self.total = 0

    def after_step(self, step, last_results):
        outputs, labels = last_results["step_ops"]
        predicted = np.argmax(outputs, axis = 1)
        self.total += predicted.shape[0]
        self.correct += np.sum(labels == predicted)

        if step % 250 == 0:
            self.logger.info(
                "Accuracy of the network on the %d step: %.2f %%"
                % (step, 100 * self.correct / self.total)
            )

    def after_epoch(self, epoch):
        self.logger.info(
            "Accuracy of the network on all test images: %.2f %%"
            % (100 * self.correct / self.total)
        )

class Iterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = functools.partial(
                tfk.losses.sparse_categorical_crossentropy,
                from_logits = True)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9)
        self.running_loss = 0.0

        # specify how to save and restore checkpoints
        self.tfcheckpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        def save(checkpoint_path):
            self.tfcheckpoint.write(checkpoint_path)
        def restore(checkpoint_path):
            self.tfcheckpoint.restore(checkpoint_path)
        # and wrap it into a LambdaCheckpointHook
        self.ckpthook = LambdaCheckpointHook(
                root_path = ProjectManager.checkpoints,
                global_step_getter = self.get_global_step,
                global_step_setter = self.set_global_step,
                save = save,
                restore = restore)
        if not self.config.get("test_mode", False):
            # write checkpoints after epoch or when interrupted
            self.hooks.append(self.ckpthook)
        else:
            # evaluate accuracy
            self.hooks.append(AccuracyHook(self))

    def initialize(self, checkpoint_path = None):
        if checkpoint_path is not None:
            self.ckpthook(checkpoint_path)

    def step_ops(self):
        if self.config.get("test_mode", False):
            return self.test_op
        else:
            return self.train_op

    def train_op(self, model, **kwargs):
        # get inputs and add channel axis to image
        inputs, labels = kwargs["image"], kwargs["class"]

        # compute loss
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = tf.reduce_mean(self.criterion(y_true = labels, y_pred = outputs))
        # optimize
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # print statistics
        self.running_loss += loss
        i = self.get_global_step()
        if i % 200 == 199:  # print every 200 mini-batches
            # use the logger instead of print to obtain both console output and
            # logging to the logfile in project directory
            self.logger.info("[%5d] loss: %.3f" % (i + 1, self.running_loss / 200))
            self.running_loss = 0.0

    def test_op(self, model, **kwargs):
        """Here we just run the model and let the hook handle the output."""
        inputs, labels = kwargs["image"], kwargs["class"]
        outputs = self.model(inputs)
        return outputs, labels

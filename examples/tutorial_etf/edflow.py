import functools
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.checkpoint_hooks.lambda_checkpoint_hook import LambdaCheckpointHook
from edflow.project_manager import ProjectManager
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

class Iterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = functools.partial(
                tfk.losses.sparse_categorical_crossentropy,
                from_logits = True)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9)
        self.running_loss = 0.0

        self.tfcheckpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
        def restore(checkpoint_path):
            self.tfcheckpoint.restore(checkpoint_path)
        def save(checkpoint_path):
            self.tfcheckpoint.save(checkpoint_path)

        self.ckpthook = LambdaCheckpointHook(
                root_path = ProjectManager.checkpoints,
                global_step_getter = self.get_global_step,
                global_step_setter = self.set_global_step,
                save = save,
                restore = restore)
        self.hooks.append(self.ckpthook)

    def initialize(self, checkpoint_path = None):
        if checkpoint_path is not None:
            self.ckpthook(checkpoint_path)

    def step_ops(self):
        return self.train_op

    def train_op(self, model, **kwargs):
        inputs, labels = kwargs["image"], kwargs["class"]
        inputs = inputs[:,:,:,None].astype(np.float32)
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = tf.reduce_mean(self.criterion(y_true = labels, y_pred = outputs))
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.running_loss += loss
        i = self.get_global_step()
        if i % 200 == 199:
            self.logger.info("[%5d] loss: %.3f" % (i + 1, self.running_loss / 200))
            self.running_loss = 0.0

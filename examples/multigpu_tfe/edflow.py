import functools
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow.keras as tfk
import numpy as np
from edflow import TemplateIterator, get_logger
from tensorflow.python.distribute.values import PerReplica


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
        # strategy setup
        self.mirrored_strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        # mirroredstrategy fails if there is only one device, use this then
        #self.mirrored_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        self.num_replicas = self.mirrored_strategy.num_replicas_in_sync
        self.logger.info("num_replicas: {}".format(self.num_replicas))
        # build model with cross-replica context
        del self.model
        with self.mirrored_strategy.scope():
            self.model = Model(self.config)
            # loss and optimizer
            self.criterion = functools.partial(
                tfk.losses.sparse_categorical_crossentropy, from_logits=True
            )
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            # to save and restore
            self.tfcheckpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.optimizer
            )

    def _replicate(self, *args):
        if self.num_replicas <= 1:
            return args
        repls = list()
        for x in args:
            xs = np.array_split(x, self.num_replicas)
            device_map = self.mirrored_strategy.extended._device_map
            per_replica = PerReplica(device_map, xs)
            repls.append(per_replica)
        return repls

    def save(self, checkpoint_path):
        self.tfcheckpoint.write(checkpoint_path)

    def restore(self, checkpoint_path):
        with self.mirrored_strategy.scope():
            self.tfcheckpoint.restore(checkpoint_path)

    def step_op(self, model, **kwargs):
        # get inputs
        inputs, labels = kwargs["image"], kwargs["class"]

        def _train_op(inputs, labels):
            # compute loss
            with tf.GradientTape() as tape:
                outputs = model(inputs)
                loss = self.criterion(y_true=labels, y_pred=outputs)
                mean_loss = tf.reduce_mean(loss)

            grads = tape.gradient(mean_loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        def train_op():
            self.mirrored_strategy.experimental_run_v2(_train_op, args=self._replicate(inputs, labels))

        def log_op():
            outputs = model(inputs)
            loss = self.criterion(y_true=labels, y_pred=outputs)
            return {"images": {}, "scalars": {
                "loss": loss,
                "min_loss": np.min(loss),
                "max_loss": np.max(loss),
                "mean_loss": np.mean(loss)}}

        def eval_op():
            return {"outputs": np.array(outputs), "loss": np.array(loss)[:, None]}

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

        prediction = np.argmax(outputs, axis=0)
        correct += labels == prediction
        loss += loss
    logger.info("Loss: {}".format(loss / len(data_in)))
    logger.info("Accuracy: {}".format(correct / len(data_in)))

import functools
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow.keras as tfk
import numpy as np
from edflow import TemplateIterator, get_logger


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
            # the default logger understands the keys "images" (written as png
            # in the log directory) and "scalars" (written to stdout and the
            # log file).
            acc = np.mean(np.argmax(outputs, axis=1) == labels)
            min_loss = np.min(loss)
            max_loss = np.max(loss)
            return {
                "images": {"inputs": inputs},
                "scalars": {
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "mean_loss": mean_loss,
                    "acc": acc,
                },
            }

        def eval_op():
            # values under "labels" are written into a single file,
            # remaining values are written into one file per example.
            # Here, "outputs" would be small enough to write into labels, but
            # for illustration we do not write it as labels.
            return {"outputs": np.array(outputs), "labels": {"loss": np.array(loss)}}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        # callbacks to run after validation
        return {"eval_op": {"accuracy": acc_callback}}


def acc_callback(root, data_in, data_out, config):
    from tqdm import trange

    logger = get_logger("acc_callback")
    correct = 0
    seen = 0
    # labels are loaded directly into memory
    loss1 = np.mean(data_out.labels["loss"])
    loss2 = 0.0
    for i in trange(len(data_in), leave=False):
        # data_in is the dataset that was used for evaluation
        labels = data_in[i]["class"]
        # data_out contains all the keys that were specified in the eval_op
        outputs = data_out[i]["outputs"]
        # labels are also available on each example
        loss = data_out[i]["labels_"]["loss"]

        prediction = np.argmax(outputs, axis=0)
        correct += labels == prediction
        loss2 += loss
    logger.info("Loss1: {}".format(loss1))
    logger.info("Loss2: {}".format(loss2 / len(data_in)))
    logger.info("Accuracy: {}".format(correct / len(data_in)))

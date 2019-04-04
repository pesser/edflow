import tensorflow as tf
from edflow.iterators.trainer import TFBaseTrainer
import mnist_tf.nn as nn
from examples.mnist_tf.nn import conv2D, dense
from tensorflow.contrib.framework.python.ops import arg_scope


# mnist example form
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
# adapted for EDFlow


def mnist_model(x):
    with arg_scope([conv2D, dense], activation=tf.nn.relu):
        features = conv2D(x, 32, 5, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(features, 64, 3, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(features, 64, 3, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)

        y = tf.layers.flatten(features)
        y = dense(y, 1024)
        y = dense(y, 512)
    logits = dense(y, 10)  # 10 classes for mnist
    probs = tf.nn.softmax(logits, dim=-1)
    return probs, logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.define_graph()
        self.variables = tf.global_variables()


    @property
    def inputs(self):
        '''
        inputs of model at inference time
        Returns
        -------

        '''
        return {'image': self.image,
                "target" : self.targets}


    @property
    def outputs(self):
        '''
        outputs of model at inference time
        Returns
        -------

        '''
        return {'probs' : self.probs,
                "classes" : self.classes}

    def define_graph(self):
        # inputs
        self.image = tf.placeholder(
            tf.float32,
            shape = (
                self.config["batch_size"],
                self.config["spatial_size"],
                self.config["spatial_size"],
                1),
            name = "image_in")
        self.targets = tf.placeholder(
            tf.float32,
            shape=(
                self.config["batch_size"], # 10 classes in mnist # TODO maybe move this to config
            ))

        # model definition
        model = nn.make_model("model", mnist_model)
        probs, logits = model(self.image)

        # outputs
        self.probs = probs
        self.logits = logits
        self.classes = tf.argmax(probs, axis=1)


class Trainer(TFBaseTrainer):
    def get_restore_variables(self):
        ''' nothing fancy here '''
        return super().get_restore_variables()


    def initialize(self, checkpoint_path = None):
        ''' in this case, we do not need to initialize anything special '''
        return super().initialize(checkpoint_path)



    def make_loss_ops(self):
        probs = self.model.outputs["probs"]
        logits = self.model.logits
        targets = self.model.inputs["target"]
        correct = tf.nn.in_top_k(probs, tf.cast(targets, tf.int32), k=1)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        ce = loss(logits, targets)

        # losses are applied for each model
        # basically, we look for the string in the variables and update them with the loss provided here
        losses = dict()
        losses["model"] = ce

        # metrics for logging
        self.log_ops["acc"] = acc
        self.log_ops["ce"] = ce
        return losses

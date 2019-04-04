import sys, os
import tensorflow as tf
import numpy as np

from edflow.iterators.trainer import TFBaseTrainer
import edflow.iterators.deeploss as deeploss
from edflow.hooks.evaluation_hooks import RestoreTFModelHook
from edflow.util import make_linear_var

import mnist_tf.nn as nn
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope

# mnist example form
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

@add_arg_scope
def conv2D(x, f, k, s=1, padding="same", activation=None):
    return tf.layers.Conv2D(f, k, strides=s, padding=padding, activation=activation)(x)


@add_arg_scope
def dense(x, f, activation=None):
    return tf.layers.dense(x, f, activation=activation)


def mnist_model(x):
    with arg_scope([conv2D, dense], activation=tf.nn.relu):
        features = conv2D(x, 32, 5, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(x, 64, 3, padding="valid")
        features = tf.layers.max_pooling2d(features, 2, strides=1)
        features = conv2D(x, 64, 3, padding="valid")
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
        # self.variables = [v for v in tf.global_variables() if not v in variables]
        # self.variables = [v for v in self.variables if not self.e1_name in v.name]
        self.variables = tf.global_variables()


    @property
    def inputs(self):
        return {'image': self.image,
                "target" : self.targets}


    @property
    def outputs(self):
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
        variables = super().get_restore_variables()
        return variables


    def initialize(self, checkpoint_path = None):
        return_ = super().initialize(checkpoint_path)
        return return_


    def make_loss_ops(self):
        probs = self.model.outputs["probs"]
        logits = self.model.logits
        targets = self.model.inputs["target"]
        correct = tf.nn.in_top_k(probs, tf.cast(targets, tf.int32), k=1)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        ce = loss(logits, targets)
        losses = dict()
        losses["model"] = ce

        self.log_ops["acc"] = acc
        self.log_ops["ce"] = ce
        return losses

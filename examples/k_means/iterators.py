import tensorflow as tf
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import RestoreTFModelHook
from edflow.iterators.tf_iterator import TFHookedModelIterator
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.iterators.tf_trainer import TFBaseTrainer
from edflow.project_manager import ProjectManager

from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook

from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.hooks.metric_hooks.tf_metric_hook import MetricHook
from edflow.hooks.checkpoint_hooks.common import MetricTuple

from edflow.project_manager import ProjectManager

import time
import numpy as np
from sklearn.metrics import accuracy_score


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


def ce_metric(probs, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    epsilon = 1e-12

    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    labels = one_hot(labels, 10)
    predictions = np.clip(probs, epsilon, 1.0 - epsilon)
    ce = -np.sum(np.log(predictions) * labels, axis=1)
    return ce


def acc_metric(y_true, y_pred):
    return np.reshape(accuracy_score(y_true, y_pred), (1, 1))


from edflow.hooks.logging_hooks.tf_logging_hook import ImageOverviewHook
from edflow.hooks.util_hooks import IntervalHook
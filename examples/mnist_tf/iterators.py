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


class Trainer(TFBaseTrainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        overviewHook = ImageOverviewHook(
            images=self.img_ops, root_path=ProjectManager.train, interval=1
        )
        ihook = IntervalHook(
            [overviewHook],
            interval=1,
            modify_each=1,
            max_interval=self.config.get("log_freq", 1000),
            get_step=self.get_global_step,
        )
        self.hooks.append(ihook)

    def get_restore_variables(self):
        """ nothing fancy here """
        return super().get_restore_variables()

    def initialize(self, checkpoint_path=None):
        """ in this case, we do not need to initialize anything special """
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
        self.img_ops["input"] = self.model.image
        self.img_ops["2input"] = self.model.image
        return losses


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        kwargs[
            "num_epochs"
        ] = (
            1000
        )  # this will keep the evaluator running forever # TODO make this much nicer
        super(Evaluator, self).__init__(*args, **kwargs)

        # input_names={"key_from_metric" : "key from data set"}
        # output_names={"key from model outputs" : "key from loss" }

        metrics = [
            MetricTuple(
                input_names={"labels": "target"},
                output_names={"probs": "probs"},
                metric=ce_metric,
                name="val_ce",
            ),
            MetricTuple(
                input_names={"y_true": "target"},
                output_names={"y_pred": "classes"},
                metric=acc_metric,
                name="val_acc",
            ),
        ]
        m_hook = MetricHook(metrics, save_root=ProjectManager.latest_eval)

        self.hooks.append(m_hook)

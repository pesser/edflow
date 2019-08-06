import pytest
from edflow import tf_util
import tensorflow as tf

tf.enable_eager_execution()
import numpy as np


def test_make_periodic_step():
    step = tf.range(20)
    start_step = 0
    period_duration_in_steps = 10
    unit_step = tf_util.make_periodic_step(step, start_step, period_duration_in_steps)
    assert np.allclose(unit_step[:10], unit_step[10:])

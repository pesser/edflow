import pytest
import numpy as np
from edflow.iterators import tf_batches as batches


def test_tf_batch_to_canvas():
    import tensorflow as tf

    tf.enable_eager_execution()
    x = np.ones((9, 100, 100, 3))
    x = tf.convert_to_tensor(x)
    canvas = batches.tf_batch_to_canvas(x)
    assert canvas.shape == (1, 300, 300, 3)

    canvas = batches.tf_batch_to_canvas(x, cols=5)
    assert canvas.shape == (1, 200, 500, 3)

    canvas = batches.tf_batch_to_canvas(x, cols=1)
    assert canvas.shape == (1, 900, 100, 3)

    canvas = batches.tf_batch_to_canvas(x, cols=0)
    assert canvas.shape == (1, 900, 100, 3)

    canvas = batches.tf_batch_to_canvas(x, cols=None)
    assert canvas.shape == (1, 300, 300, 3)

    x = np.ones((9, 100, 100, 1))
    x = tf.convert_to_tensor(x)
    canvas = batches.tf_batch_to_canvas(x)
    assert canvas.shape == (1, 300, 300, 1)

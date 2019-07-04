import pytest
import numpy as np
from edflow.iterators import batches

def test_batch_to_canvas():
    x = np.ones((9, 100, 100, 3))
    canvas = batches.batch_to_canvas(x)
    assert canvas.shape == (300, 300, 3)

    canvas = batches.batch_to_canvas(x, cols=5)
    assert canvas.shape == (200, 500, 3)

    canvas = batches.batch_to_canvas(x, cols=1)
    assert canvas.shape == (900, 100, 3)

    canvas = batches.batch_to_canvas(x, cols=0)
    assert canvas.shape == (900, 100, 3)

    canvas = batches.batch_to_canvas(x, cols=None)
    assert canvas.shape == (300, 300, 3)
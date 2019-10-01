import pytest
import numpy as np
from edflow.iterators import batches
from edflow.util import get_leaf_names, retrieve


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


def test_deep_lod2dol():
    lod = [{"a": 1, "b": {"c": 1, "d": [1, 2]}, "e": [{"a": 1}] * 2}] * 3

    dol = batches._deep_lod2dol(lod)

    ref = {
        "a": np.array([1, 1, 1]),
        "b": {
            "c": np.array([1, 1, 1]),
            "d": [np.array([1, 1, 1]), np.array([2, 2, 2])],
        },
        "e": [{"a": np.array([1, 1, 1])}] * 2,
    }

    assert get_leaf_names(dol) == get_leaf_names(ref)

    for k in get_leaf_names(dol):
        assert all(retrieve(dol, k) == retrieve(ref, k))


def test_deep_lod2dol_wrong_inputs():
    lod = [[1, 2, 3], {"a": 1}]

    with pytest.raises(TypeError):
        dol = batches._deep_lod2dol(lod)

    lod = {"a": [1, 2, 3], "b": {"a": 1}}

    with pytest.raises(TypeError):
        dol = batches._deep_lod2dol(lod)


def test_deep_lod2dol_v2():
    lod = [{"a": 1, "b": {"c": 1, "d": [1, 2]}, "e": [{"a": 1}] * 2}] * 3

    dol = batches._deep_lod2dol_v2(lod)

    ref = {
        "a": np.array([1, 1, 1]),
        "b": {
            "c": np.array([1, 1, 1]),
            "d": [np.array([1, 1, 1]), np.array([2, 2, 2])],
        },
        "e": [{"a": np.array([1, 1, 1])}] * 2,
    }

    assert get_leaf_names(dol) == get_leaf_names(ref)

    for k in get_leaf_names(dol):
        assert all(retrieve(dol, k) == retrieve(ref, k))


def test_deep_lod2dol_v2_wrong_inputs():
    lod = [[1, 2, 3], {"a": 1}]

    with pytest.raises(TypeError):
        dol = batches._deep_lod2dol_v2(lod)

    lod = {"a": [1, 2, 3], "b": {"a": 1}}

    with pytest.raises(TypeError):
        dol = batches._deep_lod2dol_v2(lod)


def test_deep_lod2dol_v3():
    lod = [{"a": 1, "b": {"c": 1, "d": [1, 2]}, "e": [{"a": 1}] * 2}] * 3

    dol = batches._deep_lod2dol_v3(lod)

    ref = {
        "a": np.array([1, 1, 1]),
        "b": {
            "c": np.array([1, 1, 1]),
            "d": [np.array([1, 1, 1]), np.array([2, 2, 2])],
        },
        "e": [{"a": np.array([1, 1, 1])}] * 2,
    }

    assert get_leaf_names(dol) == get_leaf_names(ref)

    for k in get_leaf_names(dol):
        assert all(retrieve(dol, k) == retrieve(ref, k))


def test_deep_lod2dol_v3_wrong_inputs():
    lod = [[1, 2, 3], {"a": 1}]

    with pytest.raises(Exception):
        dol = batches._deep_lod2dol_v3(lod)

    lod = {"a": [1, 2, 3], "b": {"a": 1}}

    with pytest.raises(Exception):
        dol = batches._deep_lod2dol_v3(lod)

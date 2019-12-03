import numpy as np
import PIL.Image
import math
import warnings
from edflow.iterators.resize import resize_image  # noqa
from edflow.iterators.resize import resize_uint8  # noqa
from edflow.iterators.resize import resize_float32  # noqa
from edflow.iterators.resize import resize_hfloat32  # noqa

from edflow.util import get_leaf_names, retrieve, set_value

from chainer.iterators import MultiprocessIterator

from edflow.data.dataset import DatasetMixin  # noqa


def load_image(path):
    img = PIL.Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = np.asarray(img, dtype="float32")
    x = x / 127.5 - 1.0
    return x


def save_image(x, path):
    """Save image."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.array(x, dtype="uint8")
    if x.shape[-1] == 1:
        x = np.squeeze(x)
    PIL.Image.fromarray(x).save(path)


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.ones((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx, ...]
                tiling[
                    i * X.shape[1] : (i + 1) * X.shape[1],
                    j * X.shape[2] : (j + 1) * X.shape[2],
                    :,
                ] = img
    return tiling


def plot_batch(X, out_path, cols=None):
    """Save batch of images tiled."""
    canvas = batch_to_canvas(X, cols)
    save_image(canvas, out_path)


def batch_to_canvas(X, cols=None):
    """convert batch of images to canvas"""
    if len(X.shape) == 5:
        # tile
        oldX = np.array(X)
        n_tiles = X.shape[3]
        side = math.ceil(math.sqrt(n_tiles))
        X = np.zeros(
            (oldX.shape[0], oldX.shape[1] * side, oldX.shape[2] * side, oldX.shape[4]),
            dtype=oldX.dtype,
        )
        # cropped images
        for i in range(oldX.shape[0]):
            inx = oldX[i]
            inx = np.transpose(inx, [2, 0, 1, 3])
            X[i] = tile(inx, side, side)
    n_channels = X.shape[3]
    if n_channels > 4:
        X = X[:, :, :, :3]
    if n_channels == 1:
        X = np.tile(X, [1, 1, 1, 3])
    rc = math.sqrt(X.shape[0])
    if cols is None:
        rows = cols = math.ceil(rc)
    else:
        cols = max(1, cols)
        rows = math.ceil(X.shape[0] / cols)
    canvas = tile(X, rows, cols)
    return canvas


def _deep_lod2dol(list_of_nested_things):
    """Turns a list of nested dictionaries into a nested dictionary of lists.
    This function takes care that all leafs of the nested dictionaries are 
    considered as full keys, not only the top level keys.

    Parameters
    ----------
    list_of_nested_things : list
        A list of deep nested dictionaries.

    Returns
    -------
    out : dict
        A dict containing lists of leaf entries.

    Raises
    ------
    ValueError
        Raised if the passed object is not a ``list`` or if its values are not ``dict`` s.
    """

    # Put custom exceptions in try excepts so that we do not check everytime
    # the type, only when an error occurs
    try:
        leaf_keypaths = get_leaf_names(list_of_nested_things[0])
    except Exception as e:
        if not isinstance(list_of_nested_things, list):
            raise TypeError(
                "Expected `list` but got " "{}".format(type(list_of_nested_things))
            )
        else:
            raise e

    try:
        out = {}
        for key in leaf_keypaths:
            stacked_entry = np.stack([retrieve(d, key) for d in list_of_nested_things])
            set_value(out, key, stacked_entry)
    except Exception as e:
        for v in list_of_nested_things:
            if not isinstance(v, dict):
                raise TypeError("Entries must be `dict` but got " "{}".format(type(v)))
        raise e

    return out


def _deep_lod2dol_v2(list_of_nested_things):
    """Turns a list of nested dictionaries into a nested dictionary of lists.
    This function takes care that all leafs of the nested dictionaries are 
    considered as full keys, not only the top level keys.

    .. Note::

        The difference to :func:`deep_lod2dol` is, that the correct type is
        always checked not only at exceptions.

    Parameters
    ----------
    list_of_nested_things : list
        A list of deep dictionaries

    Returns
    -------
    out : dict
        A dict containing lists of leaf entries.

    Raises
    ------
    ValueError
        Raised if the passed object is not a ``list`` or if its values are not
        ``dict`` s.
    """

    if not isinstance(list_of_nested_things, list):
        raise TypeError(
            "Expected `list` but got " "{}".format(type(list_of_nested_things))
        )
    leaf_keypaths = get_leaf_names(list_of_nested_things[0])

    for v in list_of_nested_things:
        if not isinstance(v, dict):
            raise TypeError("Entries must be `dict` but got " "{}".format(type(v)))
    out = {}
    for key in leaf_keypaths:
        stacked_entry = np.stack([retrieve(d, key) for d in list_of_nested_things])
        set_value(out, key, stacked_entry)

    return out


def _deep_lod2dol_v3(list_of_nested_things):
    """Turns a list of nested dictionaries into a nested dictionary of lists.
    This function takes care that all leafs of the nested dictionaries are 
    considered as full keys, not only the top level keys.

    .. Note::

        The difference to :func:`deep_lod2dol` is, that the correct type is
        never checked.

    Parameters
    ----------
    list_of_nested_things : list(dict(anything))
        A list of deep dictionaries

    Returns
    -------
    out : dict(anything(list))
        A dict containing lists of leaf entries.
    """

    leaf_keypaths = get_leaf_names(list_of_nested_things[0])

    out = {}
    for key in leaf_keypaths:
        stacked_entry = np.stack([retrieve(d, key) for d in list_of_nested_things])
        set_value(out, key, stacked_entry)

    return out


def _benchmark_deep_lod2dol():
    from contextlib import contextmanager
    from time import time

    @contextmanager
    def timing(description: str, n: int) -> None:
        start = time()
        yield
        ellapsed_time = (time() - start) / n * 1000

        print(f"{description}: {ellapsed_time:0.3f} ms")

    N = 100

    for bs in [1, 5, 25, 250, 1000]:
        lod = [{"a": 1, "b": {"c": 1, "d": [1, 2]}, "e": [{"a": 1}] * 2}] * bs

        with timing("v1@{: >4}".format(bs), N):
            for i in range(N):
                _deep_lod2dol(lod)

        with timing("v2@{: >4}".format(bs), N):
            for i in range(N):
                _deep_lod2dol_v2(lod)

        with timing("v3@{: >4}".format(bs), N):
            for i in range(N):
                _deep_lod2dol_v3(lod)

        print("-" * 15)

    # This results in the following on my lenovo t480s with an i7
    # v1@   1: 0.168 ms
    # v2@   1: 0.159 ms
    # v3@   1: 0.137 ms
    # ---------------
    # v1@   5: 0.185 ms
    # v2@   5: 0.189 ms
    # v3@   5: 0.185 ms
    # ---------------
    # v1@  25: 0.502 ms
    # v2@  25: 0.408 ms
    # v3@  25: 0.403 ms
    # ---------------
    # v1@ 250: 3.364 ms
    # v2@ 250: 3.740 ms
    # v3@ 250: 5.661 ms
    # ---------------
    # v1@1000: 21.364 ms
    # v2@1000: 15.858 ms
    # v3@1000: 15.648 ms
    # ---------------


deep_lod2dol = _deep_lod2dol_v2


class Iterator(MultiprocessIterator):
    """Iterator that converts a list of dicts into a dict of lists."""

    def __next__(self):
        return deep_lod2dol(super(Iterator, self).__next__())

    @property
    def n(self):
        return len(self.dataset)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


def make_batches(
    dataset, batch_size, shuffle, n_processes=8, n_prefetch=1, error_on_timeout=False
):
    # the first n_processes / batch_size batches will be quite slow for some
    # reason
    if error_on_timeout:
        warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)
    batches = Iterator(
        dataset,
        repeat=True,
        batch_size=batch_size,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
        shuffle=shuffle,
    )
    return batches


if __name__ == "__main__":
    from edflow.util import pprint

    class Dset(DatasetMixin):
        def get_example(self, idx):
            return {"im": np.random.randint(0, 255, size=[32, 32, 3])}

        def __len__(self):
            return 100

    B = make_batches(Dset(), batch_size=16, shuffle=True)

    pprint(next(B))

    print(dir(B))

    B._prefetch_loop.batch_size = 32
    B.batch_size = 32

    pprint(next(B))

    _benchmark_deep_lod2dol()

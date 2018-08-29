import numpy as np
import PIL.Image
import numpy as np
import scipy.ndimage
import math
import pickle
import os
import random

from chainer.iterators import MultiprocessIterator


def load_image(path):
    img = PIL.Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    x = np.asarray(img, dtype = "float32")
    x = x / 127.5 - 1.0
    return x


def save_image(x, path):
    """Save image."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.array(x, dtype = "uint8")
    PIL.Image.fromarray(x).save(path)


def resize(x, size):
    try:
        zoom = [size[0] / x.shape[0],
                size[1] / x.shape[1]]
    except TypeError:
        zoom = [size / x.shape[0],
                size / x.shape[1]]
    for _ in range(len(x.shape)-2):
        zoom.append(1.0)
    x = scipy.ndimage.zoom(x, zoom)
    return x


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def plot_batch(X, out_path):
    """Save batch of images tiled."""
    n_channels = X.shape[3]
    if n_channels > 4:
        X = X[:, :, :, :3]
    if n_channels == 1:
        X = np.tile(X, [1, 1, 1, 3])
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    save_image(canvas, out_path)


class Iterator(MultiprocessIterator):
    """Iterator that converts a list of dicts into a dict of lists."""

    def _lod2dol(self, lod):
        return {k: np.stack([d[k] for d in lod], 0) for k in lod[0]}

    def __next__(self):
        return self._lod2dol(super(Iterator, self).__next__())

    @property
    def n(self):
        return len(self.dataset)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


def make_batches(dataset, batch_size, shuffle):
    batches = Iterator(dataset,
                       repeat=True,
                       batch_size=batch_size,
                       n_processes=16,
                       shuffle=shuffle)
    return batches

import numpy as np
import PIL.Image
import math
from edflow.iterators.resize import resize_image  # noqa
from edflow.iterators.resize import resize_uint8  # noqa
from edflow.iterators.resize import resize_float32  # noqa
from edflow.iterators.resize import resize_hfloat32  # noqa

from chainer.iterators import MultiprocessIterator

# from chainer.dataset import DatasetMixin
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
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
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


class Iterator(MultiprocessIterator):
    """Iterator that converts a list of dicts into a dict of lists."""

    def _lod2dol(self, lod):
        try:
            return {k: np.stack([d[k] for d in lod], 0) for k in lod[0]}
        except ValueError:
            # debug which key is causing trouble
            for k in lod[0]:
                try:
                    np.stack([d[k] for d in lod])
                except BaseException:
                    print(k)
                    for d in lod:
                        print(d[k])
                    print(k)
            raise

    def __next__(self):
        try:
            return self._lod2dol(super(Iterator, self).__next__())
        except BrokenPipeError:
            pass

    @property
    def n(self):
        return len(self.dataset)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)


def make_batches(dataset, batch_size, shuffle, n_processes=8, n_prefetch=1):
    # the first n_processes / batch_size batches will be quite slow for some
    # reason
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

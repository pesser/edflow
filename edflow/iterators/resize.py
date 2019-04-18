import numpy as np
from PIL import Image


def resize_image(x, size):
    """size is expanded if necessary and swapped to Pillow"""
    try:
        size = tuple(size)
    except TypeError:
        size = (size, size)
    size = (size[1], size[0])
    resample = Image.LANCZOS
    return x.resize(size=size, resample=resample)


def resize_uint8(x, size):
    """
    x: np.ndarray of shape (height, width) or (height, width, channels) and dtype uint8
    size: int or (int, int) for target height, width
    """

    oshape = x.shape
    assert len(oshape) in [2, 3]
    if len(oshape) == 3 and oshape[-1] == 1:
        x = np.squeeze(x)
        x = resize_uint8(x, size)
        x = np.expand_dims(x, -1)
        return x
    if len(oshape) == 3 and oshape[-1] > 3:
        xs = [x[:, :, i] for i in range(oshape[-1])]
        xs = [resize_uint8(y, size) for y in xs]
        return np.stack(xs, -1)

    try:
        x = Image.fromarray(x)
    except TypeError:
        raise TypeError(type(x), x.dtype, x.shape, x.min(), x.max())
    x = resize_image(x, size)
    x = np.array(x)
    return x


def resize_float32(x, size):
    dtype = x.dtype
    assert dtype in [np.float32, np.float64], dtype
    assert -1.0 <= np.min(x) and np.max(x) <= 1.0, (np.min(x), np.max(x))
    x = (x + 1.0) * 127.5
    x = np.asarray(x, dtype=np.uint8)
    x = resize_uint8(x, size)
    x = x / 127.5 - 1.0
    return x


def resize_hfloat32(x, size):
    assert x.dtype == np.float32
    assert 0.0 <= np.min(x) and np.max(x) <= 1.0
    x = 2.0 * x - 1.0
    x = resize_float32(x, size)
    x = (x + 1.0) / 2.0
    return x

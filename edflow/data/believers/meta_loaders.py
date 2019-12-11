import numpy as np
from PIL import Image
from edflow.data.util import adjust_support


def image_loader(path, support="0->255", resize_to=None):
    """

    Parameters
    ----------
    path : str
        Where to finde the image.
    support : str
        Defines the support and data type of the loaded image. Must be one of
            - ``0->255``: The PIL default. Datatype is ``np.uint8`` and all values
              are integers between 0 and 255.
            - ``0->1``: Datatype is ``np.float32`` and all values
              are floats between 0 and 1.
            - ``-1->1``: Datatype is ``np.float32`` and all values
              are floats between -1 and 1.
    resize_to : list
        If not None, the loaded image will be resized to these dimensions. Must
        be a list of two integers or a single integer, which is interpreted as
        list of two integers with same value.

    Returns
    -------
    im : np.array
        An image loaded using :class:`PIL.Image` and adjusted to the range as
        specified.
    """

    def loader(support=support, resize_to=resize_to):
        im = Image.open(path)

        if resize_to is not None:
            if isinstance(resize_to, int):
                resize_to = [resize_to] * 2

            im = im.resize(resize_to)

        im = np.array(im)

        if support == "0->255":
            return im
        else:
            return adjust_support(im, support, "0->255")

    return loader


def numpy_loader(path):
    """

    Parameters
    ----------
    path : str
        Where to finde the array.

    Returns
    -------
    arr : np.array
        An array loaded using :class:`np.load`
    """

    def loader():
        return np.load(path)

    return loader

import numpy as np
from PIL import Image
from edflow.data.util import adjust_support


def image_loader(path, support="0->255"):
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

    Returns
    -------
    im : np.array
        An image loaded using :class:`PIL.Image` and adjusted to the range as
        specified.
    """

    im = np.array(Image.open(path))

    if support == "0->255":
        return im
    else:
        return adjust_support(im, support, "0->255")


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

    return np.load(path)

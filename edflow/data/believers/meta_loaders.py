import numpy as np
import os
from PIL import Image
from edflow.data.util import adjust_support


def image_loader(path, root="", support="0->255", resize_to=None):
    """

    Parameters
    ----------
    path : str
        Where to finde the image.
    root : str
        Root path, at which the suuplied :attr:`path` starts. E.g. if all paths
        supplied to this function are relative to
        ``/export/scratch/you_are_great/dataset``, this path would be root.
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

    def loader(support=support, resize_to=resize_to, root=root):
        path_ = os.path.join(root, path)

        im = Image.open(path_)

        if resize_to is not None:
            if isinstance(resize_to, int):
                resize_to = [resize_to] * 2

            im = im.resize(resize_to)

        im = np.array(im)

        if support == "0->255":
            return im
        else:
            return adjust_support(im, support, "0->255")

    loader.__doc__ = f"""Loads the image found at {path} relative to {root},
    scales the support to :attr:`support` (default={support}) and resizes the
    image to :attr:`resize_to` (default: {resize_to}."""

    return loader


def numpy_loader(path, root=""):
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
        path_ = os.path.join(root, path)
        return np.load(path_)

    loader.__doc__ = f"""Loads the numpy array found at {path} relative to
    {root}."""

    return loader


def category(index, categories):
    """
    Turns an abstract category label into a readable label.

    Example:

    Your dataset has the label ``pid`` which has integer entries like ``[0, 0,
    0, ..., 2, 2]`` between 0 and 3.

    Inside the dataset's ``meta.yaml`` you define

    .. code-block:: yaml

        # meta.yaml
        # ...
        loaders:
            pid: category
        loader_kwargs:
            pid:
                categories: ['besser', 'pesser', 'Mimo Tilbich']

    Now examples will be annotated with ``{pid: 'besser'}`` if the pid is
    ``0``, ``{pid: 'pesser'}`` if pid is 1 or ``{pid: 'Mimo Tilbich'}`` if the
    pid is 2.

    Note that categories can be anything that implements a ``__getitem__``
    method. You simply need to take care, that it understands the ``index``
    value it is passed by this loader function.

    Parameters
    ----------
    index : int, Hashable
        Some value that will be passed to :attr:`categories`'s
        :meth:`__getitem__` method. I.e. :attr:`categories` can be a ``list`` or
        ``dict`` or whatever you want!
    categories : list, dict, object with `__getitem__` method
        Defines the categories you have in you dataset. Will be accessed like
        ``categories[index]``

    Returns
    -------
        category : object
            ``categories[index]``
    """

    return categories[index]


DEFAULT_LOADERS = {
    "image": image_loader,
    "np": numpy_loader,
    "category": category,
    "cat": category,
}

import os
import numpy as np
import yaml
import re

from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import retrieve, get_obj_from_str, pp2mkdtable, pop_keypath
from edflow.util import walk, set_value, edprint
from edflow.data.believers.meta_loaders import DEFAULT_LOADERS

try:
    from IPython import get_ipython
    from IPython.display import display, Markdown

    __COULD_HAVE_IPYTHON__ = True
except ImportError:
    __COULD_HAVE_IPYTHON__ = False


class MetaDataset(DatasetMixin):
    """
    The :class:`MetaDataset` allows for easy data reading using a simple
    interface.

    All you need to do is hand the constructor a path and it will look for all
    data in a special format and load it as numpy arrays. If further specified
    in a meta data file or the name of the label array, when calling the
    getitem method of the dataset, a special loader function will be called.

    Let's take a look at an example data folder of the following structure:

    .. code-block:: bash

        root/
        ├ meta.yaml
        ├ images/
        │  ├ image_1.png
        │  ├ image_2.png
        │  ├ image_3.png
        ...
        │  └ image_10000.png
        ├ image:image-*-10000-*-str.npy
        ├ attr1-*-10000-*-int.npy
        ├ attr2-*-10000x2-*-int.npy
        └ kps-*-10000x17x3-*-int.npy


    The ``meta.yaml`` file looks like this:

    .. code-block:: yaml

        description: |
            This is a dataset which loads images.
            All paths to the images are in the label `image`.

        loader_kwargs:
            image:
                support: "-1->1"

    The resulting dataset has the following labels:

        - ``image_``: the paths to the images. Note the extra ``_`` at the end.
        - ``attr1``
        - ``attr2``
        - ``kps``

    When using the ``__getitem__`` method of the dataset, the image loader will
    be applied to the image label at the given index and the image will be
    loaded from the given path.

    As we have specifed loader kweyword arguments, we will get the images with
    a support of ``[-1, 1]``.

    """

    def __init__(self, root):
        """
        Parameters
        ----------
        root : str
            Where to look for all the data.
        """
        meta_path = os.path.join(root, "meta.yaml")
        self.meta = meta = yaml.safe_load(open(meta_path, "r"))

        labels = load_labels(os.path.join(root, "labels"))
        self.loaders, self.loader_kwargs = setup_loaders(labels, meta)
        self.labels = clean_keys(labels, self.loaders)

        class Lenner:
            def __init__(self):
                self.l = None
                self.visited = []

            def __call__(self, key, label):
                if self.l is None:
                    self.l = len(label)
                else:
                    if len(label) != self.l:
                        raise ValueError(
                            f"Label {key} has a different length "
                            "than the other labels.\n"
                            f"Already seen: {self.visited}"
                        )
                self.visited += [key]

        L = Lenner()
        walk(self.labels, L, pass_key=True)

        self.num_examples = L.l

        self.append_labels = True

    def __len__(self):
        return self.num_examples

    def get_example(self, idx):
        """Loads all loadable data from the labels.
        
        Parameters
        ----------
        idx : int
            The index of the example to load
        """
        example = {}

        for key, loader in self.loaders.items():
            kwargs = self.loader_kwargs[key]
            example[key] = loader(self.labels[key + "_"][idx], **kwargs)

        return example

    def __repr__(self):
        if (
            __COULD_HAVE_IPYTHON__
            and hasattr(get_ipython(), "config")
            and "IPKernelApp" in get_ipython().config
        ):
            label_str = pp2mkdtable(self.labels, True)
        else:
            label_str = pp2mkdtable(self.labels, False)

        descr = self.meta.get("description", "MetaDataset")

        repr_str = f"{descr}\n\n# Labels\n{label_str}"

        return repr_str

    def show(self):
        repr_str = self.__repr__()

        expand_ = self.expand
        self.expand = True

        if (
            __COULD_HAVE_IPYTHON__
            and hasattr(get_ipython(), "config")
            and "IPKernelApp" in get_ipython().config
        ):
            repr_str += f"\n\n# Example 0\n{pp2mkdtable(self.__getitem__(0), True)}"
            display(Markdown(repr_str))
        else:
            repr_str += f"\n\n# Example 0\n{pp2mkdtable(self.__getitem__(0), True)}"
            print(repr_str)

        self.expand = expand_


def setup_loaders(labels, meta_dict):
    """Creates a map of key -> function pairs, which can be used to postprocess
    label values at each ``__getitem__`` call.

    Loaders defined in :attr:`meta_dict` supersede those definde in the label
    keys.

    Parameters
    ----------
    labels : dict(str, numpy.memmap)
        Labels contain all load-easy dataset relevant data. If the key follows
        the pattern ``name:loader``, this function will try to finde the
        corresponding loader in :attr:`DEFAULT_LOADERS`.
    meta_dict : dict
        A dictionary containing all dataset relevent information, which is the
        same for all examples. This function will try to find the entry
        ``loaders`` in the dictionary, which must contain another ``dict`` with
        ``name:loader`` pairs. Here ``loader`` must be either an entry in
        :attr:`DEFAULT_LOADERS` or a loadable import path.
        You can additionally define an entry ``loader_kwargs``, which must
        contain ``name:dict`` pairs. The dictionary is passed as keyword
        arguments to the loader corresponding to ``name``.

    Returns
    -------
    loaders : dict
        Name, function pairs, to apply loading logic based on the labels with
        the specified names.
    loader_kwargs : dict
        Name, dict pairs. The dicts are passed to the loader functions as
        keyword arguments.
    """

    loaders = {}
    loader_kwargs = {}

    for k in labels.keys():
        k, l = loader_from_key(k)
        if l is not None:
            loaders[k] = l

    meta_loaders = retrieve(meta_dict, "loaders", default={})
    meta_loader_kwargs = retrieve(meta_dict, "loader_kwargs", default={})

    loaders.update(meta_loaders)

    for k, l in loaders.items():
        if l in DEFAULT_LOADERS:
            loaders[k] = DEFAULT_LOADERS[l]
        else:
            loaders[k] = get_obj_from_str(l)

        if k in meta_loader_kwargs:
            loader_kwargs[k] = meta_loader_kwargs[k]
        else:
            loader_kwargs[k] = {}

    return loaders, loader_kwargs


def load_labels(root):
    """
    Parameters
    ----------
    root : str
        Where to look for the labels.

    Returns
    -------
    labels : dict
        All labels as ``np.memmap`` s.
    """

    regex = re.compile(r".*-\*-.*-\*-.*\.npy")

    label_files = _get_label_files(root)

    class Loader:
        def __init__(self):
            self.labels = {}

        def __call__(self, key_path, path):
            if isinstance(path, str) and regex.match(path):
                f = os.path.basename(path)
                f_ = f[: -len(".npy")]
                key_, shape, dtype = f_.split("-*-")
                shape = tuple([int(s) for s in shape.split("x")])

                key_path = key_path.split("/")
                if len(key_path) == 1:
                    key = key_
                else:
                    key = "/".join(key_path[:-1] + [key_])

                mmap = np.memmap(path, mode="c", shape=shape, dtype=dtype)

                set_value(self.labels, key, mmap)

    L = Loader()
    walk(label_files, L, pass_key=True)

    return L.labels


def clean_keys(labels, loaders):
    """Removes all loader information from the keys.

    Parameters
    ----------
    labels : dict(str, numpy.memmap)
        Labels contain all load-easy dataset relevant data. 
    
    Returns
    -------
    labels : dict(str, numpy.memmap)
        The original labels, with keys without the ``:loader`` part.
    """

    class Cleaner:
        def __init__(self):
            self.to_delete = []
            self.to_set = []

        def __call__(self, key, val):
            k, l = loader_from_key(key)
            if l is not None:
                self.to_set += [[k + "_", retrieve(labels, key)]]
                self.to_delete += [key]

    C = Cleaner()
    walk(labels, C, pass_key=True)

    for key, val in C.to_set:
        set_value(labels, key, val)

    for key in C.to_delete:
        pop_keypath(labels, key)

    for k_ in list(loaders.keys()):
        if k_ in labels:
            k = k_ + "_"
            labels[k] = labels[k_]
            del labels[k_]

    return labels


def loader_from_key(key):
    """Returns the name, loader pair given a key."""

    if ":" in key:
        return key.split(":")
    return key, None


def _get_label_files(root):

    regex = re.compile(r".*-\*-.*-\*-.*\.npy")

    def f(path, regex):
        d = {}

        name_ = os.path.basename(path)

        if os.path.isdir(path):
            for name in os.listdir(path):
                d[name] = f(os.path.join(path, name), regex)
        else:
            if regex.match(path):
                d = path
            else:
                d = None
        return d

    root_, name = os.path.split(root)
    structure = f(root, regex)

    return structure

import numpy as np
import os


def store_label_mmap(data, root, name):
    """Stores the numpy array :attr:`data` as numpy `MemoryMap` with the naming
    convention, that is loadable by :class:`MetaDataset`.

    Parameters
    ----------
    data : numpy.ndarray
        The data to store.
    root : str:
        Where to store the memory map.
    name : str
        The name of the array. If loaded by :class:`MetaDataset` this will be
        the key in the labels dictionary at which one can find the data.
    """

    shape_str = "x".join([str(int(x)) for x in data.shape])
    mmap_path = os.path.join(root, f"{name}-*-{shape_str}-*-{data.dtype}.npy")

    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=data.shape)
    mmap[:] = data

from tqdm import tqdm, trange
from zipfile import ZipFile, ZIP_DEFLATED  # , ZIP_BZIP2, ZIP_LZMA
import numpy as np
import os
import pickle

from multiprocessing.managers import BaseManager
import queue

from edflow.data.dataset_mixin import DatasetMixin


def make_server_manager(port=63127, authkey=b"edcache"):
    inqueue = queue.Queue()
    outqueue = queue.Queue()

    class InOutManager(BaseManager):
        pass

    InOutManager.register("get_inqueue", lambda: inqueue)
    InOutManager.register("get_outqueue", lambda: outqueue)
    manager = InOutManager(address=("", port), authkey=authkey)
    manager.start()
    print("Started manager server at {}".format(manager.address))
    return manager


def make_client_manager(ip, port=63127, authkey=b"edcache"):
    class InOutManager(BaseManager):
        pass

    InOutManager.register("get_inqueue")
    InOutManager.register("get_outqueue")
    manager = InOutManager(address=(ip, port), authkey=authkey)
    manager.connect()
    print("Connected to server at {}".format(manager.address))
    return manager


def pickle_and_queue(
    dataset_factory, inqueue, outqueue, naming_template="example_{}.p"
):
    """Parallelizable function to retrieve and queue examples from a Dataset.

    Parameters
    ----------
    dataset_factory : chainer.DatasetMixin
        A dataset factory, with methods described in :class:`CachedDataset`.
    indices : list
        List of indices, used to retrieve samples from dataset.
    queue : mp.Queue
        Queue to put the samples in.
    naming_template : str
        Formatable string, which defines the name of the stored file given
        its index.

    """
    pbar = tqdm()
    dataset = dataset_factory()
    while True:
        try:
            indices = inqueue.get_nowait()
        except queue.Empty:
            return

        for idx in indices:
            try:
                example = dataset[idx]
            except BaseException:
                print("Error getting example {}".format(idx))
                raise
            pickle_name = naming_template.format(idx)
            pickle_bytes = pickle.dumps(example)

            outqueue.put([pickle_name, pickle_bytes])
            pbar.update(1)


class ExamplesFolder(object):
    """Contains all examples and labels of a cached dataset."""

    def __init__(self, root):
        self.root = root

    def read(self, name):
        with open(os.path.join(self.root, name), "rb") as example:
            return example.read()


class _CacheDataset(DatasetMixin):
    """Only used to avoid initializing the original dataset."""

    def __init__(self, root, name, _legacy=True):
        self.root = root
        self.name = name

        filespath = os.path.join(root, "cached", name)

        if _legacy:
            zippath = filespath + ".zip"
            # naming_template = 'example_{}.p'
            with ZipFile(zippath, "r") as zip_f:
                filenames = zip_f.namelist()
        else:
            filenames = os.listdir(filespath)

        def is_example(name):
            return name.startswith("example_") and name.endswith(".p")

        examplefilenames = [n for n in filenames if is_example(n)]
        self.n = len(examplefilenames)

    def __len__(self):
        return self.n


class CachedDataset(DatasetMixin):
    """Using a Dataset of single examples creates a cached (saved to memory)
    version, which can be accessed way faster at runtime.

    To avoid creating the dataset multiple times, it is checked if the cached
    version already exists.

    Calling `__getitem__` on this class will try to retrieve the samples from
    the cached dataset to reduce the preprocessing overhead.

    The cached dataset will be stored in the root directory of the base dataset
    in the subfolder `cached` with name `name.zip`.

    Besides the usual DatasetMixin interface, datasets to be cached must
    also implement

        root        # (str) root folder to cache into
        name        # (str) unqiue name

    Optionally but highly recommended, they should provide

        in_memory_keys  # list(str) keys which will be collected from examples

    The collected values are stored in a dict of list, mapping an
    in_memory_key to a list containing the i-ths value at the i-ths place.
    This data structure is then exposed via the attribute `labels` and
    enables rapid iteration over useful labels without loading each example
    seperately. That way, downstream datasets can filter the indices of the
    cached dataset efficiently, e.g. filtering based on train/eval splits.

    Caching proceeds as follows:
    Expose a method which returns the dataset to be cached, e.g.

        def DataToCache():
          path = "/path/to/data"
          return MyCachableDataset(path)

    Start caching server on host <server_ip_or_hostname>:

        edcache --server --dataset import.path.to.DataToCache

    Wake up a worker bee on same or different hosts:

        edcache --address <server_ip_or_hostname> --dataset import.path.to.DataCache  # noqa

    Start a cacherhive!
    """

    _legacy = True

    def __init__(
        self,
        dataset,
        force_cache=False,
        keep_existing=True,
        _legacy=True,
        chunk_size=64,
    ):
        """Given a dataset class, stores all examples in the dataset, if this
        has not yet happened.

        Parameters
        ----------
        dataset : object
            Dataset class which defines the following methods: \n
            - `root`: returns the path to the raw data \n
            - `name`: returns the name of the dataset -> best be unique \n
            - `__len__`: number of examples in the dataset \n
            - `__getitem__`: returns a sindle datum \n
            - `in_memory_keys`: returns all keys, that are stored \n
            alongside the dataset, in a `labels.p` file. This
            allows to retrive labels more quickly and can be used
            to filter the data more easily.
        force_cache : bool
            If True the dataset is cached even if an existing, cached version
            is overwritten.
        keep_existing : bool
            If True, existing entries in cache will not be recomputed and only
            non existing examples are appended to the cache. Useful if caching
            was interrupted.
        _legacy : bool
            Read from the cached Zip file. Deprecated mode.
            Future Datasets should not write into zips as read times are
            very long.
        chunksize : int
            Length of the index list that is sent to the worker.
        """

        self.force_cache = force_cache
        self.keep_existing = keep_existing
        self._legacy = _legacy

        self.base_dataset = dataset
        self._root = root = dataset.root
        name = dataset.name
        self.chunk_size = chunk_size

        self.store_dir = os.path.join(root, "cached")
        self.store_path = os.path.join(self.store_dir, name)
        if _legacy:
            self.store_path += ".zip"

        # leading_zeroes = str(len(str(len(self))))
        # self.naming_template = 'example_{:0>' + leading_zeroes + '}.p'
        # above might be better, but for compatibility we need this right
        # now, because pickle_and_queue did not receive the updated template
        self.naming_template = "example_{}.p"
        self._labels_name = "labels.p"

        os.makedirs(self.store_dir, exist_ok=True)
        if self.force_cache:
            self.cache_dataset()

    @classmethod
    def from_cache(cls, root, name, _legacy=True):
        """Use this constructor to avoid initialization of original dataset
        which can be useful if only the cached zip file is available or to
        avoid expensive constructors of datasets."""
        dataset = _CacheDataset(root, name, _legacy)
        return cls(dataset, _legacy=_legacy)

    def __getstate__(self):
        """Close file before pickling."""
        if hasattr(self, "zip"):
            self.zip.close()
        self.zip = None
        self.currentpid = None
        return self.__dict__

    @property
    def fork_safe_zip(self):
        if self._legacy:
            currentpid = os.getpid()
            if getattr(self, "_initpid", None) != currentpid:
                self._initpid = currentpid
                self.zip = ZipFile(self.store_path, "r")
            return self.zip
        return ExamplesFolder(self.store_path)

    def cache_dataset(self):
        """Checks if a dataset is stored. If not iterates over all possible
        indices and stores the examples in a file, as well as the labels."""

        if not os.path.isfile(self.store_path) or self.force_cache:
            print("Caching {}".format(self.store_path))
            manager = make_server_manager()
            inqueue = manager.get_inqueue()
            outqueue = manager.get_outqueue()

            N_examples = len(self.base_dataset)
            indices = np.arange(N_examples)
            if self.keep_existing and os.path.isfile(self.store_path):
                with ZipFile(self.store_path, "r") as zip_f:
                    zipfilenames = zip_f.namelist()
                zipfilenames = set(zipfilenames)
                indices = [
                    i
                    for i in indices
                    if not self.naming_template.format(i) in zipfilenames
                ]
                print("Keeping {} cached examples.".format(N_examples - len(indices)))
                N_examples = len(indices)
            print("Caching {} examples.".format(N_examples))
            index_chunks = [
                indices[i : i + self.chunk_size]
                for i in range(0, len(indices), self.chunk_size)
            ]
            for chunk in index_chunks:
                inqueue.put(chunk)
            print("Waiting for results.")

            pbar = tqdm(total=N_examples)
            mode = "a" if self.keep_existing else "w"
            with ZipFile(self.store_path, mode, ZIP_DEFLATED) as self.zip:
                done_count = 0
                while True:
                    if done_count == N_examples:
                        break
                    pickle_name, pickle_bytes = outqueue.get()
                    self.zip.writestr(pickle_name, pickle_bytes)
                    pbar.update(1)
                    done_count += 1

            # after everything is done, we store memory keys seperately for
            # more efficient access
            # Note that this is always called, in case one wants to add labels
            # after caching has finished. This will add a new file with the
            # same name to the zip and it is currently not possible to delete
            # the old one. Preliminary tests have shown that the read method
            # returns the newest file if multiple ones are available but this
            # is _not_ documented or guaranteed in the API. If you experience
            # problems, try to write a new zip file with desired contents or
            # delete cached zip and cache again.
            memory_dict = dict()
            if hasattr(self.base_dataset, "in_memory_keys"):
                print("Caching Labels.")
                memory_keys = self.base_dataset.in_memory_keys
                for key in memory_keys:
                    memory_dict[key] = list()

                for idx in trange(len(self.base_dataset)):
                    example = self[idx]  # load cached version
                    # extract keys
                    for key in memory_keys:
                        memory_dict[key].append(example[key])

            with ZipFile(self.store_path, "a", ZIP_DEFLATED) as zipfile:
                zipfile.writestr(self._labels_name, pickle.dumps(memory_dict))
            print("Finished caching.")

    def __len__(self):
        """Number of examples in this Dataset."""
        return len(self.base_dataset)

    @property
    def labels(self):
        """Returns the labels associated with the base dataset, but from the
        cached source."""
        if not hasattr(self, "_labels"):
            labels = self.fork_safe_zip.read(self._labels_name)
            labels = pickle.loads(labels)
            self._labels = labels
        return self._labels

    @property
    def root(self):
        """Returns the root to the base dataset."""
        return self._root

    def get_example(self, i):
        """Given an index i, returns a example."""

        example_name = self.naming_template.format(i)
        example_file = self.fork_safe_zip.read(example_name)

        example = pickle.loads(example_file)

        return example


class PathCachedDataset(CachedDataset):
    """Used for simplified decorator interface to dataset caching."""

    def __init__(self, dataset, path):
        self.force_cache = False
        self.keep_existing = True

        self.base_dataset = dataset
        self.store_dir = os.path.split(path)[0]
        self.store_path = path

        self.naming_template = "example_{}.p"
        self._labels_name = "labels.p"

        os.makedirs(self.store_dir, exist_ok=True)

        self.lenfile = self.store_path + ".p"
        if not os.path.exists(self.lenfile):
            self.force_cache = True
        self.cache_dataset()
        if not os.path.exists(self.lenfile):
            with open(self.lenfile, "wb") as f:
                pickle.dump(len(self.base_dataset), f)

    def __len__(self):
        if not (self.base_dataset is None or os.path.exists(self.lenfile)):
            return len(self.base_dataset)
        if not hasattr(self, "_len"):
            with open(self.lenfile, "rb") as f:
                self._len = pickle.load(f)
        return self._len


def cachable(path):
    """Decorator to cache datasets. If not cached, will start a caching server,
    subsequent calls will just load from cache. Currently all worker must be
    able to see the path. Be careful, function parameters are ignored on
    furture calls.
    Can be used on any callable that returns a dataset. Currently the path
    should be the path to a zip file to cache into - i.e. it should end in zip.
    """

    def decorator(fn):
        def wrapped(*args, **kwargs):
            if os.path.exists(path + ".p"):
                # cached version ready
                return PathCachedDataset(None, path)
            elif os.path.exists(path + "parameters.p"):
                # zip exists but not pickle with length - caching server
                # started and we are a worker bee
                with open(path + "parameters.p", "rb") as f:
                    args, kwargs = pickle.load(f)
                return fn(*args, **kwargs)
            else:
                # start caching server
                dataset = fn(*args, **kwargs)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                with open(path + "parameters.p", "wb") as f:
                    pickle.dump((args, kwargs), f)
                return PathCachedDataset(dataset, path)

        return wrapped

    return decorator

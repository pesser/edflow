'''All handy dataset classes we use.'''

import os
import pickle
from zipfile import ZipFile, ZIP_DEFLATED  # , ZIP_BZIP2, ZIP_LZMA
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm
from chainer.dataset import DatasetMixin

from multiprocessing.managers import BaseManager
import queue


def make_server_manager(port = 63127, authkey = b"edcache"):
    inqueue = queue.Queue()
    outqueue = queue.Queue()
    class InOutManager(BaseManager):
        pass
    InOutManager.register("get_inqueue", lambda: inqueue)
    InOutManager.register("get_outqueue", lambda: outqueue)
    manager = InOutManager(address=("", port), authkey = authkey)
    manager.start()
    print("Started manager server at {}".format(manager.address))
    return manager


def make_client_manager(ip, port = 63127, authkey = b"edcache"):
    class InOutManager(BaseManager):
        pass
    InOutManager.register("get_inqueue")
    InOutManager.register("get_outqueue")
    manager = InOutManager(address=(ip, port), authkey = authkey)
    manager.connect()
    print("Connected to server at {}".format(manager.address))
    return manager


def pickle_and_queue(dataset_factory,
                     inqueue,
                     outqueue,
                     naming_template='example_{}.p'):
    '''Parallelizable function to retrieve and queue examples from a Dataset.

    Args:
        dataset_factory (() -> chainer.DatasetMixin): A dataset factory, with methods described in
            :class:`CachedDataset`.
        indeces (list): List of indeces, used to retrieve samples from dataset.
        queue (mp.Queue): Queue to put the samples in.
        naming_template (str): Formatable string, which defines the name of
            the stored file given its index.
    '''
    dataset = dataset_factory()
    while True:
        try:
            indices = inqueue.get_nowait()
        except queue.Empty:
            return

        for idx in indices:
            example = dataset[idx]
            pickle_name = naming_template.format(idx)
            pickle_bytes = pickle.dumps(example)

            outqueue.put([pickle_name, pickle_bytes])


class CachedDataset(DatasetMixin):
    """Using a Dataset of single examples creates a cached (saved to memory)
    version, which can be accessed way faster at runtime.

    To avoid creating the dataset multiple times, it is checked if the cached
    version already exists.

    Calling `__getitem__` on this class will try to retrieve the samples from
    the cached dataset to reduce the preprocessing overhead.

    The cached dataset will be stored in the root directory of the base dataset
    in the subfolder `cached`."""

    def __init__(self, dataset, force_cache=False, n_workers=2,
            keep_existing = True):
        '''Given a dataset class, stores all examples in the dataset, if this
        has not yet happened.

        Args:
            dataset (object): Dataset class which defines the following
                methods:
                    - `root`: returns the path to the raw data
                    - `name`: returns the name of the dataset -> best be unique
                    - `__len__`: number of examples in the dataset
                    - `__getitem__`: returns a sindle datum
                    - `in_memory_keys`: returns all keys, that are stored
                        alongside the dataset, in a `labels.p` file. This
                        allows to retrive labels more quickly and can be used
                        to filter the data more easily.
            force_cache (bool): If True the dataset is cached even if an
                existing, cached version is overwritten.
            n_workers (int): Number of workers to use during caching.
            keep_existing (bool): If True, existing entries in cache will
                not be recomputed and only non existing examples are
                appended to the cache. Useful if caching was interrupted.
        '''

        self.force_cache = force_cache
        self.n_workers = n_workers
        self.keep_existing = keep_existing

        self.base_dataset = dataset
        self._root = root = dataset.root
        name = dataset.name

        self.store_dir = os.path.join(root, 'cached')
        self.store_path = os.path.join(self.store_dir, name + '.zip')
        self.label_path = os.path.join(root, 'cached', name + '_labels.p')

        leading_zeroes = str(len(str(len(self))))
        self.naming_template = 'example_{:0>' + leading_zeroes + '}.p'

        os.makedirs(self.store_dir, exist_ok=True)
        self.cache_dataset()

        self.zip = ZipFile(self.store_path, 'r')

    def cache_dataset(self):
        '''Checks if a dataset is stored. If not iterates over all possible
        indeces and stores the examples in a file, as well as the labels.'''

        if not os.path.isfile(self.store_path) or self.force_cache:
            manager = make_server_manager()
            inqueue = manager.get_inqueue()
            outqueue = manager.get_outqueue()

            N_examples = len(self.base_dataset)
            indeces = np.arange(N_examples)
            if self.keep_existing and os.path.isfile(self.store_path):
                with ZipFile(self.store_path, 'r') as zip_f:
                    zipfilenames = zip_f.namelist()
                indeces = [i for i in indeces if
                        not self.naming_template.format(i) in zipfilenames]
                print("Keeping {} cached examples.".format(N_examples - len(indeces)))
                N_examples = len(indeces)
            chunk_size = 64
            index_chunks = [indeces[i:i+chunk_size]
                    for i in range(0, len(indeces), chunk_size)]
            for chunk in index_chunks:
                inqueue.put(chunk)
            print("Waiting for results.")

            pbar = tqdm(total=N_examples)
            mode = "a" if self.keep_existing else "w"
            with ZipFile(self.store_path, mode, ZIP_DEFLATED) as zip_f:
                done_count = 0
                while True:
                    pickle_name, pickle_bytes = outqueue.get()
                    zip_f.writestr(pickle_name, pickle_bytes)
                    pbar.update(1)
                    done_count += 1
                    if done_count == N_examples:
                        break

            self.zip = ZipFile(self.store_path, 'r')
            # after everything is done, we store memory keys seperately for
            # more efficient access
            memory_dict = dict()
            if hasattr(self.base_dataset, 'in_memory_keys'):
                print('Caching Labels.')
                memory_keys = self.base_dataset.in_memory_keys
                for key in memory_keys:
                    memory_dict[key] = list()
                for idx in range(len(self.base_dataset)):
                    example = self[idx] # load cached version
                    # extract keys
                    for key in memory_keys:
                        memory_dict[key].append(example[key])
            # dump to disk
            with open(self.label_path, 'wb') as labels_file:
                pickle.dump(memory_dict, labels_file)
            print("Finished caching.")
            self.zip.close()

    def __len__(self):
        '''Number of examples in this Dataset.'''
        return len(self.base_dataset)

    @property
    def labels(self):
        '''Returns the labels asociated with the base dataset, but from the
        cached source.'''
        with open(self.label_path, 'r') as labels_file:
            labels = pickle.load(labels_file)
        return labels

    @property
    def root(self):
        '''Returns the root to the base dataset.'''
        return self._root

    def get_example(self, i):
        '''Given an index i, returns a example.'''

        example_name = self.naming_template.format(i)
        example_file = self.zip.read(example_name)

        example = pickle.loads(example_file)

        return example

'''All handy dataset classes we use.'''

import os
import pickle
from zipfile import ZipFile, ZIP_DEFLATED  # , ZIP_BZIP2, ZIP_LZMA
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm
from chainer.dataset import DatasetMixin


def pickle_and_queue(dataset,
                     indeces,
                     queue,
                     naming_template='example_{}.p',
                     store_keys=[]):
    '''Parallelizable function to retrieve and queue examples from a Dataset.

    Args:
        dataset (chainer.DatasetMixin): A dataset, with methods described in
            :class:`CachedDataset`.
        indeces (list): List of indeces, used to retrieve samples from dataset.
        queue (mp.Queue): Queue to put the samples in.
        naming_template (str): Formatable string, which defines the name of
            the stored file given its index.
        store_keys (list): Keys or indeces of values to be stored extra.
    '''

    for idx in indeces:
        example = dataset[idx]

        pickle_name = naming_template.format(idx)
        pickle_bytes = pickle.dumps(example)

        extra_vals = dict((k,example[k]) for k in store_keys)

        queue.put([pickle_name, pickle_bytes, extra_vals])

    queue.put(['Done', None, None])


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
            Q = Queue()

            N_examples = len(self.base_dataset)
            indeces = np.arange(N_examples)
            if self.keep_existing and os.path.isfile(self.store_path):
                with ZipFile(self.store_path, 'r') as zip_f:
                    zipfilenames = zip_f.namelist()
                indeces = [i for i in indeces if
                        not self.naming_template.format(i) in zipfilenames]
                print("Keeping {} cached examples.".format(N_examples - len(indeces)))
                N_examples = len(indeces)
            index_lists = np.array_split(indeces, self.n_workers)

            memory_keys = list()
            memory_dict = dict()
            if hasattr(self.base_dataset, 'in_memory_keys'):
                memory_keys = self.base_dataset.in_memory_keys
                for key in memory_keys:
                    memory_dict[key] = list()

            pbar = tqdm(total=N_examples)

            print('Caching dataset using {} workers. '.format(self.n_workers),
                  'This might take a while.')
            mode = "a" if self.keep_existing else "w"
            with ZipFile(self.store_path, mode, ZIP_DEFLATED) as zip_f:
                processes = list()
                for n in range(self.n_workers):
                    p_args = (self.base_dataset,
                              index_lists[n],
                              Q,
                              self.naming_template,
                              memory_keys)
                    p = Process(target=pickle_and_queue, args=p_args)
                    processes.append(p)

                for p in processes:
                    p.start()

                done_count = 0
                while True:
                    pickle_name, pickle_bytes, extra_vals = Q.get()

                    for key in memory_keys:
                        memory_dict[key] += [extra_vals[key]]

                    if not pickle_name == 'Done':
                        zip_f.writestr(pickle_name, pickle_bytes)
                        pbar.update(1)
                    else:
                        done_count += 1

                    if done_count == self.n_workers:
                        break

                for p in processes:
                    p.join()

            print('Caching Labels.')
            with open(self.label_path, 'wb') as labels_file:
                pickle.dump(memory_dict, labels_file)

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

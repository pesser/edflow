'''All handy dataset classes we use.
Our Datasets (TODO usually) inherit from the `chainer.dataset.DatasetMixin`.
Please note that we overwrite two methods here: The `__getitem__` and the
`get_example` method. These do not change their functionallty and there is
not problem in not using our version of `DatasetMixin`.

Here is what we did:

:codeblock: python

DatasetMixin(DatasetMixin):
    def __getitem__(self, index):
        \'\'\'Don't raise BrokenPipeError as those usually occur during some
        other Exception.\'\'\'
        try:
            return super().__getitem__(index)
        except Exception as e:
            if not isinstance(e, BrokenPipeError):
                traceback.print_exc()
                raise e

    def get_example(self, i):
        \'\'\'Add the index `i` to the returned datum. This assumes that you
        always return a dictionary.\'\'\'

        result = super().get_example(i)
        result['_index'] = i

        return result
'''

import os
import pickle
import traceback
from zipfile import ZipFile, ZIP_DEFLATED  # , ZIP_BZIP2, ZIP_LZMA
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm, trange
from chainer.dataset import DatasetMixin
# TODO maybe just pull
# https://github.com/chainer/chainer/blob/v4.4.0/chainer/dataset/dataset_mixin.py
# into the rep to avoid dependency on chainer for this one mixin - it doesnt
# even do that much and it would provide better documentation as this is
# actually our base class for datasets

from multiprocessing.managers import BaseManager
import queue

from edflow.main import traceable_method, get_implementations_from_config


DatasetMixin.__getitem__ = traceable_method(DatasetMixin.__getitem__,
                                            ignores=(BrokenPipeError))


def indexed_method(method, key='_index'):
    def imethod(index, *args, **kwargs):
        result = method(*args, **kwargs)
        result[key] = index
        return result
    return imethod


DatasetMixin.get_examples = indexed_method(DatasetMixin.get_example, '_index')


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
        dataset_factory (() -> chainer.DatasetMixin): A dataset factory, with
            methods described in :class:`CachedDataset`.
        indeces (list): List of indeces, used to retrieve samples from dataset.
        queue (mp.Queue): Queue to put the samples in.
        naming_template (str): Formatable string, which defines the name of
            the stored file given its index.
    '''
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
            except:
                print("Error getting example {}".format(idx))
                raise
            pickle_name = naming_template.format(idx)
            pickle_bytes = pickle.dumps(example)

            outqueue.put([pickle_name, pickle_bytes])
            pbar.update(1)


class _CacheDataset(DatasetMixin):
    """Only used to avoid initializing the original dataset."""
    def __init__(self, root, name):
        self.root = root
        self.name = name

        zippath = os.path.join(root, "cached", name+".zip")
        naming_template = 'example_{}.p'
        with ZipFile(zippath, 'r') as zip_f:
            zipfilenames = zip_f.namelist()
        def is_example(name):
            return name.startswith("example_") and name.endswith(".p")
        examplefilenames = [n for n in zipfilenames if is_example(n)]
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

        edcache --address <server_ip_or_hostname> --dataset import.path.to.DataCache

    Start a cacherhive!
    """

    def __init__(self,
                 dataset,
                 force_cache=False,
                 keep_existing=True):
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
            keep_existing (bool): If True, existing entries in cache will
                not be recomputed and only non existing examples are
                appended to the cache. Useful if caching was interrupted.
        '''

        self.force_cache = force_cache
        self.keep_existing = keep_existing

        self.base_dataset = dataset
        self._root = root = dataset.root
        name = dataset.name

        self.store_dir = os.path.join(root, 'cached')
        self.store_path = os.path.join(self.store_dir, name + '.zip')

        #leading_zeroes = str(len(str(len(self))))
        #self.naming_template = 'example_{:0>' + leading_zeroes + '}.p'
        # above might be better, but for compatibility we need this right
        # now, because pickle_and_queue did not receive the updated template
        self.naming_template = 'example_{}.p'
        self._labels_name = 'labels.p'

        os.makedirs(self.store_dir, exist_ok=True)
        if self.force_cache:
            self.cache_dataset()

    @classmethod
    def from_cache(cls, root, name):
        """Use this constructor to avoid initialization of original dataset
        which can be useful if only the cached zip file is available or to
        avoid expensive constructors of datasets."""
        dataset = _CacheDataset(root, name)
        return cls(dataset)

    def __getstate__(self):
        """Close file before pickling."""
        if hasattr(self, "zip"):
            self.zip.close()
        self.zip = None
        self.currentpid = None
        return self.__dict__

    @property
    def fork_safe_zip(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self.zip = ZipFile(self.store_path, 'r')
        return self.zip

    def cache_dataset(self):
        '''Checks if a dataset is stored. If not iterates over all possible
        indeces and stores the examples in a file, as well as the labels.'''

        if not os.path.isfile(self.store_path) or self.force_cache:
            print("Caching {}".format(self.base_dataset.name))
            manager = make_server_manager()
            inqueue = manager.get_inqueue()
            outqueue = manager.get_outqueue()

            N_examples = len(self.base_dataset)
            indeces = np.arange(N_examples)
            if self.keep_existing and os.path.isfile(self.store_path):
                with ZipFile(self.store_path, 'r') as zip_f:
                    zipfilenames = zip_f.namelist()
                zipfilenames = set(zipfilenames)
                indeces = [i for i in indeces if
                        not self.naming_template.format(i) in zipfilenames]
                print("Keeping {} cached examples.".format(N_examples - len(indeces)))
                N_examples = len(indeces)
            print("Caching {} examples.".format(N_examples))
            chunk_size = 64
            index_chunks = [indeces[i:i+chunk_size]
                    for i in range(0, len(indeces), chunk_size)]
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
            memory_dict = dict()
            if hasattr(self.base_dataset, 'in_memory_keys'):
                print('Caching Labels.')
                memory_keys = self.base_dataset.in_memory_keys
                for key in memory_keys:
                    memory_dict[key] = list()

                for idx in trange(len(self.base_dataset)):
                    example = self[idx] # load cached version
                    # extract keys
                    for key in memory_keys:
                        memory_dict[key].append(example[key])

            with ZipFile(self.store_path, "a", ZIP_DEFLATED) as zipfile:
                zipfile.writestr(self._labels_name,
                        pickle.dumps(memory_dict))
            print("Finished caching.")

    def __len__(self):
        '''Number of examples in this Dataset.'''
        return len(self.base_dataset)

    @property
    def labels(self):
        '''Returns the labels associated with the base dataset, but from the
        cached source.'''
        if not hasattr(self, "_labels"):
            labels = self.fork_safe_zip.read(self._labels_name)
            labels = pickle.loads(labels)
            self._labels = labels
        return self._labels

    @property
    def root(self):
        '''Returns the root to the base dataset.'''
        return self._root

    def get_example(self, i):
        '''Given an index i, returns a example.'''

        example_name = self.naming_template.format(i)
        example_file = self.fork_safe_zip.read(example_name)

        example = pickle.loads(example_file)

        return example


class SubDataset(DatasetMixin):
    """A subset of a given dataset."""
    def __init__(self, data, subindices):
        self.data = data
        self.subindices = subindices
        try:
            len(self.subindices)
        except TypeError:
            print("Expected a list of subindices.")
            raise

    def get_example(self, i):
        """Get example and process. Wrapped to make sure stacktrace is
        printed in case something goes wrong and we are in a
        MultiprocessIterator."""
        return self.data[self.subindices[i]]

    def __len__(self):
        return len(self.subindices)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            self._labels = dict()
            labels = self.data.labels
            for k in labels:
                self._labels[k] = [labels[k][i] for i in self.subindices]
        return self._labels


class ProcessedDataset(DatasetMixin):
    """A dataset with data processing applied."""
    def __init__(self, data, process):
        self.data = data
        self.process = process

    def get_example(self, i):
        """Get example and process. Wrapped to make sure stacktrace is
        printed in case something goes wrong and we are in a
        MultiprocessIterator."""
        d = self.data.get_example(i)
        d.update(self.process(**d))
        return d

    def __len__(self):
        return len(self.data)

    @property
    def labels(self):
        # relay if data is cached
        return self.data.labels


class ConcatenatedDataset(DatasetMixin):
    """A dataset which concatenates given datasets."""
    def __init__(self, *datasets, balanced = False):
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)
        self.balanced = balanced
        if self.balanced:
            max_length = np.max(self.lengths)
            for data_idx in range(len(self.datasets)):
                data_length = len(self.datasets[data_idx])
                if data_length != max_length:
                    cycle_indices = [i % data_length for i in range(max_length)]
                    self.datasets[data_idx] = SubDataset(
                            self.datasets[data_idx], cycle_indices)
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)


    def get_example(self, i):
        """Get example and add dataset index to it."""
        did = np.where(i < self.boundaries)[0][0]
        if did > 0:
            local_i = i - self.boundaries[did-1]
        else:
            local_i = i
        example = self.datasets[did][local_i]
        example["did"] = did
        return example

    def __len__(self):
        return sum(self.lengths)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            labels = self.datasets[0].labels
            for i in range(1, len(self.datasets)):
                new_labels = self.datasets[i].labels
                for k in labels:
                    labels[k] += new_labels[k]
            self._labels = labels
        return self._labels


class ExampleConcatenatedDataset(DatasetMixin):
    '''Concatenates a list of datasets along the example axis.
    E.g.:
        dset1 = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, ...]
        dset2 = [{'a': 6, 'b': 7}, {'a': 8, 'b': 9}, ...]

        dset_conc = ExampleConcatenatedDataset(dset1, dset2)
        print(dset_conc[0])
        # {'a_0': 1, 'a_1': 6,
        #  'b_0': 2, 'b_1': 7,
        #  'a': ['a_0', 'a_1'],
        #  'b': ['b_0', 'b_1']}

    The new keys (e.g. `a_0, a_1`) are numbered in the order they are taken
    from datasets. Additionally, the original, shared key is preserved and
    returns the list of newly generated keys in correct order.
    '''

    def __init__(self, *datasets):
        '''Args:
            *datasets (DatasetMixin): All the datasets to concatenate. Each
                dataset must return a dict as example!
        '''
        assert np.all(np.equal(len(datasets[0]), [len(d) for d in datasets]))
        self.datasets = datasets
        self.set_example_pars()

    def set_example_pars(self, start=None, stop=None, step=None):
        '''Allows to manipulate the length and step of the returned example
        lists.'''

        self.example_slice = slice(start, stop, step)

    def __len__(self):
        return len(self.datasets[0])

    @property
    def labels(self):
        '''Now each index corresponds to a sequence of labels.'''
        if not hasattr(self, '_labels'):
            self._labels = dict()
            for idx, dataset in self.datasets:
                for k in dataset.labels:
                    if k in self._labels:
                        self._labels[k] += [dataset.labels[k]]
                    else:
                        self._labels[k] = [dataset.labels[k]]
        return self._labels

    def get_example(self, i):
        examples = [d[i] for d in self.datasets[self.example_slice]]

        new_examples = {}
        for idx, ex in enumerate(examples):
            for key, value in ex.items():
                new_key = '{}_{}'.format(key, idx)
                if key in new_examples:
                    new_examples[key] += [new_key]
                else:
                    new_examples[key] = [new_key]
                new_examples[new_key] = value

        return new_examples


class SequenceDataset(DatasetMixin):
    '''Wraps around a dataset and returns sequences of examples.
    Given the length of those sequences the number of available examples
    is reduced by this length times the step taken. Additionally each
    example must have a frame id `fid`, by which it can be filtered. This is to
    ensure that each frame is taken from the same video.'''

    def __init__(self, dataset, length, step=1):
        '''Args:
            dataset (DatasetMixin): Dataset from which single frame examles
                are taken.
            length (int): Length of the returned sequences in frames.
            step (int): Step between returned frames. Must be `>= 1`.

        This dataset will have `len(dataset) - length * step` examples.
        '''

        self.step = step
        self.length = length

        frame_ids = dataset.labels['fid']
        top_indeces = np.where(np.array(frame_ids) >= length * step - 1)[0]

        all_subdatasets = []
        for i in range(length * step):
            indeces = top_indeces - i
            subdset = SubDataset(dataset, indeces)
            all_subdatasets += [subdset]

        all_subdatasets = all_subdatasets[::-1]

        self.dset = ExampleConcatenatedDataset(*all_subdatasets)
        self.dset.set_example_pars(step=self.step)

    @property
    def labels(self):
        return self.dset.labels

    def __len__(self):
        return len(self.dset)

    def get_example(self, i):
        '''Retreives a list of examples starting at i.'''

        return self.dset[i]


def getSeqDataset(config):
    '''This allows to not define a dataset class, but use a baseclass and a
    `length` and `step` parameter in the supplied `config` to load and
    sequentialize a dataset.

    A config passed to edflow would the look like this:

    .. codeblock:: yaml

        dataset: edflow.data.dataset.getSeqDataSet
        model: Some Model
        iterator: Some Iterator

        seqdataset:
                dataset: import.path.to.your.basedataset,
                length: 3,
                step: 1}

    ``getSeqDataSet`` will import the base ``dataset`` and pass it to
    :class:`SequenceDataset` together with ``length`` and ``step`` to
    make the actually used dataset.

    Args:
        config (dict): An edflow config, with at least the keys
            ``seqdataset`` and nested inside it ``dataset``, ``seq_length`` and
            ``seq_step``.

    Returns:
        :class:`SequenceDataset`: A Sequence Dataset based on the basedataset.
    '''

    ks = 'seqdataset'
    base_dset = get_implementations_from_config(config[ks],
                                                ['dataset'])['dataset']
    base_dset = base_dset(config=config)

    return SequenceDataset(base_dset,
                           config[ks]['seq_length'],
                           config[ks]['seq_step'])


def JoinedDataset(dataset, key, n_joins):
    """Concat n_joins random samples based on the condition that
    example_i[key] == example_j[key] for all i,j. Key must be in labels of
    dataset."""
    labels = np.asarray(dataset.labels[key])
    unique_labels = np.unique(labels)
    index_map = dict()
    for value in unique_labels:
        index_map[value] = np.nonzero(labels == value)[0]
    join_indices = [list(range(len(dataset)))] # example_0 is original example
    prng = np.random.RandomState(1)
    for k in range(n_joins - 1):
        indices = [prng.choice(index_map[value]) for value in labels]
        join_indices.append(indices)
    datasets = [SubDataset(dataset, indices) for indices in join_indices]
    dataset = ExampleConcatenatedDataset(*datasets)

    return dataset


if __name__ == '__main__':
    from hbu_journal.data.prjoti import CachedPrjoti

    prjoti = CachedPrjoti()
    s_prjoti = SequenceDataset(prjoti, 5, 1)

    example = s_prjoti[10]
    keys = [k for k in example]

    def key(k):
        kk = k.split('_')
        if len(kk) > 1:
            idx = int(kk[1])
            kk = kk[0]
        else:
            idx = -1
            kk = kk[0]
        return kk, idx

    keys = sorted(keys, key=key)
    print(keys)

    l = 4
    for s in range(1, 4):
        s_prjoti = SequenceDataset(prjoti, l, s)
        for i in range(0, 4):
            example = s_prjoti[i]
            print('Start index: {}, Length: {}, Step: {}'.format(i, l, s))
            print([example[k] for k in example['fid']])

    print([sorted(k.keys(), key=key) for k in s_prjoti[:3]])

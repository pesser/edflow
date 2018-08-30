'''All handy dataset classes we use.'''

import os
import pickle
import tarfile

from chainer.dataset import DatasetMixin


class CachedDataset(DatasetMixin):
    """Using a Dataset of single examples creates a cached (saved to memory)
    version, which can be accessed way faster at runtime.

    To avoid creating the dataset multiple times, it is checked if the cached
    version already exists.

    Calling `__getitem__` on this class will try to retrieve the samples from
    the cached dataset to reduce the preprocessing overhead.

    The cached dataset will be stored in the root directory of the base dataset
    in the subfolder `cached`."""

    def __init__(self, dataset, force_cache=False):
        '''Given a dataset class, stores all examples in the dataset, if this
        has not yet happened.

        Args:
            dataset (object): Dataset class which defines the following
                methods:
                    - `root`: returns the path to the raw data
                    - `name`: returns the name of the dataset -> best be unique
                    - `__len__`: number of examples in the dataset
                    - `__getitem__`: returns a sindle datum
                    - `labels`: returns all labels per datum.
            force_cache (bool): If True the dataset is cached even if an
                existing, cached version is overwritten.
        '''

        self.force_cache = force_cache

        self.base_dataset = dataset
        root = dataset.root
        name = dataset.name

        self.store_path = os.path.join(root, 'cached', name)
        self.label_path = os.path.join(root, 'cached', name + '_labels.p')

        leading_zeroes = len(str(len(self)))
        self.naming_template = 'example_{:0>' + leading_zeroes + '}.p'

        os.makedirs(self.store_path, exist_ok=True)
        self.cache_dataset()

        self.tar = tarfile.open(self.store_path, 'r')

    def cache_dataset(self):
        '''Checks if a dataset is stored. If not iterates over all possible
        indeces and stores the examples in a file, as well as the labels.'''

        if not os.path.isfile(self.base_dataset) or self.force_cache:
            with tarfile.open(self.store_path + '.tar', 'w') as tar:
                for idx in range(len(self.base_dataset)):
                    example = self.base_dataset[idx]

                    pickle_name = self.naming_template.format(idx)
                    store_name = os.path.join(self.store_path, pickle_name)

                    with open(store_name, 'wb') as pickle_file:
                        pickle.dump(example, pickle_file)

                    tar.add(store_name, arcname=pickle_name)

            with open(self.label_path, 'wb') as labels_file:
                pickle.dump(self.base_dataset.labels, labels_file)

    def __len__(self):
        '''Number of examples in this Dataset.'''
        return len(self.base_dataset)

    @property
    def labels(self):
        with open(self.label_path, 'r') as labels_file:
            labels = pickle.load(labels_file)
        return labels

    def get_example(self, i):
        '''Given an index i, returns a example.'''

        example_name = self.naming_template.format(i)
        example_file = self.tar.extractfile(example_name)

        example = pickle.load(example_file)

        return example

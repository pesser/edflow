'''All handy dataset classes we use.'''

import os
import pickle
from tqdm import trange
from zipfile import ZipFile, ZIP_DEFLATED  # , ZIP_BZIP2, ZIP_LZMA

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
            print('Caching dataset. This might take a while.')
            with ZipFile(self.store_path, 'w', ZIP_DEFLATED) as zip_f:
                for idx in trange(len(self.base_dataset), desc='Example'):
                    if idx > 100:
                        break
                    example = self.base_dataset[idx]

                    pickle_name = self.naming_template.format(idx)
                    pickle_bytes = pickle.dumps(example)

                    zip_f.writestr(pickle_name, pickle_bytes)

            print('Caching Labels.')
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
        example_file = self.zip.read(example_name)

        example = pickle.loads(example_file)

        return example

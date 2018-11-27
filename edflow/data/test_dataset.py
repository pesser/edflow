import unittest
import numpy as np
import os

# The base class
from edflow.data.dataset import DatasetMixin
# Cached Classes
from edflow.data.dataset import CachedDataset, \
                                PathCachedDataset
# Class definitions
from edflow.data.dataset import SubDataset, \
                                LabelDataset, \
                                ProcessedDataset, \
                                ExtraLabelsDataset, \
                                ConcatenatedDataset, \
                                ExampleConcatenatedDataset, \
                                SequenceDataset, \
                                UnSequenceDataset, \
                                RandomlyJoinedDataset, \
                                DataFolder
# Function Definitions
from edflow.data.dataset import JoinedDataset
# Config getters
from edflow.data.dataset import getSeqDataset, \
                                getDebugDataset


class TestDset(DatasetMixin):
    def __init__(self, size=5, n_labels=3):
        self.data = []
        self.labels = {'l'+str(i): [] for i in range(n_labels)}
        self.labels['fid'] = []
        self.labels['join'] = []

        for i in range(size):
            example = {'a': np.array([1, 2, 3])+i,
                       'b': np.array([4, 5, 6])+i,
                       'c': np.array([7, 8, 9])+i,
                       'fid': i,
                       'join': i // 5}
            self.data += [example]

            self.labels['fid'] += [i]
            self.labels['join'] += [i // 5]
            for l in range(n_labels):
                self.labels['l'+str(l)] += ['label-{}-{}'.format(l, i)]

    def get_example(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


class TestGetitem(unittest.TestCase):
    '''Calls getitem on all dataset classes and tests if no errors occur,
    as well as if the returned example is a dict.'''

    def get_items(self, dset, indices=[0, -1, slice(0, -1, 2)]):
        for idx in indices:
            example = dset[idx]

            self.assertIsNotNone(example)
            if not isinstance(idx, slice):
                self.assertTrue(isinstance(example, dict))
            else:
                for sub_ex in example:
                    self.assertTrue(isinstance(sub_ex, dict))

    def test_SubDataset(self):
        TD = TestDset()
        D = SubDataset(TD, [1, 3])
        self.get_items(D)

    def test_LabelDataset(self):
        TD = TestDset()
        D = LabelDataset(TD)
        self.get_items(D)

    def test_ProcessedDataset(self):
        TD = TestDset()

        def proc(**kwargs):
            for k, v in kwargs.items():
                kwargs[k] = v + 1
            return kwargs

        D = ProcessedDataset(TD, proc)
        self.get_items(D)

    def test_ExtraLabelsDataset(self):
        TD = TestDset()

        def proc(data, idx):
            return {'nl1': 'new_label1', 'nl2': 'new_label2'}

        D = ExtraLabelsDataset(TD, proc)
        self.get_items(D)

    def test_ConcatenatedDataset(self):
        TD1 = TestDset(size=5, n_labels=2)
        TD2 = TestDset(size=3, n_labels=4)

        D = ConcatenatedDataset(TD1, TD2)
        self.get_items(D)

    def test_ExampleConcatenatedDataset(self):
        TD1 = TestDset(size=5, n_labels=2)
        TD2 = TestDset(size=5, n_labels=2)

        D = ExampleConcatenatedDataset(TD1, TD2)
        self.get_items(D)

    def test_SequenceDataset(self):
        TD = TestDset(size=20)

        i = 0
        for l in [3, 5]:
            for s in [1, 3]:
                with self.subTest(i):
                    D = SequenceDataset(TD, l, s)
                    self.get_items(D)
                    i += 1

    def test_UnSequenceDataset(self):
        TD = TestDset(size=20)
        STD = SequenceDataset(TD, 4)

        D = UnSequenceDataset(STD)
        self.get_items(D)

    def test_RandomlyJoinedDataset(self):
        TD = TestDset(size=10)

        D = RandomlyJoinedDataset(TD, 'join', 2)
        self.get_items(D)

    def test_DataFolder(self):
        root = '/tmp/df123'
        setup_datafolder_env(root)

        def label(path):
            if '.npz' == path[-4:]:
                return {'name': path}

        def read(path):
            example = np.load(path)
            return example

        D = DataFolder(root, read, label, ['name'])
        self.get_items(D)

        teardown_datafolder_env(root)


def setup_datafolder_env(root):
    os.makedirs(root, exist_ok=True)
    TD = TestDset()

    for i in range(len(TD)):
        example = TD[i]
        np.savez(os.path.join(root, str(i)), **example)


def teardown_datafolder_env(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


if __name__ == '__main__':
    unittest.main()

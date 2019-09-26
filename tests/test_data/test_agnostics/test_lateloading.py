import pytest
from edflow.debug import DebugDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.believers.sequence import SequenceDataset
from edflow.data.agnostics.late_loading import LateLoadingDataset


class Base(DatasetMixin):
    def get_example(i, idx):
        def _routine():
            return idx

        return {'image': _routine}

    def __len__(self):
        return 10


def test_late_loading_dset():

    B = Base()

    L = LateLoadingDataset(B)

    d = L[0]
    assert d == {'image': 0, 'index_': 0}
    d = L[5]
    assert d == {'image': 5, 'index_': 5}

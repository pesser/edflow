import pytest

from edflow.data.dataset import DatasetMixin
from edflow.debug import DebugDataset


def test_dset_mxin():
    class MyDset(DatasetMixin):
        def get_example(self, idx):
            return {"a": 1}

        def __len__(self):
            # Cannot be guessed from get_example!
            return 10

    D = MyDset()
    ex = D[0]
    assert "index_" in ex
    assert "a" in ex

    with pytest.raises(Exception):
        # Must compare len with idx
        ex[100]


def test_dset_mxin_app_labels():
    class MyDset(DatasetMixin):
        def __init__(self):
            self.labels = {"l": [1, 2, 3]}
            self.append_labels = True

        def get_example(self, idx):
            return {"a": 1}

        def __len__(self):
            return len(self.labels["l"])

    D = MyDset()
    ex = D[0]
    assert "l" in ex
    assert "a" in ex

    with pytest.raises(Exception):
        ex[100]

    D.append_labels = False
    ex = D[0]
    assert "l" not in ex
    assert "a" in ex

    with pytest.raises(Exception):
        ex[100]


def test_dset_mxin_data_attr():
    class MyDset(DatasetMixin):
        def __init__(self):
            self.data = DebugDataset(size=10)

    D = MyDset()
    ex = D[0]
    assert "val" in ex
    assert "other" in ex
    assert "index_" in ex

    with pytest.raises(Exception):
        ex[100]

    lbs = D.labels
    l = lbs["lbl1"][0]

    with pytest.raises(Exception):
        lbs["lbl1"][100]


def test_dset_mxin_data_attr_app_labels():
    class MyDset(DatasetMixin):
        def __init__(self):
            self.data = DebugDataset(size=10)
            self.append_labels = True

    D = MyDset()
    ex = D[0]
    assert "val" in ex
    assert "other" in ex
    assert "index_" in ex
    assert "lbl1" in ex
    assert "lbl2" in ex

    with pytest.raises(Exception):
        ex[100]

    lbs = D.labels
    l = lbs["lbl1"][0]
    l = lbs["lbl2"][0]

    with pytest.raises(Exception):
        lbs["lbl1"][100]
        lbs["lbl2"][100]


def test_dset_mxin_ops():
    """Basically test the ConcatenatedDataset"""

    class MyDset(DatasetMixin):
        def __init__(self):
            self.data = DebugDataset(size=10)

    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    D3 = D1 + D2
    assert len(D3) == 20
    D3[13]
    D3 = D2 + D1
    assert len(D3) == 20
    D3[13]

    D4 = 3 * D1
    assert len(D4) == 30
    D4[13]
    D4[23]
    D4 = D1 * 3
    assert len(D4) == 30
    D4[13]
    D4[23]


if __name__ == "__main__":
    test_dset_mxin()
    test_dset_mxin_ops()
    test_dset_mxin_data_attr()
    test_dset_mxin_app_labels()
    test_dset_mxin_data_attr_app_labels()

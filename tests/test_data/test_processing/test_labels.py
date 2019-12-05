import pytest
import numpy as np
from edflow.debug import DebugDataset
from edflow.data.processing.labels import LabelDataset, ExtraLabelsDataset
from edflow.util import set_value


def test_label_dset():
    D = DebugDataset(size=10)

    L = LabelDataset(D)

    assert L[0] == {"label1": 0, "label2": 0, "index_": 0, "base_index_": 0}

    assert len(L) == 10
    assert len(L.labels["label1"]) == 10


def test_extra_labels():
    D = DebugDataset(size=10)
    D.append_labels = True

    E = ExtraLabelsDataset(D, lambda dset, idx: {"new": idx})

    de = E[0]
    ref = D[0]
    set_value(ref, "labels_/new", 0)

    assert de == ref

    assert "new" in E.labels
    assert np.all(E.labels["new"] == np.arange(10))

import pytest

from edflow.debug import DebugDataset
from edflow.data.processing.processed import ProcessedDataset


def test_processed_dset():
    D = DebugDataset(size=10)

    P = ProcessedDataset(D, lambda val, index_, other: {"val": val ** 2})

    assert len(D) == len(P)
    assert len(D.labels["label1"]) == len(P.labels["label1"])

    dp = P[0]
    ref = D[0]
    ref["val"] = ref["val"] ** 2

    assert dp == ref


def test_processed_dset_no_update():
    D = DebugDataset(size=10)

    P = ProcessedDataset(D, lambda val, index_, other: {"val": val ** 2}, False)

    assert len(D) == len(P)
    assert len(D.labels["label1"]) == len(P.labels["label1"])

    dp = P[0]

    assert dp == {"val": 0, "index_": 0}

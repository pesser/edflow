from edflow.data.agnostics.concatenated import ConcatenatedDataset
from edflow.data.agnostics.concatenated import ExampleConcatenatedDataset

from edflow.debug import DebugDataset


def test_ConcatenatedDataset():
    D = DebugDataset(size=10)
    C = ConcatenatedDataset(D, D)

    assert len(C) == 20
    ref = D[2]
    d = C[2]
    assert d == ref
    d = C[12]
    assert d == ref

    assert len(C.labels["labels1"]) == 20

    lref = D.labels['label1'][2]
    l = C.labels["label1"][2]
    assert l == lref

    l = C.labels[12]
    assert l == lref

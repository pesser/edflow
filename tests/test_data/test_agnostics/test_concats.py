import pytest
import numpy as np

from edflow.data.agnostics.concatenated import ConcatenatedDataset
from edflow.data.agnostics.concatenated import ExampleConcatenatedDataset

from edflow.debug import DebugDataset


def test_ConcatenatedDataset():
    D = DebugDataset(size=10)
    C = ConcatenatedDataset(D, D)

    assert len(C) == 20
    ref = D[2]
    ref["dataset_index_"] = 0
    d = C[2]
    assert d == ref
    d = C[12]
    ref["dataset_index_"] = 1
    ref["index_"] = 12
    assert d == ref

    assert len(C.labels["label1"]) == 20

    lref = D.labels["label1"][2]
    l = C.labels["label1"][2]
    assert l == lref

    l = C.labels["label1"][12]
    assert l == lref


def test_ConcatenatedDataset_balanced():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=20)
    C = ConcatenatedDataset(D1, D2, balanced=True)

    assert len(C) == 40
    ref = D1[2]
    ref["dataset_index_"] = 0
    ref["index_"] = 2
    d = C[2]
    assert d == ref
    ref["index_"] = 12
    d = C[12]
    assert d == ref

    ref = D2[2]
    ref["dataset_index_"] = 1
    ref["index_"] = 22
    d = C[22]
    assert d == ref
    ref = D2[12]
    ref["dataset_index_"] = 1
    ref["index_"] = 32
    d = C[32]
    assert d == ref

    assert len(C.labels["label1"]) == 40

    lref = D1.labels["label1"][2]
    l = C.labels["label1"][2]
    assert l == lref
    l = C.labels["label1"][12]
    assert l == lref

    l = C.labels["label1"][12]
    assert l == lref


def test_ExampleConcatenatedDataset():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    E = ExampleConcatenatedDataset(D1, D2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2, 2], "index_": 2, "other": [2, 2]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i, i] for i in range(10)])

    D3 = DebugDataset(size=20)

    with pytest.raises(AssertionError):
        ExampleConcatenatedDataset(D1, D3)


def test_ExampleConcatenatedDataset_step():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    E = ExampleConcatenatedDataset(D1, D2)
    E.set_example_pars(step=2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2], "index_": 2, "other": [2]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i, i] for i in range(10)])

    D3 = DebugDataset(size=20)

    with pytest.raises(AssertionError):
        ExampleConcatenatedDataset(D1, D3)

    D4 = DebugDataset(size=10)
    D5 = DebugDataset(size=10)

    E = ExampleConcatenatedDataset(D1, D2, D4, D5)
    E.set_example_pars(step=2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2, 2], "index_": 2, "other": [2, 2]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i, i] for i in range(10)])


def test_ExampleConcatenatedDataset_slicing():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10, offset=1)
    D3 = DebugDataset(size=10, offset=2)
    D4 = DebugDataset(size=10, offset=3)

    E = ExampleConcatenatedDataset(D1, D2, D3, D4)
    E.set_example_pars(start=1, step=2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2+1, 2+3], "index_": 2, "other": [2+1, 2+3]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i+1, i+3] for i in range(10)])

    E.set_example_pars(start=0, stop=-1, step=2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2, 2+2], "index_": 2, "other": [2, 2+2]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i, i+2] for i in range(10)])

    E.set_example_pars(start=1, stop=-1, step=2)
    assert len(E) == 10
    d = E[2]

    assert d == {"val": [2+1], "index_": 2, "other": [2+1]}

    assert len(E.labels["label1"]) == 10
    assert np.all(E.labels["label1"] == [[i+1] for i in range(10)])

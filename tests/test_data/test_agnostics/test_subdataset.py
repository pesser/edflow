import pytest
import numpy as np
from edflow.debug import DebugDataset
from edflow.data.agnostics.subdataset import SubDataset


def test_sub():
    D = DebugDataset(10)

    I = np.array([9, 1, 2, 4, 3, 5, 7, 6, 8, 0])

    S = SubDataset(D, I)

    ref0 = {"val": 9, "other": 9, "index_": 0}
    ref2 = {"val": 2, "other": 2, "index_": 2}
    ref6 = {"val": 7, "other": 7, "index_": 6}

    assert S[0] == ref0
    assert S[2] == ref2
    assert S[6] == ref6

    assert all(S.labels["label1"] == I)

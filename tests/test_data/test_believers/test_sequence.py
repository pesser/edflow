import pytest
import numpy as np

from edflow.data.believers.sequence import SequenceDataset, UnSequenceDataset
from edflow.data.believers.sequence import getSeqDataset
from edflow.debug import DebugDataset



def test_sequence_dset_vanilla():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)
    D3 = DebugDataset(size=10)

    D = D1 + D2 + D3

    S = SequenceDataset(D, 3, fid_key='label1')

    assert len(S) == (3 * (10-2))


def test_sequence_dset_offset_fid():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)
    D3 = DebugDataset(size=10, offset=3)
    D4 = DebugDataset(size=10, offset=3)

    D = D1 + D2 + D3 + D4

    labels = np.array(D.labels['label1'])
    diffs = labels[1:] - labels[:-1]
    idxs = np.where(diffs <= 0)[0] + 1

    with pytest.raises(ValueError):
        S = SequenceDataset(D, 3, fid_key='label1')


def test_seq_wrong_fid_dtype():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    D = D1 + D2
    D.labels['label1'] = np.array(D.labels['label1'], dtype=float)

    with pytest.raises(TypeError):
        S = SequenceDataset(D, 3, fid_key='label1')


def test_seq_wrong_fid_len():
    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    D = D1 + D2
    D.labels['label1'] = np.array(D.labels['label1'])[:10]

    with pytest.raises(ValueError):
        S = SequenceDataset(D, 3, fid_key='label1')

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
    assert len(S.labels['label1']) == (3 * (10-2))

    d = S[0]
    ref = {'val': [0, 1, 2], 'other': [0, 1, 2], 'index_': 0,
           'dataset_index_': [0, 0, 0]}

    assert d == ref

    l = np.array(S.labels['label1'][:3])
    refl = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    assert np.all(l == refl)


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


# =============================================================================


def test_unseq_vanilla():

    D1 = DebugDataset(size=10)
    D2 = DebugDataset(size=10)

    D = D1 + D2

    S = SequenceDataset(D, 3, fid_key='label1')

    U = UnSequenceDataset(S)

    print(np.shape(U.labels['label1']))

    assert len(U) == len(S) * 3
    assert len(U.labels['label1']) == len(S.labels['label1']) * 3

    du = U[[0, 1, 2]]
    ds = S[0]

    for k in ds.keys():
        if k == 'index_':
            continue
        for i in range(3):
            print(k, i)
            val_u = du[i][k]
            val_s = ds[k][i]

            assert val_s == val_u


# =============================================================================


def test_get_seq():
    config = {
            'dataset': 'edflow.data.believers.sequence.getSeqDataset',
            'seqdataset': {'dataset': 'edflow.debug.ConfigDebugDataset',
                           'length': 3,
                           'step': 1,
                           'fid_key': 'label1'},
            'size': 10
            }

    S1 = getSeqDataset(config)

    D = DebugDataset(size=10)
    S2 = SequenceDataset(D, 3, fid_key='label1')

    assert len(S1) == len(S2)
    assert len(S1.labels['label1']) == len(S2.labels['label1'])
    assert len(S1) == len(S1.labels['label1'])

    s1 = S1[0]
    s2 = S2[0]

    assert s1 == s2

    assert np.all(S1.labels['label1'] == S2.labels['label1'])

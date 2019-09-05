import pytest
import os

from edflow.data.agnostics.csv_dset import CsvDataset


def test_load_and_output():
    root = os.path.dirname(__file__)
    filename = os.path.join(root, 'test.csv')

    D = CsvDataset(filename)
    print(D.data)

    d = D[2]
    print(d)

    assert d == {'a': 2.1, 'b': 2.2, 'c': 2.3, 'index_': 2}

    assert all(D.labels['a'] == [0.1, 1.1, 2.1, 3.1])
    assert all(D.labels['b'] == [0.2, 1.2, 2.2, 3.2])
    assert all(D.labels['c'] == [0.3, 1.3, 2.3, 3.3])

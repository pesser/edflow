from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.agnostics.subdataset import SubDataset
import numpy as np

from edflow.data.dataset_mixin import ConcatenatedDataset


class ExampleConcatenatedDataset(DatasetMixin):
    """Concatenates a list of datasets along the example axis.

    .. note::
        All datasets must be of same length and must return examples with the
        same keys and behind those keys with the same type and shape.

    If dataset A returns examples of form ``{'a': x, 'b': x}`` and dataset
    B of form ``{'a': y, 'b': y}`` the ``ExampleConcatenatedDataset(A, B)``
    return examples of form ``{'a': [x, y], 'b': [x, y]}``.

    """

    def __init__(self, *datasets):
        """
        Parameters
        ----------
        *datasets : DatasetMixin
            All the datasets to concatenate.
        """
        assert np.all(np.equal(len(datasets[0]), [len(d) for d in datasets]))
        self.datasets = datasets
        self.set_example_pars()

    def set_example_pars(self, start=None, stop=None, step=None):
        """Allows to manipulate the length and step of the returned example
        lists."""

        self.example_slice = slice(start, stop, step)
        self.slice_changed = True

    def __len__(self):
        return len(self.datasets[0])

    @property
    def labels(self):
        """Now each index corresponds to a sequence of labels."""
        if not hasattr(self, "_labels") or self.slice_changed:
            self._labels = dict()
            for idx, dataset in enumerate(self.datasets[self.example_slice]):
                for k in dataset.labels:
                    if k in self._labels:
                        self._labels[k] += [dataset.labels[k]]
                    else:
                        self._labels[k] = [dataset.labels[k]]

            for k, v in self._labels.items():
                v = np.array(v)
                # sometimes numpy arrays or lists are given as labels
                # their axes stay at the same positions.
                trans = [1, 0] + list(range(2, len(v.shape)))
                self._labels[k] = v.transpose(*trans)
            self.slice_changed = False
        return self._labels

    def get_example(self, i):
        examples = [d[i] for d in self.datasets[self.example_slice]]

        new_examples = {}
        for idx, ex in enumerate(examples):
            for key, value in ex.items():
                if key in new_examples:
                    new_examples[key] += [value]
                else:
                    new_examples[key] = [value]

        return new_examples


class DisjunctExampleConcatenatedDataset(DatasetMixin):
    """Concatenates a list of disjunct datasets.

    .. note::
        All datasets must be of same length and labels and returned keys must
        be disjunct. If labels or keys are not disjunct, set the optional
        parameter `disjunct` to False, to use the value of the last dataset
        containing the key. Datasets can have different length if `same_length`
        is set to False.

    If dataset A returns examples of form ``{'a': w, 'b': x}`` and dataset
    B of form ``{'c': y, 'd': z}`` the ``DisjunctExampleConcatenatedDataset(A, B)``
    return examples of form ``{'a': w, 'b': x, 'c': y, 'd': z}``.

    """

    def __init__(self, *datasets, disjunct=True, same_length=True):
        """
        Parameters
        ----------
        *datasets : DatasetMixin
            All the datasets to concatenate.
        disjunct : bool
            labels and returned keys do not have to be disjunct. Last
            datasetet overwrites values
        same_length : bool
            Datasets do not have to be of same length. Concatenated dataset
            has length of smallest dataset.
        """
        self.disjunct = disjunct
        self.same_length = same_length
        if self.same_length:
            assert np.all(np.equal(len(datasets[0]), [len(d) for d in datasets]))
        else:
            self.length = min([len(d) for d in datasets])
        self.datasets = datasets
        self.labels = {}
        for d in datasets:
            if self.disjunct:
                assert set(self.labels).isdisjoint(d.labels)
            self.labels.update(d.labels)

    def __len__(self):
        if self.same_length:
            return len(self.datasets[0])
        else:
            return self.length

    def get_example(self, i):
        examples = {}
        for d in self.datasets:
            example_new = d[i]
            if self.disjunct:
                assert set(examples).difference({"index_"}).isdisjoint(example_new)
            examples.update(example_new)

        return examples

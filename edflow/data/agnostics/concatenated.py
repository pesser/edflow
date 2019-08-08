from edflow.data.dataset_mixin import DatasetMixin
import numpy as np


class ConcatenatedDataset(DatasetMixin):
    """A dataset which concatenates given datasets."""

    def __init__(self, *datasets, balanced=False):
        """
        Parameters
        ----------
        *datasets : DatasetMixin
            All datasets we want to concatenate
        balanced : bool
            If ``True`` all datasets are padded to the length of the longest
            dataset. Padding is done in a cycled fashion.
        """
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)
        self.balanced = balanced
        if self.balanced:
            max_length = np.max(self.lengths)
            for data_idx in range(len(self.datasets)):
                data_length = len(self.datasets[data_idx])
                if data_length != max_length:
                    cycle_indices = [i % data_length for i in range(max_length)]
                    self.datasets[data_idx] = SubDataset(
                        self.datasets[data_idx], cycle_indices
                    )
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)

    def get_example(self, i):
        """Get example and add dataset index to it."""
        did = np.where(i < self.boundaries)[0][0]
        if did > 0:
            local_i = i - self.boundaries[did - 1]
        else:
            local_i = i
        example = self.datasets[did][local_i]
        example["dataset_index_"] = did
        return example

    def __len__(self):
        return sum(self.lengths)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            labels = dict(self.datasets[0].labels)
            for i in range(1, len(self.datasets)):
                new_labels = self.datasets[i].labels
                for k in labels:
                    labels[k] = labels[k] + new_labels[k]
            self._labels = labels
        return self._labels


class ExampleConcatenatedDataset(DatasetMixin):
    """Concatenates a list of datasets along the example axis.
    .. Warning:: Docu is wrong!
    E.g.: \n
    \tdset1 = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, ...] \n
    \tdset2 = [{'a': 6, 'b': 7}, {'a': 8, 'b': 9}, ...] \n

    \tdset_conc = ExampleConcatenatedDataset(dset1, dset2) \n
    \tprint(dset_conc[0]) \n
    \t# {'a_0': 1, 'a_1': 6, \n
    \t#  'b_0': 2, 'b_1': 7, \n
    \t#  'a': ['a_0', 'a_1'], \n
    \t#  'b': ['b_0', 'b_1']} \n

    The new keys (e.g. `a_0, a_1`) are numbered in the order they are taken
    from datasets. Additionally, the original, shared key is preserved and
    returns the list of newly generated keys in correct order.
    """

    def __init__(self, *datasets):
        """
        Parameters
        ----------
        *datasets : DatasetMixin
            All the datasets to concatenate. Each dataset must return a dict
            as example!
        """
        assert np.all(np.equal(len(datasets[0]), [len(d) for d in datasets]))
        self.datasets = datasets
        self.set_example_pars()

    def set_example_pars(self, start=None, stop=None, step=None):
        """Allows to manipulate the length and step of the returned example
        lists."""

        self.example_slice = slice(start, stop, step)

    def __len__(self):
        return len(self.datasets[0])

    @property
    def labels(self):
        """Now each index corresponds to a sequence of labels."""
        if not hasattr(self, "_labels"):
            self._labels = dict()
            for idx, dataset in enumerate(self.datasets):
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

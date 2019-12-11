from edflow.data.dataset_mixin import DatasetMixin
import numpy as np


class LabelDataset(DatasetMixin):
    """A label only dataset to avoid loading unnecessary data."""

    def __init__(self, data):
        """
        Parameters
        ----------
        data : DatasetMixin
            Some dataset where we are only interested in the labels.
        """

        self.data = data
        self.keys = sorted(self.data.labels.keys())

    def get_example(self, i):
        """Return only labels of example."""
        example = dict((k, self.data.labels[k][i]) for k in self.keys)
        example["base_index_"] = i
        return example


class ExtraLabelsDataset(DatasetMixin):
    """A dataset with extra labels added."""

    def __init__(self, data, labeler):
        """
        Parameters
        ----------
        data : DatasetMixin
            Some Base dataset you want to add labels to
        labeler : Callable
            Must accept two arguments: a ``Dataset`` and an index ``i`` and
            return a dictionary of labels to add or overwrite. For all indices
            the keys in the returned ``dict`` must be the same and the type
            and shape of the values at those keys must be the same per key.
        """
        self.data = data
        self._labeler = labeler
        self._new_keys = sorted(self._labeler(self.data, 0).keys())
        self._new_labels = dict()
        for k in self._new_keys:
            self._new_labels[k] = [None for _ in range(len(self.data))]
        for i in range(len(self.data)):
            new_labels = self._labeler(self.data, i)
            for k in self._new_keys:
                self._new_labels[k][i] = new_labels[k]
        self._labels = dict(self.data.labels)
        self._labels.update(self._new_labels)

        labels = {}
        for k, v in self._labels.items():
            labels[k] = np.array(v)
        self._labels = labels

        self.append_labels = True

    @property
    def labels(self):
        return self._labels

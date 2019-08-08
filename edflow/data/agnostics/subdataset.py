from edflow.data.dataset_mixin import DatasetMixin


class SubDataset(DatasetMixin):
    """A subset of a given dataset."""

    def __init__(self, data, subindices):
        self._data = data
        self.subindices = subindices
        try:
            len(self.subindices)
        except TypeError:
            print("Expected a list of subindices.")
            raise

    def get_example(self, i):
        """Get example and process. Wrapped to make sure stacktrace is
        printed in case something goes wrong and we are in a
        MultiprocessIterator."""
        return self._data[self.subindices[i]]

    def __len__(self):
        return len(self.subindices)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            self._labels = dict()
            labels = self.data.labels
            for k in labels:
                self._labels[k] = [labels[k][i] for i in self.subindices]
        return self._labels

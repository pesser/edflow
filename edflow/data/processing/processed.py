from edflow.data.dataset_mixin import DatasetMixin


class ProcessedDataset(DatasetMixin):
    """A dataset with data processing applied."""

    def __init__(self, data, process, update=True):
        self.data = data
        self.process = process
        self.update = update

    def get_example(self, i):
        """Get example and process. Wrapped to make sure stacktrace is
        printed in case something goes wrong and we are in a
        MultiprocessIterator."""
        d = self.data[i]
        p = self.process(**d)
        if self.update:
            d.update(p)
            return d
        else:
            return p

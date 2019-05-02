from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.data.dataset import DatasetMixin


class DebugModel(object):
    def __init__(self, *a, **k):
        pass

    def __call__(self, *args, **kwargs):
        pass


def debug_step_op(model, *args, **kwargs):
    if "val" not in kwargs:
        return None
    else:
        return kwargs["val"]


class DebugIterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def step_ops(self):
        return [debug_step_op]


class DebugDataset(DatasetMixin):
    def __init__(self, size=100, *args, **kwargs):
        self.size = size

    def get_example(self, i):
        if i < self.size:
            return {"val": i, "index_": i, "other": i}
        else:
            raise IndexError("Out of bounds")

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            self._labels = {
                k: [i for i in range(self.size)] for k in ["index_", "other"]
            }
        return self._labels

    def __len__(self):
        return self.size

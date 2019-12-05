from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.data.dataset import DatasetMixin
import numpy as np


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
    def __init__(
        self,
        size=100,
        offset=0,
        other_labels=False,
        other_ex_keys=False,
        *args,
        **kwargs
    ):
        self.size = size
        self.offset = offset
        self.other_labels = other_labels
        self.other_ex_keys = other_ex_keys

    def get_example(self, i):
        if i < self.size:
            i += self.offset
            if self.other_ex_keys:
                ex = {"val_other": i, "other_other": i}
            else:
                ex = {"val": i, "other": i}
            return dict({"index_": i}, **ex)
        else:
            raise IndexError("Out of bounds")

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            if self.other_labels:
                keys = ["label1_other", "label2_other"]
            else:
                keys = ["label1", "label2"]
            self._labels = {
                k: np.array([i + self.offset for i in range(self.size)]) for k in keys
            }
        return self._labels

    def __len__(self):
        return self.size


class ConfigDebugDataset(DebugDataset):
    def __init__(self, config):
        super().__init__(**config)

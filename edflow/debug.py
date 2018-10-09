from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.data.dataset import DatasetMixin


class DebugModel(object):
    def __init__(self, *a, **k):
        pass

    def __call__(self, *args, **kwargs):
        pass


def debug_step_op(model, *args, **kwargs):
    if 'val' not in kwargs:
        return None
    else:
        return kwargs['val']


class DebugIterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def step_ops(self):
        return [debug_step_op]


class DebugDataset(DatasetMixin):
    def __init__(self, *args, **kwargs):
        pass

    def get_example(self, i):
        return {'val': i, 'index_': i}

    def __len__(self):
        return 100

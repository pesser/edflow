import pytest, subprocess
from common import run_edflow, fullname
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.iterators.batches import DatasetMixin

class Model(object):
    def __init__(self, config):
        self.config = config

class Iterator(PyHookedModelIterator):
    def initialize(self, *args, **kwargs):
        raise Exception("TestAbort")

class Dataset(DatasetMixin):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 1000

    def get_example(self, i):
        return {"foo": 0}


def run_test():
    config = dict()
    config["model"] = fullname(Model)
    config["iterator"] = fullname(Iterator)
    config["dataset"] = fullname(Dataset)
    config["batch_size"] = 16
    config["num_steps"] = 100
    run_edflow("0024", config)

def test():
    subprocess.run('python -c "from test_hanging_process import run_test; run_test()"',
            shell = True, check = False,
            timeout = 60)

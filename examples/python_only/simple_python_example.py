"""
When using edflow, you are not limited to training models with pytorch or
tensorflow dependencies. The underlying framework of ``edflow`` is written in
pure python, which allows you to run any python code you want.

The following example is only meant to show you what necessary and handy
building blocks you can use. Nothing fancy is going to happen.

Run this example using the following lines of code:
.. codeblock:: bash

    # Set this variable
    EXPATH=/path/to/your/edflow/examples/python_only

    cd $EXPATH
    edflow -n py_example -t config.yaml
"""


from edflow.data.dataset import DatasetMixin, PRNGMixin
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.hook import Hook
from edflow.hooks.util_hooks import IntervalHook
from edflow.custom_logging import get_logger


class Memory(object):
    """This class defines the handling of data storage used by our model
    :class:`Model` defined below."""

    def __init__(self, name, size, mem_type):
        self.name = name
        self.size = size

        assert mem_type in ["fifo", "lifo", "fifo_inv", "lifo_inv"]
        self.mt = mem_type

        self._memory = []

    def __add__(self, other):
        assert isinstance(
            other, (list, Memory)
        ), "can only add list or Memory to Memory"

        if isinstance(other, Memory):
            other = other._memory

        if "_inv" in self.mt:
            new_mem = other + self._memory
        else:
            new_mem = self._memory + other

        if len(new_mem) >= self.size:
            if "_inv" in self.mt:
                new_mem = new_mem[:-1]
            else:
                new_mem = new_mem[:-2] + new_mem[-1:]

        new_obj = Memory(self.name, self.size, self.mt)
        new_obj._memory = new_mem

        return new_obj

    def __iadd__(self, other):
        self = self + other

        return self

    def __str__(self):
        if len(self) > 3:
            exs = ", ".join([str(e) for e in self._memory[:2]])
            exs += "..., "
            exs += str(self._memory[-1])
        else:
            exs = ", ".join([str(e) for e in self._memory])
        return "{}: {} examples, [{}]".format(self.name, len(self), exs)

    def __len__(self):
        return len(self._memory)


class Model(object):
    def __init__(self, config):
        """Our model simply expects any keyword argument input when being
        called and stores the values in an internal memory.

        This memory is setup on construction of our model and using the config
        we can define, how the interal memory is supposed to work, i.e. should
        it have a certain size and how should inputs be handled if the memory
        is full.

        Args:
            config (dict): The config dict should contain the following keys:
                mem_size (int): Defines how many examples per keyword are
                    stored in the internal memory of the model. If
                    ``mem_size <= 0`` the memory is not limited.
                queue_type (str): Defines how inputs are appendend to the
                    memory and what happens when the memory is full.
                    The possible options are:
                        - ``fifo``: First in, first out. Examples are appended
                          to the memory. When it is full the first example is
                          kicked out.
                        - ``lifo``: Last in, first out. Examples are appended
                          to the memory. When the memory is full the last
                          example is kicked out.
                        - ``fifo_inv``: Like ``fifo`` but examples are
                          prepended.
                        - ``lifo_inv``: Like ``lifo`` but examples are
                          prepended.
        """

        self.q_type = qt = config.get("queue_type", "fifo")
        self.m_size = ms = config.get("mem_size", -1)

        self._memories = {}

    def __call__(self, **kwargs):
        """Remember examples. If the kwarg has not yet been seen add a new
        queue with that name, otherwise add the example to the exisiting queue.
        """

        for name, value in kwargs.items():
            if name == "index_":
                continue
            if name not in self._memories:
                self._memories[name] = Memory(name, self.m_size, self.q_type)

            self._memories[name] += list(value)

    def __str__(self):
        keys = sorted(self._memories.keys())

        return "\n".join([str(self._memories[k]) for k in keys])


class SomeData(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        """This Dataset randomly generates examples of type ``int``, behind
        keys defined in the :attr:`config`.

        Args:
            config (dict): The config should contain the following keys:
                - ``example_names`` (list): A list containing all possible
                  example keys.
                - ``num_examples`` (int): The length of this dataset
        """

        self.possible_keys = config.get("example_names", ["a", "b", "c"])

        assert all(
            [isinstance(k, str) for k in self.possible_keys]
        ), "Keys defined by `example_names` must all be strings!"

        self.num_examples = config.get("num_examples", 1000)

    def get_example(self, idx):
        keys = self.possible_keys

        example = {}
        for k in keys:
            example[k] = self.prng.randint(-100, 100)

        return example

    def __len__(self):
        return self.num_examples


class PrintMemHook(Hook):
    """This Hook let's us see how the memories of our model grows."""

    def __init__(self, model, gs_getter=None):
        self.model = model
        self.gs_getter = gs_getter

        self.logger = get_logger(self)

    def after_step(self, step, *args, **kwargs):
        if self.gs_getter is not None:
            step = self.gs_getter()

        self.logger.info("{}:\n{}".format(step, self.model))


# TODO(jhaux) Add a visualization hook!
class PyRunner(PyHookedModelIterator):
    """The Runner plugs together dataset and model and does noting or real
    importance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        PM = PrintMemHook(self.model, self.get_global_step)
        self.hooks += [IntervalHook([PM], 1)]

    def step_ops(self):
        def default_op(model, **kwargs):
            model(**kwargs)

        return default_op

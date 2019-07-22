Tutorial
========

PyTorch
----------

We think that a good way to learn ``edflow`` is by example(s).
Thus, we translate a simple classification code (the introductory ``PyTorch``
`example <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_
running on the CIFAR10 dataset) written in ``PyTorch`` to the appropriate ``edflow`` code.
In particular, a detailed step-by-step explanation of the following parts is provided:

- How to set up the (*required*) ``dataset`` class for ``edflow``
- How to include the classification network (which can then be replaced by any other network in new projects) in the (*required*) ``Model`` class.
- Setting up an ``Ìterator`` (often called *Trainer*) to execute training via the ``step_ops`` method.

As a plus, a brief introduction to data logging via pre-build and custom ``Hooks`` is given.

The config file
~~~~~~~~~~~~~~~~~~~
As mentioned before, each ``edflow`` training is fully set up by its **config** file (e.g. ``train.yaml``).
This file specifies all (tunable) hyper-parameters and paths to the ``Dataset``, ``Model`` and ``Iterator`` used in the project.

Here, the *config.yaml* file is rather short:

.. code-block::

 dataset: tutorial_pytorch.edflow.Dataset
 model: tutorial_pytorch.edflow.Model
 iterator: tutorial_pytorch.edflow.Iterator
 batch_size: 4
 num_epochs: 2

 n_classes: 10

Note that the first five keys are **required** by ``edflow``. The key ``n_classes`` is set to illustrate
the usage of custom keys (e.g. if training only on a subset of all CIFAR10 classes, ...)

Setting up the data
~~~~~~~~~~~~~~~~~~~~

**Necessary Imports**

.. code-block:: python

 import torch
 import torchvision
 import torchvision.transforms as transforms
 import torch.nn as nn
 import torch.nn.functional as F
 import torch.optim as optim

 from edflow.data.dataset import DatasetMixin
 from edflow.iterators.model_iterator import PyHookedModelIterator
 from edflow.hooks.pytorch_hooks import PyCheckpointHook
 from edflow.hooks.hook import Hook
 from edflow.hooks.checkpoint_hooks.torch_checkpoint_hook import RestorePytorchModelHook
 from edflow.project_manager import ProjectManager

Every edflow program requires a ``dataset`` class:

.. code-block:: python

 class Dataset(DatasetMixin):
    """We just initialize the same dataset as in the tutorial and only have to
    implement __len__ and get_example."""

    def __init__(self, config):
        self.train = not config.get("test_mode", False)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=self.train, download=True, transform=transform
        )
        self.dataset = dataset

Our dataset is thus conceptually similar to the ``PyTorch`` ``dataset``. The ``__get_item()__``
method required for pytorch datasets is overwritten by ``get_example()``. We set an additional ``self.train``
flag to unify train- and testdata in this class and make switching between them convenient. It is
noteworthy that a ``Dataloader`` is *not* required in edflow; dataloading methods are inherited from
the base class.

Note that every custom dataset has to implement the methods ``__len()__`` and ``get_example(index)``.
Here, ``get_example(self, index)`` just indexes the ``torchvision.dataset`` and returns the according
numpy arrays (transformed from ``torch.tensor``).

.. code-block:: python

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        """edflow assumes  a dictionary containing values that can be stacked
        by np.stack(), e.g. numpy arrays or integers."""
        x, y = self.dataset[i]
        return {"x": x.numpy(), "y": y}


Building the model
~~~~~~~~~~~~~~~~~~~~
Having specified a dataset we need to define a model to actually run a training.
``edflow`` expects a ``Model`` object which initializes the underlying ``nn.Module`` model.
Here, ``Net`` is the same model that is used in the official PyTorch tutorial; we just recycle it here.

.. code-block:: python

    class Model(object):
        def __init__(self, config):
            """For illustration we read `n_classes` from the config."""
            self.net = Net(n_classes=config["n_classes"])

        def __call__(self, x):
            return self.net(torch.tensor(x))

        def parameters(self):
            return self.net.parameters()

Nothing unusual here (model definition)...

.. code-block:: python

 class Net(nn.Module):
     def __init__(self, n_classes):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(3, 6, 5)
         self.pool = nn.MaxPool2d(2, 2)
         self.conv2 = nn.Conv2d(6, 16, 5)
         self.fc1 = nn.Linear(16 * 5 * 5, 120)
         self.fc2 = nn.Linear(120, 84)
         self.fc3 = nn.Linear(84, n_classes)

     def forward(self, x):
         x = self.pool(F.relu(self.conv1(x)))
         x = self.pool(F.relu(self.conv2(x)))
         x = x.view(-1, 16 * 5 * 5)
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return x

How to actually train (Iterator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Right now we have a rather static model and a dataset but can not do much with it -  that's where the ``Iterator``
comes into play. For ``PyTorch``, this class inherits from ``PyHookedModelIterator`` as follows:


.. code-block:: python

 from edflow.iterators.model_iterator import PyHookedModelIterator

 class Iterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

An ``Iterator`` can for example hold the optimizers used for training, as well as the loss functions.
In our example we use a standard stochastic gradient descent optimizer and cross-entropy loss.
Most important, however, is the (*required*) ``step_ops()`` method: This method provides a pointer towards
the function used to do operations on the data, i.e. as returned by the ``get_example()`` method.
In the example at hand this is the ``train_op()`` method. Note that all ops which should be run as
``step_ops()`` require the ``model`` and the keyword arguments as returned by the ``get_example()`` method
(strictly in this order). We add an if-else statement to directly distinguish between training and testing mode.
This is not necessary; we could also define an ``Evaluator`` (based on ``PyHookedModelIterator``) and point to it
in a ``test.yaml`` file.

.. code-block:: python

 def step_ops(self):
        if self.config.get("test_mode", False):
            return self.test_op
        else:
            return self.train_op

    def train_op(self, model, x, y, **kwargs):
        """All ops to be run as step ops receive model as the first argument
        and keyword arguments as returned by get_example of the dataset."""

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = x, y

Thus, having defined an ``Iterator`` makes the usual

.. code-block:: python

 for epoch in epochs:
    for data in dataloader:
        # do something fancy

loops obsolete (compare to the 'classic' pytorch example).


The following block contains the full Iterator:

.. code-block:: python

 class Iterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.running_loss = 0.0

        self.restorer = RestorePytorchModelHook(
            checkpoint_path=ProjectManager.checkpoints, model=self.model.net
        )
        if not self.config.get("test_mode", False):
            # we add a hook to write checkpoints of the model each epoch or when
            # training is interrupted by ctrl-c
            self.ckpt_hook = PyCheckpointHook(
                root_path=ProjectManager.checkpoints, model=self.model.net
            )  # PyCheckpointHook expects a torch.nn.Module
            self.hooks.append(self.ckpt_hook)
        else:
            # evaluate accuracy
            self.hooks.append(AccuracyHook(self))

    def initialize(self, checkpoint_path=None):
        # restore model from checkpoint
        if checkpoint_path is not None:
            self.restorer(checkpoint_path)

    def step_ops(self):
        if self.config.get("test_mode", False):
            return self.test_op
        else:
            return self.train_op

    def train_op(self, model, x, y, **kwargs):
        """All ops to be run as step ops receive model as the first argument
        and keyword arguments as returned by get_example of the dataset."""

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = x, y

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, torch.tensor(labels))
        loss.backward()
        self.optimizer.step()

        # print statistics
        self.running_loss += loss.item()
        i = self.get_global_step()
        if i % 200 == 199:  # print every 200 mini-batches
            # use the logger instead of print to obtain both console output and
            # logging to the logfile in project directory
            self.logger.info("[%5d] loss: %.3f" % (i + 1, self.running_loss / 200))
            self.running_loss = 0.0

    def test_op(self, model, x, y, **kwargs):
        """Here we just run the model and let the hook handle the output."""
        images, labels = x, y
        outputs = self.model(images)
        return outputs, labels

To run the code, just enter

 $ edflow -t tutorial_pytorch/config.yaml

into your terminal.

Hooks
~~~~~~
Coming soon. Stay tuned :)


Tensorflow
----------

#TODO
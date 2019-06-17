Data Sets and Batching
======================

Basics
------
**EDflow** is pretty much built around your data.
At the core of every training or evaluation is the data, that is utilized.
Through **EDflow** it is easier than ever to reuse data sets, give them additional features or prepare them for evaluation.

To begin with, you have to inherit from a data set call from ``edflow.data.dataset`` e.g. ``DatasetMixin``.
Each class comes with practical features that save code and are (or should) be tested thoroughly.

Every Dataset class **must** include the functions ``get_example(self, index)``, where index is an ``int``, and ``__len__(self)``.
``__len__(self)`` returns the length of your data set i.e. the number of images. Later on, one epoch is defined as iterating through all indices from 0 to ``__len__(self)__-1``.

``get_example(self, index)`` gets the current index as argument.
Normally, these indices are drawn at random but every index is used once in an epoch, which makes for nice, evenly distributed data.
The method must return a ``dict`` with ``string`` s as keys and the data as element.
A nice example would be MNIST.
Typically, ``get_example`` would return a ``dict`` like::

    {label: int, image: np.array}

Naturally, you do not have to use these keys and the ``dict`` can contain as many keys and data of any type as you want.

Batches
-------
If you want to use batches of data you do not have to change anything but the config.
Batches are automatically created based on the key ``batch_size`` which you specify in the config.
One of the advantages of **EDFLow** is, that if your model runs with a batch size of one, it runs with any batch size.

Advanced Data Sets
------------------
If you fancy more complex data sets i.e. triplets for metric learning or sequences of video frames, take a look at these advanced data set classes:

- SequentialDataset
- CachedDataset
- LabelDataset
- ProcessedDataset
- ExtraLabelsDataset
- ConcatenatedDataset
- ExampleConcatenatedDataset
- UnSequenceDataset
- getSeqDataset
- JoinedDataset
- getDebugDataset
- RandomlyJoinedDataset
- DataFolder
- CsvDataset

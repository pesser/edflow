Data Sets and Batching
======================

Basics
------
**edflow** is pretty much built around your data.
At the core of every training or evaluation is the data, that is utilized.
Through **edflow** it is easier than ever to reuse data sets, give them
additional features or prepare them for evaluation.

To begin with, you have to inherit from a data set call from
``edflow.data.dataset`` e.g. ``DatasetMixin``.
Each class comes with practical features that save code and are (or should) be
tested thoroughly.

Every Dataset class **must** include the methods ``get_example(self, idx)``,
where ``idx`` is an ``int``, and ``__len__(self)``.
``__len__(self)`` returns the length of your data set i.e. the number of
images. Later on, one epoch is defined as iterating through all indices from 0
to ``__len__(self)__-1``.

``get_example(self, index)`` gets the current index as argument.
Normally, these indices are drawn at random but every index is used once in an
epoch, which makes for nice, evenly distributed data.
The method must return a ``dict`` with ``string`` s as keys and the data as
element.  A nice example would be MNIST.
Typically, ``get_example`` would return a ``dict`` like::

    {label: int, image: np.array}

Naturally, you do not have to use these keys and the ``dict`` can contain as
many keys and data of any type as you want.

Batches
-------
If you want to use batches of data you do not have to change anything but the
config.
Batches are automatically created based on the key ``batch_size`` which you
specify in the config.
.. One of the advantages of **EDFLow** is, that if your model runs with a batch
   size of one, it runs with any batch size.

Advanced Data Sets
------------------
.. If you fancy more complex data sets i.e. triplets for metric learning or
   sequences of video frames, take a look at these advanced data set classes:
There is a wealth of Dataset manipulation classes, which almost all manipulate
the base dataset by manipulating the indices passed to the dataset.

- SubDataset
- SequenceDataset
- ConcatenatedDataset
- ExampleConcatenatedDataset

More exist, but the above are the ones used most as a recent survey has
shown[#1]_.

.. [#1] Johannes Haux: I use SubDataset, SequenceDataset, ConcatenatedDataset,
   ExampleConcatenatedDataset. The rest I do not use.


:class:`SubDataset`
------------------
Given a dataset and an arbitrary list
of indices, which must be in the range ``[0, len(dataset_]``, it will change
the way the indices are interpreted.



.. - LabelDataset
.. - CachedDataset
.. - ProcessedDataset
.. - ExtraLabelsDataset
.. - UnSequenceDataset
.. - getSeqDataset
.. - JoinedDataset
.. - getDebugDataset
.. - RandomlyJoinedDataset
.. - DataFolder
.. - CsvDataset

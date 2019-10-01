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

A cool feature when working with examples of nested dictionaries is, that they
behave the same as their batch versions! I.e. you can access the same keys in
the same order in a single example and in a batch of examples and still end up
at the value or batch ofl values you would expect.

.. code-block:: python

    example = {'a': 1, 'b': {'c': 1}, 'd': [1, 2]}

    # after applting our batching algorithm on a list of three of the above examples:
    batch_of_3_examples = {'a': [1, 1, 1], 'b': {'c': [1, 1, 1]}, 'd': [[1, 1, 1], [2, 2, 2]]}

    example['a'] == 1  # True
    example['d'][0] == 1  # True

    batch_of_3_examples['a'] == [1, 1, 1]  # True
    batch_of_3_examples['d'][0] == [1, 1, 1]  # True

This comes in especially handy when you use the utility functions found at
``edflow.util`` for handling nested structures, as you now can use the same
keys anytime:

.. code-block:: python

    from edflow.util import retrieve

    retrieve(example, 'a') == 1  # True
    retrieve(example, 'd/0') == 1  # True

    retrieve(batch_of_3_examples, 'a') == [1, 1, 1]  # True
    retrieve(batch_of_3_examples, 'd/0') == [1, 1, 1]  # True

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
shown [#1]_.

.. [#1] Johannes Haux: I use SubDataset, SequenceDataset, ConcatenatedDataset,
   ExampleConcatenatedDataset. The rest I do not use.

Dataset Workflow
----------------

.. warning::

    Datasets, which are specified in the edflow config, must accept one
    positional argument ``config``!

A basic workflow with data in **edflow** looks like this:

1. Load the raw data into some :class:`DatasetMixin` derived custom class.
2. Use this dataset in a different class, which accepts a
   ``config``-dictionary, containing all relevant parameters, e.g. for making
   splits (e.g. train, valid).

This workflow allows to separate the raw loading of the data and reusing it in
various settings. Of course you can merge both steps or add many more.

.. note::

    You can also define a function, which accepts a ``config``, to build you
    Dataset __class__. During construction of the dataset, edflow only expects
    the module defined in the ``config`` behind ``dataset`` to accept the
    config as parameter.
    This behaviour is discouraged though, as one cannot inherit from those
    functions, limiting reusability.

It is also worth noting, that limiting the nestedness of your Dataset pipeline
greatly increases reusability as it helps understanding what is happening to
the raw data.

To further increase the usefulness of your datasets always add documentation
and especially add an example, of what an example from you dataset might look
like. This can be beautifully done using the function
:func:`edflow.util.pp2mkdtable`, which formats the content of the example
as markdown grid-table:

.. code-block:: python

    from edflow.util import pp2mkdtable

    D = MyDataset()
    example = D[10]

    nicely_formatted_string = pp2mkdtable(example)

    # Just copy it from the terminal
    print(nicely_formatted_string)

    # Or write it to a file
    with open('output.md', 'w+') as example_file:
        example_file.write(nicely_formatted_string)

:class:`SubDataset`
-------------------
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

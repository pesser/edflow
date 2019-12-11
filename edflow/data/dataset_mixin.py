from chainer.dataset import DatasetMixin as DatasetMixin_
import numpy as np
from edflow.util import walk, update

# handle bug with mocked chainer.dataset.DatasetMixin import
if hasattr(DatasetMixin_, "_mock_name"):
    DatasetMixin_ = object


class DatasetMixin(DatasetMixin_):
    """Our fork of the `chainer
    <https://docs.chainer.org/en/stable/reference/datasets.html>`_-``Dataset``
    class. Every Dataset used with ``edflow`` should at some point inherit from
    this baseclass.

    Notes
    -----

    **Necessary and best practices**

    When implementing your own dataset you need to specify the following methods:

        - ``__len__`` defines how many examples are in the dataset
        - ``get_example`` returns one of those examples given an index. The example must be a dictionary

    **Labels**

    Additionally the dataset class should specify an attribute :attr:`labels`,
    which works like a dictionary with lists or arrays behind each keyword, that
    have the same length as the dataset. The dictionary can also be empty if
    you do not want to define labels.

    The philosophy behind having both a :meth:`get_example` method and the
    :attr:`labels` attribute is to split the dataset into compute heavy and
    easy parts. Labels should be quick to load at construction time, e.g. by
    loading a ``.npy`` file or a ``.csv``. They can then be used to quickly
    manipulate the dataset. When getting the actual example we can do the heavy
    lifting like loading and/or manipulating images.

    .. warning:: 
        Labels must be ``dict`` s of ``numpy`` arrays and not ``list`` s!
        Otherwise many operations do not work and result in incomprehensible
        errors.

    **Batching**

    As one usually works with batched datasets, the compute heavy steps can be
    hidden through parallelization. This is all done by the
    :func:`make_batches`, which is invoked by ``edflow`` automatically.

    **Default Behaviour**

    As one sometimes stacks and chains multiple levels of datasets it can
    become cumbersome to define ``__len__``, ``get_example`` and ``labels``, if
    all one wants to do is evaluate their respective implementations of some
    other dataset, as can be seen in the code example below:

    .. code-block:: python

        SomeDerivedDataset(DatasetMixin):
            def __init__(self):
                self.other_data = SomeOtherDataset()
                self.labels = self.other_data.labels

            def __len__(self):
                return len(self.other_data)

            def get_example(self, idx):
                return self.other_data[idx]

    This can be omitted when defining a :attr:`data` attribute when
    constructing the dataset. :class:`DatasetMixin` implements these methods
    with the default behaviour to wrap around the corresponding methods of the
    underlying :attr:`data` attribute. Thus the above example becomes

    .. code-block:: python

        SomeDerivedDataset(DatasetMixin):
            def __init__(self):
                self.data = SomeOtherDataset()

    If ``self.data`` has a :attr:`labels` attribute, labels of the derived
    dataset will be taken from ``self.data``.

    **``+`` and ``*``**

    Sometimes you want to concatenate two datasets or multiply the length of
    one dataset by concatenating it several times to itself. This can easily
    be done by adding Datasets or multiplying one by an integer factor.

    .. code-block:: python

        A = C + B  # Adding two Datasets
        D = 3 * A  # Multiplying two datasets

    The above is equivalent to

    .. code-block:: python

        A = ConcatenatedDataset(C, B)  # Adding two Datasets
        D = ConcatenatedDataset(A, A, A)  # Multiplying two datasets

    **Labels in the example ``dict``**

    Oftentimes it is good to store and load some values as lables as it can
    increase performance and decrease storage size, e.g. when storing scalar
    values. If you need these values to be returned by the :func:`get_example`
    method, simply activate this behaviour by setting the attribute
    :attr:`append_labels` to ``True``.

    .. code-block:: python

        SomeDerivedDataset(DatasetMixin):
            def __init__(self):
                self.labels = {'a': [1, 2, 3]}
                self.append_labels = True

            def get_example(self, idx):
                return {'a' : idx**2, 'b': idx}

            def __len__(self):
                return 3

        S = SomeDerivedDataset()
        a = S[2]
        print(a)  # {'a': 3, 'b': 2}

        S.append_labels = False
        a = S[2]
        print(a)  # {'a': 4, 'b': 2}

    Labels are appended to your example, after all code is executed from your
    :attr:`get_example` method. Thus, if there are keys in your labels, which
    can also be found in the examples, the label entries will override the
    values in you example, as can be seen in the example above.
    """

    def _d_msg(self, val):
        """Informs the user that val should be a dict."""

        return (
            "The edflow version of DatasetMixin requires the "
            "`get_example` method to return a `dict`. Yours returned a "
            "{}".format(type(val))
        )

    def __getitem__(self, i):
        ret_dict = super().__getitem__(i)

        if isinstance(i, slice):
            start = i.start or 0
            stop = i.stop
            step = i.step or 1
            for idx, d in zip(range(start, stop, step), ret_dict):
                if not isinstance(d, dict):
                    raise ValueError(self._d_msg(d))
                d["index_"] = idx
                self._maybe_append_labels(d, idx)

        elif isinstance(i, list) or isinstance(i, np.ndarray):
            for idx, d in zip(i, ret_dict):
                if not isinstance(d, dict):
                    raise ValueError(self._d_msg(d))
                d["index_"] = idx
                self._maybe_append_labels(d, idx)

        else:
            if not isinstance(ret_dict, dict):
                raise ValueError(self._d_msg(ret_dict))

            ret_dict["index_"] = i
            self._maybe_append_labels(ret_dict, i)

        self._maybe_expand(ret_dict)

        return ret_dict

    def _maybe_append_labels(self, datum, index):
        if self.append_labels:

            def label_getter(labels):
                return labels[index]

            labels = walk(self.labels, label_getter)
            update(datum, {"labels_": labels})

    def _maybe_expand(self, nested_object):
        if self.expand:
            walk(nested_object, self._expander, inplace=True)

    def _expander(self, val):
        if callable(val):
            val = val()
        return val

    def __len__(self):
        """Add default behaviour for datasets defining an attribute
        :attr:`data`, which in turn is a dataset. This happens often when
        stacking several datasets on top of each other.

        The default behaviour now is to return ``len(self.data)`` if possible,
        and otherwise revert to the original behaviour.
        """
        if hasattr(self, "data"):
            return len(self.data)
        else:
            return super().__len__()

    def get_example(self, *args, **kwargs):
        """
        .. note::

            Please the documentation of :class:`DatasetMixin` to not be
            confused.

        Add default behaviour for datasets defining an attribute
        :attr:`data`, which in turn is a dataset. This happens often when
        stacking several datasets on top of each other.

        The default behaviour now is to return ``self.data.get_example(idx)``
        if possible, and otherwise revert to the original behaviour.
        """
        if hasattr(self, "data"):
            return self.data.get_example(*args, **kwargs)
        else:
            return super().get_example(*args, **kwargs)

    def __mul__(self, val):
        """Returns a ConcatenatedDataset of multiples of itself.

        Parameters
        ----------
        val : int
            How many times do you want this dataset stacked?

        Returns
        -------
        ConcatenatedDataset
            A dataset of ``val``-times the length as ``self``.

        """

        assert isinstance(val, int), "Datasets can only be multiplied by ints"

        if val > 1:
            concs = [self] * val
            return ConcatenatedDataset(*concs)
        else:
            return self

    def __rmul__(self, val):
        return self.__mul__(val)

    def __add__(self, dset):
        """Concatenates self with the other dataset :attr:`dset`.

        Parameters
        ----------
        dset : DatasetMixin
            Another dataset to be concatenated behind ``self``.

        Returns
        -------
        ConcatenatedDataset
            A dataset of form ``[self, dset]``.
        """

        assert isinstance(dset, DatasetMixin), "Can only add DatasetMixins"

        return ConcatenatedDataset(self, dset)

    @property
    def labels(self):
        """Add default behaviour for datasets defining an attribute
        :attr:`data`, which in turn is a dataset. This happens often when
        stacking several datasets on top of each other.

        The default behaviour is to return ``self.data.labels``
        if possible, and otherwise revert to the original behaviour.
        """
        if hasattr(self, "data"):
            return self.data.labels
        elif hasattr(self, "_labels"):
            return self._labels
        else:
            return super().labels

    @labels.setter
    def labels(self, labels):
        if hasattr(self, "data"):
            self.data.labels = labels
        else:
            self._labels = labels

    @property
    def append_labels(self):
        if not hasattr(self, "_append_labels"):
            self._append_labels = False
        return self._append_labels

    @append_labels.setter
    def append_labels(self, value):
        self._append_labels = value

    @property
    def expand(self):
        if not hasattr(self, "_expand"):
            self._expand = False
        return self._expand

    @expand.setter
    def expand(self, value):
        self._expand = value


# We need this here to avoid circular imports
class ConcatenatedDataset(DatasetMixin):
    """A dataset which concatenates given datasets."""

    def __init__(self, *datasets, balanced=False):
        """
        Parameters
        ----------
        *datasets : DatasetMixin
            All datasets we want to concatenate
        balanced : bool
            If ``True`` all datasets are padded to the length of the longest
            dataset. Padding is done in a cycled fashion.
        """
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)
        self.balanced = balanced
        if self.balanced:
            max_length = np.max(self.lengths)
            for data_idx in range(len(self.datasets)):
                data_length = len(self.datasets[data_idx])
                if data_length != max_length:
                    cycle_indices = [i % data_length for i in range(max_length)]
                    self.datasets[data_idx] = SubDataset(
                        self.datasets[data_idx], cycle_indices
                    )
        self.lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum(self.lengths)

    def get_example(self, i):
        """Get example and add dataset index to it."""
        did = np.where(i < self.boundaries)[0][0]
        if did > 0:
            local_i = i - self.boundaries[did - 1]
        else:
            local_i = i
        example = self.datasets[did][local_i]
        example["dataset_index_"] = did
        return example

    def __len__(self):
        return sum(self.lengths)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            new_labels = {}
            label_keys = self.datasets[0].labels.keys()

            for k in label_keys:
                labels = [d.labels[k] for d in self.datasets]
                new_labels[k] = np.concatenate(labels)

            self._labels = new_labels
        return self._labels


# Need this here to avoid circular imports
class SubDataset(DatasetMixin):
    """A subset of a given dataset."""

    def __init__(self, data, subindices):
        self.data = data
        self.subindices = subindices
        try:
            len(self.subindices)
        except TypeError:
            print("Expected a list of subindices.")
            raise

    def get_example(self, i):
        """Get example and process. Wrapped to make sure stacktrace is
        printed in case something goes wrong and we are in a
        MultiprocessIterator."""
        return self.data[self.subindices[i]]

    def __len__(self):
        return len(self.subindices)

    @property
    def labels(self):
        # relay if data is cached
        if not hasattr(self, "_labels"):
            self._labels = dict()
            labels = self.data.labels
            for k in labels:
                self._labels[k] = np.array(labels[k])[self.subindices]
        return self._labels

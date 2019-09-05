from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.agnostics.concatenated import ExampleConcatenatedDataset
import numpy as np


class SequenceDataset(DatasetMixin):
    """Wraps around a dataset and returns sequences of examples.
    Given the length of those sequences the number of available examples
    is reduced by this length times the step taken. Additionally each
    example must have a frame id :attr:`fid_key` specified in the labels, by
    which it can be filtered.
    This is to ensure that each frame is taken from the same video.

    This class assumes that examples come sequentially with :attr:`fid_key` and
    that frame id ``0`` exists.

    The SequenceDataset also exposes the Attribute ``self.base_indices``,
    which holds at each index ``i`` the indices of the elements contained in
    the example from the sequentialized dataset.
    """

    def __init__(self, dataset, length, step=1, fid_key="fid"):
        """
        Parameters
        ----------
        dataset : DatasetMixin
            Dataset from which single frame examples are taken.
        length : int
            Length of the returned sequences in frames.
        step : int
            Step between returned frames. Must be `>= 1`.
        fid_key : str
            Key in labels, at which the frame indices can be found.

        This dataset will have `len(dataset) - length * step` examples.
        """

        self.step = step
        self.length = length

        frame_ids = dataset.labels[fid_key]
        top_indeces = np.where(np.array(frame_ids) >= (length * step - 1))[0]

        all_subdatasets = []
        base_indices = []
        for i in range(length * step):
            indeces = top_indeces - i
            base_indices += [indeces]
            subdset = SubDataset(dataset, indeces)
            all_subdatasets += [subdset]

        all_subdatasets = all_subdatasets[::-1]

        self.data = ExampleConcatenatedDataset(*all_subdatasets)
        self.data.set_example_pars(step=self.step)
        self.base_indices = np.array(base_indices).transpose(1, 0)[:, ::-1]


class UnSequenceDataset(DatasetMixin):
    """Flattened version of a :class:`SequenceDataset`.
    Adds a new key ``seq_idx`` to each example, corresponding to the sequence
    index and a key ``example_idx`` corresponding to the original index.
    The ordering of the dataset is kept and sequence examples are ordererd as
    in the sequence they are taken from.

    .. warning:: This will not create the original non-sequence dataset! The
        new dataset contains ``sequence-length x len(SequenceDataset)``
        examples.

    If the original dataset would be represented as a 2d numpy array the
    ``UnSequence`` version of it would be the concatenation of all its rows:

    .. code-block:: python

        a = np.arange(12)
        seq_dataset = a.reshape([3, 4])
        unseq_dataset = np.concatenate(seq_dataset, axis=-1)

        np.all(a == unseq_dataset))  # True
    """

    def __init__(self, seq_dataset):
        """
        Parameters
        ----------
        seq_dataset : SequenceDataset
            A :class:`SequenceDataset` with attributes :attr:`length`.
        """
        self.data = seq_dataset
        try:
            self.seq_len = self.data.length
        except BaseException:
            # Try to get the seq_length from the labels
            key = list(self.data.labels.keys())[0]
            self.seq_len = len(self.data.labels[key][0])

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            self._labels = self.data.labels
            for k, v in self.labels.items():
                self._labels[k] = np.concatenate(v, axis=-1)
        return self._labels

    def __len__(self):
        """Length is equal to the fixed sequences length times the number of
        sequences."""
        return self.seq_len * len(self.data)

    def get_example(self, i):
        """Examples are gathered with the index
        ``i' = i // seq_len + i % seq_len``
        """
        example_idx = i // self.seq_len
        seq_idx = i % self.seq_len

        example = self.data[example_idx]
        seq_example = {}
        for k, v in example.items():
            # index is added by DatasetMixin
            if k != "index_":
                seq_example[k] = v[seq_idx]
        seq_example.update({"seq_idx": seq_idx, "example_idx": example_idx})

        return seq_example


def getSeqDataset(config):
    """This allows to not define a dataset class, but use a baseclass and a
    `length` and `step` parameter in the supplied `config` to load and
    sequentialize a dataset.

    A config passed to edflow would the look like this:

    .. code-block:: yaml

        dataset: edflow.data.dataset.getSeqDataSet
        model: Some Model
        iterator: Some Iterator

        seqdataset:
                dataset: import.path.to.your.basedataset,
                length: 3,
                step: 1}

    ``getSeqDataSet`` will import the base ``dataset`` and pass it to
    :class:`SequenceDataset` together with ``length`` and ``step`` to
    make the actually used dataset.

    Parameters
    ----------
    config : dict
	An edflow config, with at least the keys
            ``seqdataset`` and nested inside it ``dataset``, ``seq_length`` and
            ``seq_step``.

    Returns
    -------
    :class:`SequenceDataset`
        A Sequence Dataset based on the basedataset.
    """

    ks = "seqdataset"
    base_dset = get_implementations_from_config(config[ks], ["dataset"])["dataset"]
    base_dset = base_dset(config=config)

    return SequenceDataset(base_dset, config[ks]["seq_length"], config[ks]["seq_step"])
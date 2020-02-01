from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.agnostics.concatenated import ExampleConcatenatedDataset
from edflow.util import get_obj_from_str
import numpy as np


def get_sequence_view(frame_ids, length, step=1, strategy="raise", base_step=1):
    """Generates a view on some base dataset given its sequence indices
    :attr:`seq_indices`.

    Parameters
    ----------
    seq_indices : np.ndarray
        An array of *sorted* frame indices. Must be of type `int`.
    length : int
        Length of the returned sequences in frames.
    step : int
        Step between returned frames. Must be `>= 1`.
    strategy : str
        How to handle bad sequences, i.e. sequences starting with a 
        :attr:`fid_key` > 0.
        - ``raise``: Raise a ``ValueError``
        - ``remove``: remove the sequence
        - ``reset``: remove the sequence
    base_step : int
        Step between base frames of returned sequences. Must be `>=1`.

    This view will have `len(dataset) - length * step` entries and shape
    `[len(dataset) - length * step, lenght]`.
    """

    if frame_ids.ndim != 1:
        raise ValueError(
            "Frame ids must be supplied as a sequence of "
            "scalars with the same length as the dataset! Here we "
            "have np.shape(dataset.labels[{}]) = {}`.".format(
                fid_key, np.shape(frame_ids)
            )
        )
    if frame_ids.dtype != np.int:
        raise TypeError(
            f"Frame ids must be supplied as ints, but are {frame_ids.dtype}"
        )

    # Gradient
    diffs = frame_ids[1:] - frame_ids[:-1]
    # All indices where the fid is not monotonically growing
    idxs = np.array([0] + list(np.where(diffs != 1)[0] + 1))
    # Values at these indices
    start_fids = frame_ids[idxs]

    # Bad starts
    badboys = start_fids != 0
    good_seq_idxs = None
    if np.any(badboys):
        n = sum(badboys)
        i_s = "" if n == 1 else "s"
        areis = "is" if n == 1 else "are"
        id_s = "ex" if n == 1 else "ices"
        if strategy == "raise":
            raise ValueError(
                "Frame id sequences must always start with 0. "
                "There {} {} sequence{} starting with the follwing id{}: "
                "{} at ind{} {} in the frame_ids.".format(
                    areis, n, i_s, i_s, start_fids[badboys], id_s, idxs[badboys]
                )
            )

        elif strategy == "remove":
            idxs_stop = np.array(list(idxs[1:]) + [None])
            starts = idxs[badboys]
            stops = idxs_stop[badboys]

            bad_seq_mask = np.zeros(len(frame_ids), dtype=bool)
            for bad_start_idx, bad_stop_idx in zip(starts, stops):
                bad_seq_mask[bad_start_idx:bad_stop_idx] = True

            frame_ids[bad_seq_mask] = 0

        elif strategy == "reset":
            frame_ids = np.copy(frame_ids)  # Don't try to override

            idxs_stop = np.array(list(idxs[1:]) + [None])
            starts = idxs[badboys]
            stops = idxs_stop[badboys]
            vals = start_fids[badboys]

            for val, bad_sa_idx, bad_so_idx in zip(vals, starts, stops):
                frame_ids[bad_sa_idx:bad_so_idx] = (
                    frame_ids[bad_sa_idx:bad_so_idx] - val
                )
        else:
            raise ValueError(
                "Strategy of SequenceDataset must be one of "
                "`raise`, `remove` or `reset` but is "
                "{}".format(strategy)
            )

    top_indices = np.where(
        np.logical_and(
            frame_ids >= (length * step - 1),
            (frame_ids - (length * step - 1)) % base_step == 0,
        )
    )[0]

    base_indices = []
    for i in range(length * step):
        indices = top_indices - i
        base_indices += [indices]

    base_indices = np.array(base_indices).transpose(1, 0)[:, ::-1]
    base_indices = base_indices[:, ::step]

    return base_indices


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

    def __init__(
        self, dataset, length, step=1, fid_key="fid", strategy="raise", base_step=1
    ):
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
        strategy : str
            How to handle bad sequences, i.e. sequences starting with a 
            :attr:`fid_key` > 0.
            - ``raise``: Raise a ``ValueError``
            - ``remove``: remove the sequence
            - ``reset``: remove the sequence
        base_step : int
            Step between base frames of returned sequences. Must be `>=1`.

        This dataset will have `len(dataset) - length * step` examples.
        """

        self.step = step
        self.length = length

        frame_ids = np.array(dataset.labels[fid_key])
        if frame_ids.ndim != 1 or len(frame_ids) != len(dataset):
            raise ValueError(
                "Frame ids must be supplied as a sequence of "
                "scalars with the same length as the dataset! Here we "
                "have np.shape(dataset.labels[{}]) = {}`.".format(
                    fid_key, np.shape(frame_ids)
                )
            )
        if frame_ids.dtype != np.int:
            raise TypeError(
                "Frame ids must be supplied as ints, but are {}".format(frame_ids.dtype)
            )

        # Gradient
        diffs = frame_ids[1:] - frame_ids[:-1]
        # All indices where the fid is not monotonically growing
        idxs = np.array([0] + list(np.where(diffs != 1)[0] + 1))
        # Values at these indices
        start_fids = frame_ids[idxs]

        # Bad starts
        badboys = start_fids != 0
        if np.any(badboys):
            n = sum(badboys)
            i_s = "" if n == 1 else "s"
            areis = "is" if n == 1 else "are"
            id_s = "ex" if n == 1 else "ices"
            if strategy == "raise":
                raise ValueError(
                    "Frame id sequences must always start with 0. "
                    "There {} {} sequence{} starting with the follwing id{}: "
                    "{} at ind{} {} in the dataset.".format(
                        areis, n, i_s, i_s, start_fids[badboys], id_s, idxs[badboys]
                    )
                )

            elif strategy == "remove":
                idxs_stop = np.array(list(idxs[1:]) + [None])
                starts = idxs[badboys]
                stops = idxs_stop[badboys]

                bad_seq_mask = np.ones(len(dataset), dtype=bool)
                for bad_start_idx, bad_stop_idx in zip(starts, stops):
                    bad_seq_mask[bad_start_idx:bad_stop_idx] = False

                good_seq_idxs = np.arange(len(dataset))[bad_seq_mask]
                dataset = SubDataset(dataset, good_seq_idxs)

                frame_ids = dataset.labels[fid_key]

            elif strategy == "reset":
                frame_ids = np.copy(frame_ids)  # Don't try to override

                idxs_stop = np.array(list(idxs[1:]) + [None])
                starts = idxs[badboys]
                stops = idxs_stop[badboys]
                vals = start_fids[badboys]

                for val, bad_sa_idx, bad_so_idx in zip(vals, starts, stops):
                    frame_ids[bad_sa_idx:bad_so_idx] = (
                        frame_ids[bad_sa_idx:bad_so_idx] - val
                    )

                dataset.labels[fid_key] = frame_ids

                frame_ids = dataset.labels[fid_key]
            else:
                raise ValueError(
                    "Strategy of SequenceDataset must be one of "
                    "`raise`, `remove` or `reset` but is "
                    "{}".format(strategy)
                )

        top_indices = np.where(
            np.logical_and(
                frame_ids >= (length * step - 1),
                (frame_ids - (length * step - 1)) % base_step == 0,
            )
        )[0]

        all_subdatasets = []
        base_indices = []
        for i in range(length * step):
            indices = top_indices - i
            base_indices += [indices]
            subdset = SubDataset(dataset, indices)
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
        self.base_data = seq_dataset
        try:
            self.base_seq_len = self.data.length
        except BaseException:
            # Try to get the seq_length from the labels
            key = list(self.base_data.labels.keys())[0]
            self.seq_len = len(self.base_data.labels[key][0])

        self.labels = {}
        for k, v in self.base_data.labels.items():
            self.labels[k] = np.concatenate(v, axis=-1)

    def __len__(self):
        """Length is equal to the fixed sequences length times the number of
        sequences."""
        return self.seq_len * len(self.base_data)

    def get_example(self, i):
        """Examples are gathered with the index
        ``i' = i // seq_len + i % seq_len``
        """
        example_idx = i // self.seq_len
        seq_idx = i % self.seq_len

        example = self.base_data[example_idx]
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
                dataset: import.path.to.your.basedataset
                length: 3
                step: 1
                fid_key: fid
                base_step: 1

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
    base_dset = get_obj_from_str(config[ks]["dataset"])
    base_dset = base_dset(config=config)

    S = SequenceDataset(
        base_dset,
        config[ks]["length"],
        config[ks]["step"],
        fid_key=config[ks]["fid_key"],
        base_step=config[ks].get("base_step", 1),
    )

    return S

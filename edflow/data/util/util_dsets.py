import numpy as np
from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import PRNGMixin
from edflow.util import retrieve
from edflow.main import get_obj_from_str


def JoinedDataset(dataset, key, n_joins):
    """Concat n_joins random samples based on the condition that
    example_i[key] == example_j[key] for all i,j. Key must be in labels of
    dataset."""
    labels = np.asarray(dataset.labels[key])
    unique_labels = np.unique(labels)
    index_map = dict()
    for value in unique_labels:
        index_map[value] = np.nonzero(labels == value)[0]
    join_indices = [list(range(len(dataset)))]  # example_0 is original example
    prng = np.random.RandomState(1)
    for k in range(n_joins - 1):
        indices = [prng.choice(index_map[value]) for value in labels]
        join_indices.append(indices)
    datasets = [SubDataset(dataset, indices) for indices in join_indices]
    dataset = ExampleConcatenatedDataset(*datasets)

    return dataset


def getDebugDataset(config):
    """Loads a dataset from the config and makes ist reasonably small.
    The config syntax works as in :func:`getSeqDataset`. See there for
    more extensive documentation.

    Parameters
    ----------
    config : dict
	An edflow config, with at least the keys
            ``debugdataset`` and nested inside it ``dataset``,
            ``debug_length``, defining the basedataset and its size.

    Returns
    -------
    :class:`SubDataset`:
        A dataset based on the basedataset of the specifed length.
    """

    ks = "debugdataset"
    base_dset = get_implementations_from_config(config[ks], ["dataset"])["dataset"]
    base_dset = base_dset(config=config)
    indices = np.arange(config[ks]["debug_length"])

    return SubDataset(base_dset, indices)


class RandomlyJoinedDataset(DatasetMixin, PRNGMixin):
    """
    Load multiple examples which have the same label.

    Required config parameters:
        :RandomlyJoinedDataset/dataset:    The dataset from which to load
                                            examples.
        :RandomlyJoinedDataset/key:        The key of the label to join on.

    Optional config parameters:
        :test_mode=False:                            If True, behaves deterministic.
        :RandomlyJoinedDataset/n_joins=2:            How many examples to load.
        :RandomlyJoinedDataset/balance=False:        If True and not in test_mode,
                                                      sample join labels uniformly.
        :RandomlyJoinedDataset/avoid_identity=True:  If True and not in test_mode,
                                                      never return a pair containing the same
                                                      image if possible.

    The i-th example returns:
        :'examples': A list of examples, where each example has the same label
         as specified by key. If data_balancing is `False`, the first element of
         the list will be the `i-th` example of the dataset.

    The dataset's labels are the same as that of dataset. Be careful,
    `examples[j]` of the i-th example does not correspond to the i-th entry of
    the labels but to the `examples[j]["index_"]`-th entry.
    """

    def __init__(self, config):
        self.dataset = retrieve(config, "RandomlyJoinedDataset/dataset")
        self.dataset = get_obj_from_str(self.dataset)
        self.dataset = self.dataset(config)
        self.key = retrieve(config, "RandomlyJoinedDataset/key")
        self.n_joins = retrieve(config, "RandomlyJoinedDataset/n_joins", default=2)

        self.test_mode = retrieve(config, "test_mode", default=False)
        self.avoid_identity = retrieve(
            config, "RandomlyJoinedDataset/avoid_identity", default=True
        )
        self.balance = retrieve(config, "RandomlyJoinedDataset/balance", default=False)

        # self.index_map is used to select a partner for each example.
        # In test_mode it is a list containing a single partner index for each
        # example, otherwise it is a dict containing all indices for a given
        # join label
        self.join_labels = np.asarray(self.dataset.labels[self.key])
        unique_labels = np.unique(self.join_labels)
        self.index_map = dict()
        for value in unique_labels:
            self.index_map[value] = np.nonzero(self.join_labels == value)[0]
        if self.test_mode:
            prng = np.random.RandomState(0)
            self.index_map = [
                prng.choice(self.index_map[self.join_labels[i]], self.n_joins - 1)
                for i in range(len(self.dataset))
            ]

    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        """Careful this can only give labels of the original item, not the
        joined ones. Use 'examples[j]["index\_"]' to get the correct label
        index."""
        return self.dataset.labels

    def get_example(self, i):
        if self.test_mode:
            join_indices = self.index_map[i]
        else:
            if self.balance:
                label_id = self.prng.choice(list(self.index_map.keys()))
                i = self.prng.choice(self.index_map[label_id])
            join_value = self.join_labels[i]
            choices = self.index_map[join_value]
            replace = True
            if self.avoid_identity:
                if len(choices) > 1:
                    choices = [idx for idx in choices if not idx == i]
                if len(choices) >= self.n_joins - 1:
                    replace = False
            join_indices = self.prng.choice(choices, self.n_joins - 1, replace=replace)
        join_indices = np.concatenate([[i], join_indices])

        return {"examples": self.dataset[join_indices]}


class DataFolder(DatasetMixin):
    """Given the root of a possibly nested folder containing datafiles and a
    Callable that generates the labels to the datafile from its full name, this
    class creates a labeled dataset.

    A filtering of unwanted Data can be achieved by having the ``label_fn``
    return ``None`` for those specific files. The actual files are only
    read when ``__getitem__`` is called.

    If for example ``label_fn`` returns a dict with the keys ``['a', 'b',
    'c']`` and ``read_fn`` returns one with keys ``['d', 'e']`` then the dict
    returned by ``__getitem__`` will contain the keys ``['a', 'b', 'c', 'd',
    'e', 'file_path_', 'index_']``.
    """

    def __init__(
        self,
        image_root,
        read_fn,
        label_fn,
        sort_keys=None,
        in_memory_keys=None,
        legacy=True,
        show_bar=False,
    ):
        """
        Parameters
        ----------
        image_root : str
            Root containing the files of interest.
        read_fn : Callable
            Given the path to a file, returns the datum as a dict.
        label_fn : Callable
            Given the path to a file, returns a dict of
            labels. If ``label_fn`` returns ``None``, this file is ignored.
        sort_keys : list
            A hierarchy of keys by which the data in this Dataset are sorted.
        in_memory_keys : list
            keys which will be collected from examples when the dataset is cached.
        legacy : bool
            Use the old read ethod, where only the path to the
            current file is passed to the reader. The new version will
            see all labels, that have been previously collected.
        show_bar : bool
            Show a loading bar when loading labels.
        """

        self.root = image_root
        self.read = read_fn
        self.label_fn = label_fn
        self.sort_keys = sort_keys

        self.legacy = legacy
        self.show_bar = show_bar

        if in_memory_keys is not None:
            assert isinstance(in_memory_keys, list)
            self.in_memory_keys = in_memory_keys

        self._read_labels()

    def _read_labels(self):
        import operator

        if self.show_bar:
            n_files = 0
            for _ in os.walk(self.root):
                n_files += 1

            iterator = tqdm(os.walk(self.root), total=n_files, desc="Labels")
        else:
            iterator = tqdm(os.walk(self.root))

        self.data = []
        self.labels = {}
        for root, dirs, files in iterator:
            for f in files:
                path = os.path.join(root, f)
                labels = self.label_fn(path)

                if labels is not None:
                    datum = {"file_path_": path}
                    datum.update(labels)
                    self.data += [datum]

        if self.sort_keys is not None:
            self.data.sort(key=operator.itemgetter(*self.sort_keys))

        for datum in self.data:
            for k, v in datum.items():
                if k not in self.labels:
                    self.labels[k] = []
                self.labels[k] += [v]

    def get_example(self, i):
        """Load the files specified in example ``i``."""
        datum = self.data[i]
        path = datum["file_path_"]

        if self.legacy:
            file_content = self.read(path)
        else:
            file_content = self.read(**datum)

        example = dict()
        example.update(datum)
        example.update(file_content)

        return example

from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import PRNGMixin


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
    """Joins similiar JoinedDataset but randomly selects from possible joins.
    """

    def __init__(self, dataset, key, n_joins):
        """
    Parameters
    ----------
    dataset : DatasetMixin
	Dataset to join in.
    key : str
	Key to join on. Must be in dataset labels.
    n_joins : int
	Number of examples to join.
    """
        self.dataset = dataset
        self.key = key
        self.n_joins = n_joins

        labels = np.asarray(dataset.labels[key])
        unique_labels = np.unique(labels)
        self.index_map = dict()
        for value in unique_labels:
            self.index_map[value] = np.nonzero(labels == value)[0]

    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        """Careful this can only give labels of the original item, not the
        joined ones."""
        return self.dataset.labels

    def get_example(self, i):
        example = self.dataset[i]
        join_value = example[self.key]

        choices = [idx for idx in self.index_map[join_value] if not idx == i]
        join_indices = self.prng.choice(choices, self.n_joins - 1, replace=False)

        examples = [example] + [self.dataset[idx] for idx in join_indices]

        new_examples = {}
        for ex in examples:
            for key, value in ex.items():
                if key in new_examples:
                    new_examples[key] += [value]
                else:
                    new_examples[key] = [value]

        return new_examples


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

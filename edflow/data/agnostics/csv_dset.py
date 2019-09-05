from edflow.data.dataset_mixin import DatasetMixin
import warnings
import pandas as pd
import numpy as np


class CsvDataset(DatasetMixin):
    """Using a csv file as index, this Dataset returns only the entries in the
    csv file, but can be easily extended to load other data using the
    :class:`ProcessedDatasets`.
    """

    def __init__(self, csv_root, **pandas_kwargs):
        """
        Parameters
        ----------
        csv_root : str
            Path/to/the/csv containing all datapoints. The
            first line in the file should contain the names for the
            attributes in the corresponding columns.
        pandas_kwargs : kwargs
            Passed to :func:`pandas.read_csv` when loading the csv file.
        """

        self.root = csv_root
        self.data = pd.read_csv(csv_root, **pandas_kwargs)

        # Stacking allows to also contain higher dimensional data in the csv
        # file like bounding boxes or keypoints.
        # Just make sure to load the data correctly, e.g. by passing the
        # converter ast.literal_val for the corresponding column.
        with warnings.catch_warnings():
            # Pandas will complain, that we are trying to add a column when
            # doing `self.data.labels = labels`. We can ignore this message.
            warnings.simplefilter("ignore", category=UserWarning)

            self.labels = {k: np.stack(self.data[k].values) for k in self.data}

    def get_example(self, idx):
        """Returns all entries in row :attr:`idx` of the labels."""

        # Labels are a pandas dataframe. `.iloc[idx]` returns the row at index
        # idx. Converting to dict results in column_name: row_entry pairs.
        return dict(self.data.iloc[idx])

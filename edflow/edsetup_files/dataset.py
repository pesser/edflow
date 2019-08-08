from edflow.data.dataset import DatasetMixin, PRNGMixin
import numpy as np


class Dataset(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        """
        A pure dataset initialisation with random inputs and labels.

        Args:
            config (dict): The config for the training.
        """
        self.config = config
        self.num_example = config.get("num_example")
        self.feature_dimension = config.get("feature_dimension")
        self.example_names = config.get("example_names")

        self.inputs = np.random.rand(self.num_example, self.feature_dimension)
        self.labels = self.inputs

    def get_example(self, idx):
        """
        Return a dictionary you're going to work with in the iterator.

        Parameters
        ----------
        idx (int): The index of the sample of the dataset that shall be returned.

        Returns
        -------
        example (dict): These will be retrieved by their respective keys in the step_op method of the iterator.
        """
        inputs = self.inputs[idx]
        labels = self.labels[idx]

        example = {"inputs": inputs, "labels": labels}
        return example

    def __len__(self):
        """
        Returns the length of the dataset.
        Returns
        -------
        An integer equal to the length of the dataset.
        """
        return self.num_example

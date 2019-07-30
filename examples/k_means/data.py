import tensorflow as tf
from edflow.iterators.batches import DatasetMixin, make_batches
from edflow.iterators.batches import resize_float32 as resize
import numpy as np

from edflow.nn.tf_nn import np_one_hot


class Dataset_MNIST(DatasetMixin):
    def __init__(self, config):
        """
        Parameters
        ----------
        config: dict
            dictionary representing config options
        """
        self.config = config

        # Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        self.data_train, self.data_test = tf.keras.datasets.mnist.load_data(
            path="mnist.npz"
        )
        self.im_shape = config.get(
            "spatial_size", [28, 28]
        )  # if not found, default value
        if isinstance(self.im_shape, int):
            self.im_shape = [self.im_shape] * 2

    def preprocess(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0
        r = resize(image, self.im_shape)
        return np.expand_dims(r, -1)

    def get_example(self, idx):
        """
        similar to getitem, this method returns A DICT of values for the index idx.
        Parameters
        ----------
        idx : int
            index to get

        Returns
        -------
            a dict of data for this idx

        """
        example = dict()

        image = self.data_train[0][idx]
        class_ = self.data_train[1][idx]
        example["image"] = np.reshape(self.preprocess(image), (-1))
        example["target"] = np_one_hot(class_, 10)
        return example

    def __len__(self):
        return len(self.data_train[0])

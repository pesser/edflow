import tensorflow as tf
from edflow.iterators.batches import DatasetMixin, make_batches
from edflow.iterators.batches import resize_float32 as resize
import numpy as np

class Dataset_MNIST(DatasetMixin):
    def __init__(self, config):
        '''
        Parameters
        ----------
        config: str
            path to config file
        '''
        self.config = config

        # Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        self.data_train, self.data_test = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        self.im_shape = config.get('spatial_size', [28, 28]) # if not found, default value
        if isinstance(self.im_shape, int):
            self.im_shape = [self.im_shape] * 2

    def preprocess(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0
        r = resize(image, self.im_shape)
        return np.expand_dims(r, -1)



    def get_example(self, idx):
        example = dict()

        image = self.data_train[0][idx]
        class_ = self.data_train[1][idx]
        example["image"] = self.preprocess(image)
        example["target"] = class_
        return example


    def __len__(self):
        return len(self.data_train[0]) # RIGHT implementation, yields 60000
        # return len(self.data_train) # WRONG implementation, yields 2


if __name__ == '__main__':
    config = {
        "batch_size" : 8,
        "spatial_size" : 28
    }
    d = Dataset_MNIST(config)

    diter = make_batches(d, 16, True, 1, n_prefetch=1)
    for i in range(10):
        n = next(diter)
        for nn in n.values(): print(nn.shape)
    # yields batches of size 2


    diter = make_batches(d, 16, False, 1, n_prefetch=1)
    for i in range(10):
        n = next(diter)
        for nn in n.values(): print(nn.shape)
    # yield batches of size 16
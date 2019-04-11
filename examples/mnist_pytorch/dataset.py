from torchvision.datasets import MNIST

from edflow.data.dataset import DatasetMixin


class Dataset_MNIST(DatasetMixin):
    def __init__(self, config):
        self.config = config
        self.data = MNIST(root="./data", train=True, transform=None, download=True)
        self.im_shape = config.get("spatial_size", [28, 28])
        if isinstance(self.im_shape, int):
            self.im_shape = [self.im_shape] * 2

    def __len__(self):
        return len(self.data)

    def get_example(self, idx):
        example = dict()
        example["image"] = self.data[idx][0]
        example["target"] = self.data[idx][1]
        return example

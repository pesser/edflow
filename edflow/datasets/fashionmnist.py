import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
import urllib
import gzip
import struct

import edflow.datasets.utils as edu


def read_mnist_file(path):
    # https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    with gzip.open(path, "rb") as f:
        zero, data_type, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


class FashionMNIST(edu.DatasetMixin):
    NAME = "FashionMNIST"
    URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    FILES = dict(
        TRAIN_DATA="train-images-idx3-ubyte.gz",
        TRAIN_LABELS="train-labels-idx1-ubyte.gz",
        TEST_DATA="t10k-images-idx3-ubyte.gz",
        TEST_LABELS="t10k-labels-idx1-ubyte.gz",
    )

    def __init__(self, config=None):
        self.config = config or dict()
        self.logger = edu.get_logger(self)
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = edu.get_root(self.NAME)
        self._data_path = Path(self.root).joinpath("data.p")
        if not edu.is_prepared(self.root):
            # prep
            self.logger.info("Preparing dataset {} in {}".format(self.NAME, self.root))
            root = Path(self.root)
            urls = dict(
                (v, urllib.parse.urljoin(self.URL, v)) for k, v in self.FILES.items()
            )
            local_files = edu.download_urls(urls, target_dir=root)
            data = dict()
            for k, v in local_files.items():
                data[k] = read_mnist_file(v)
            with open(self._data_path, "wb") as f:
                pickle.dump(data, f)
            edu.mark_prepared(self.root)

    def _get_split(self):
        split = (
            "test" if self.config.get("test_mode", False) else "train"
        )  # default split
        if self.NAME in self.config:
            split = self.config[self.NAME].get("split", split)
        return split

    def _load(self):
        with open(self._data_path, "rb") as f:
            self._data = pickle.load(f)
        split = self._get_split()
        assert split in ["train", "test"]
        self.logger.info("Using split: {}".format(split))
        if split == "test":
            self._images = self._data[self.FILES["TEST_DATA"]]
            self._data_labels = self._data[self.FILES["TEST_LABELS"]]
        else:
            self._images = self._data[self.FILES["TRAIN_DATA"]]
            self._data_labels = self._data[self.FILES["TRAIN_LABELS"]]
        self.labels = {"class": self._data_labels, "image": self._images}
        self._length = self._data_labels.shape[0]

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def _preprocess_example(self, example):
        example["image"] = example["image"] / 127.5 - 1.0
        example["image"] = example["image"][:, :, None].astype(np.float32)

    def get_example(self, i):
        example = self._load_example(i)
        self._preprocess_example(example)
        return example

    def __len__(self):
        return self._length


class FashionMNISTTrain(FashionMNIST):
    def _get_split(self):
        return "train"


class FashionMNISTTest(FashionMNIST):
    def _get_split(self):
        return "test"


if __name__ == "__main__":
    print("train")
    d = FashionMNIST()
    print(len(d))
    e = d[0]
    x, y = e["image"], e["class"]
    print(x.dtype, x.shape, x.min(), x.max(), y)

    print("test")
    d = FashionMNIST({"FashionMNIST": {"split": "test"}})
    print(len(d))
    e = d[0]
    x, y = e["image"], e["class"]
    print(x.dtype, x.shape, x.min(), x.max(), y)

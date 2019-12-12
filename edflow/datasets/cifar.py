import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
import urllib

import edflow.datasets.utils as edu


class CIFAR10(edu.DatasetMixin):
    NAME = "CIFAR10"
    URL = "https://www.cs.toronto.edu/~kriz/"
    FILES = dict(DATA="cifar-10-python.tar.gz")

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
            edu.unpack(local_files["cifar-10-python.tar.gz"])
            base = os.path.join(self.root, "cifar-10-batches-py")
            labels = list()
            filenames = list()
            datas = list()
            for batch_file in ["data_batch_{}".format(i) for i in range(1, 6)]:
                with open(os.path.join(base, batch_file), "rb") as f:
                    batch_data = pickle.load(f, encoding="bytes")
                labels += batch_data["labels".encode()]
                filenames += [
                    fname.decode() for fname in batch_data["filenames".encode()]
                ]
                datas.append(batch_data["data".encode()])
            with open(os.path.join(base, "test_batch"), "rb") as f:
                test_data = pickle.load(f, encoding="bytes")
            test_labels = test_data["labels".encode()]
            test_filenames = [
                fname.decode() for fname in test_data["filenames".encode()]
            ]
            test_datas = test_data["data".encode()]
            with open(os.path.join(base, "batches.meta"), "rb") as f:
                _meta = pickle.load(f, encoding="bytes")
            meta = {
                "label_names": [
                    name.decode() for name in _meta["label_names".encode()]
                ],
                "num_vis": _meta["num_vis".encode()],
                "num_cases_per_batch": _meta["num_cases_per_batch".encode()],
            }

            # convert to (32,32,3) RGB uint8
            images = np.concatenate(datas, axis=0)
            images = np.reshape(images, [-1, 3, 32, 32])
            images = np.transpose(images, [0, 2, 3, 1])
            test_images = test_datas
            test_images = np.reshape(test_images, [-1, 3, 32, 32])
            test_images = np.transpose(test_images, [0, 2, 3, 1])

            filenames = np.array(filenames)
            test_filenames = np.array(test_filenames)
            labels = np.array(labels)
            test_labels = np.array(test_labels)

            data = {
                "train": dict(images=images, filenames=filenames, labels=labels),
                "test": dict(
                    images=test_images, filenames=test_filenames, labels=test_labels
                ),
                "meta": meta,
            }
            with open(self._data_path, "wb") as f:
                pickle.dump(data, f)
            edu.mark_prepared(self.root)

    def get_split(self):
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
        self.labels = {
            "class": self._data[split]["labels"],
            "image": self._data[split]["images"],
            "filename": self._data[split]["filenames"],
        }
        self._length = self.labels["class"].shape[0]

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def _preprocess_example(self, example):
        example["image"] = example["image"] / 127.5 - 1.0
        example["image"] = example["image"].astype(np.float32)

    def get_example(self, i):
        example = self._load_example(i)
        self._preprocess_example(example)
        return example

    def __len__(self):
        return self._length


class CIFAR10Train(CIFAR10):
    def _get_split(self):
        return "train"


class CIFAR10Test(CIFAR10):
    def _get_split(self):
        return "test"


if __name__ == "__main__":
    from edflow.util import pp2mkdtable

    print("train")
    d = CIFAR10()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))
    x, y = e["image"], e["class"]
    print(x.dtype, x.shape, x.min(), x.max(), y)

    print("test")
    d = CIFAR10({"CIFAR10": {"split": "test"}})
    print(len(d))
    from PIL import Image

    Image.fromarray(((x + 1.0) * 127.5).astype(np.uint8)).save("cifar10_example.png")

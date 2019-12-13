import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
import urllib
from PIL import Image

import edflow.datasets.utils as edu


class CelebA(edu.DatasetMixin):
    NAME = "CelebA"
    URL = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
    FILES = [
        "img_align_celeba.zip",
        "list_eval_partition.txt",
        "identity_CelebA.txt",
        "list_attr_celeba.txt",
    ]

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
            local_files = dict()

            local_files[self.FILES[0]] = edu.prompt_download(
                self.FILES[0], self.URL, root, content_dir="img_align_celeba"
            )
            if not os.path.exists(os.path.join(root, "img_align_celeba")):
                self.logger.info("Extracting {}".format(local_files[self.FILES[0]]))
                edu.unpack(local_files["img_align_celeba.zip"])

            for v in self.FILES[1:]:
                local_files[v] = edu.prompt_download(v, self.URL, root)

            with open(os.path.join(self.root, "list_eval_partition.txt"), "r") as f:
                list_eval_partition = f.read().splitlines()
                fnames = [s[:10] for s in list_eval_partition]
                list_eval_partition = np.array(
                    [int(s[11:]) for s in list_eval_partition]
                )
            with open(os.path.join(self.root, "list_attr_celeba.txt"), "r") as f:
                list_attr_celeba = f.read().splitlines()
                attribute_descriptions = list_attr_celeba[1]
                list_attr_celeba = list_attr_celeba[2:]
                assert len(list_attr_celeba) == len(list_eval_partition)
                assert [s[:10] for s in list_attr_celeba] == fnames
                list_attr_celeba = np.array(
                    [[int(x) for x in s[11:].split()] for s in list_attr_celeba]
                )
            with open(os.path.join(self.root, "identity_CelebA.txt"), "r") as f:
                identity_celeba = f.read().splitlines()
                assert [s[:10] for s in identity_celeba] == fnames
                identity_celeba = np.array([int(s[11:]) for s in identity_celeba])

            data = {
                "fname": np.array(
                    [os.path.join("img_align_celeba/{}".format(s)) for s in fnames]
                ),
                "partition": list_eval_partition,
                "identity": identity_celeba,
                "attributes": list_attr_celeba,
            }
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
        assert split in ["train", "test", "val"]
        self.logger.info("Using split: {}".format(split))
        if split == "train":
            self.split_indices = np.where(self._data["partition"] == 0)[0]
        elif split == "val":
            self.split_indices = np.where(self._data["partition"] == 1)[0]
        elif split == "test":
            self.split_indices = np.where(self._data["partition"] == 2)[0]
        self.labels = {
            "fname": self._data["fname"][self.split_indices],
            "partition": self._data["partition"][self.split_indices],
            "identity": self._data["identity"][self.split_indices],
            "attributes": self._data["attributes"][self.split_indices],
        }
        self._length = self.labels["fname"].shape[0]

    def _load_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        example["image"] = Image.open(os.path.join(self.root, example["fname"]))
        if not example["image"].mode == "RGB":
            example["image"] = example["image"].convert("RGB")
        example["image"] = np.array(example["image"])
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


class CelebATrain(CelebA):
    def _get_split(self):
        return "train"


class CelebAVal(CelebA):
    def _get_split(self):
        return "val"


class CelebATest(CelebA):
    def _get_split(self):
        return "test"


if __name__ == "__main__":
    from edflow.util import pp2mkdtable

    print("train")
    d = CelebA()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))
    x, y = e["image"], e["attributes"]
    print(x.dtype, x.shape, x.min(), x.max(), y)

    print("test")
    dtest = CelebA({"CelebA": {"split": "test"}})
    print(len(dtest))
    from PIL import Image

    Image.fromarray(((x + 1.0) * 127.5).astype(np.uint8)).save("celeba_example.png")

    id_ = e["identity"]
    id_indices = np.where(d.labels["identity"] == id_)[0][1:]
    for i, id_idx in enumerate(id_indices):
        x = d[id_idx]["image"]
        Image.fromarray(((x + 1.0) * 127.5).astype(np.uint8)).save(
            "celeba_example_{}.png".format(i)
        )

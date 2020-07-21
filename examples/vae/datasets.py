from edflow.data.dataset import PRNGMixin, DatasetMixin
from edflow.util import retrieve
import albumentations
import os
import pandas as pd
import cv2
from edflow.data.util import adjust_support
import numpy as np


COLORS = np.array(
    [
        [0, 0, 0],
        [0, 0, 255],
        [50, 205, 50],
        [139, 78, 16],
        [144, 238, 144],
        [211, 211, 211],
        [250, 250, 255],
    ]
)
W = np.power(255, [0, 1, 2])

HASHES = np.sum(W * COLORS, axis=-1)
HASH2COLOR = {h: c for h, c in zip(HASHES, COLORS)}
HASH2IDX = {h: i for i, h in enumerate(HASHES)}


def rgb2index(segmentation_rgb):
    """
    turn a 3 channel RGB color to 1 channel index color
    """
    s_shape = segmentation_rgb.shape
    s_hashes = np.sum(W * segmentation_rgb, axis=-1)
    print(np.unique(segmentation_rgb.reshape((-1, 3)), axis=0))
    func = lambda x: HASH2IDX[int(x)]
    segmentation_idx = np.apply_along_axis(func, 0, s_hashes.reshape((1, -1)))
    segmentation_idx = segmentation_idx.reshape(s_shape[:2])
    return segmentation_idx


class Deepfashion(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.size = retrieve(config, "spatial_size", default=256)
        self.root = os.path.join("data", "deepfashion")
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler])

        df = pd.read_csv(
            os.path.join(self.root, "Anno", "list_bbox_inshop.txt"),
            skiprows=1,
            sep="\s+",
        )
        self.fnames = list(df.image_name)

    def __len__(self):
        return len(self.fnames)

    def imread(self, path):
        img = cv2.imread(path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_example(self, i):
        fname = self.fnames[i]
        fname2 = "/".join(fname.split("/")[1:])

        fname_iuv, _ = os.path.splitext(fname2)
        fname_iuv = fname_iuv + "_IUV.png"

        fname_segmentation, _ = os.path.splitext(fname2)
        fname_segmentation = fname_segmentation + "_segment.png"

        img_path = os.path.join(self.root, "Img", fname)
        segmentation_path = os.path.join(
            self.root, "Anno", "segmentation", "img_highres", fname_segmentation
        )
        iuv_path = os.path.join(self.root, "Anno", "densepose", "img_iuv", fname_iuv)

        img = self.imread(img_path)
        img = adjust_support(img, "-1->1", "0->255")

        segmentation = cv2.imread(segmentation_path, -1)[:, :, :3]
        segmentation = rgb2index(
            segmentation
        )  # TODO: resizing changes aspect ratio, which might not be okay
        segmentation = cv2.resize(
            segmentation, (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )
        iuv = self.imread(iuv_path)

        example = {"img": img, "segmentation": segmentation, "iuv": iuv}

        return example


class Deepfashion_Img(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.size = retrieve(config, "spatial_size", default=256)
        self.root = os.path.join("data", "deepfashion")
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler])

        df = pd.read_csv(
            os.path.join(self.root, "Anno", "list_bbox_inshop.txt"),
            skiprows=1,
            sep="\s+",
        )
        self.fnames = list(df.image_name)

    def __len__(self):
        return len(self.fnames)

    def imread(self, path):
        img = cv2.imread(path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_example(self, i):
        fname = self.fnames[i]

        img_path = os.path.join(self.root, "Img", fname)

        img = self.imread(img_path)
        img = adjust_support(img, "-1->1", "0->255")

        example = {
            "img": img.astype(np.float32),
        }

        return example


class Deepfashion_Img_Val(Deepfashion_Img):
    def __len__(self):
        return 100

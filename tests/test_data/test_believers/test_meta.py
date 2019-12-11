import pytest
import numpy as np
import os

from edflow.data.believers.meta import MetaDataset
from edflow.util import walk, retrieve


def _setup(root, N=100):
    from PIL import Image

    root = os.path.join(root, "META__test_data__META")
    root = os.path.abspath(root)
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)

    paths = np.array([os.path.join(root, "images", f"{i:0>3d}.png") for i in range(N)])

    mmap_path = os.path.join(root, "labels", f"image:image-*-{N}-*-{paths.dtype}.npy")
    mmap = np.memmap(mmap_path, dtype=paths.dtype, mode="w+", shape=(N,))
    mmap[:] = paths

    data = np.arange(N)
    mmap_path = os.path.join(root, "labels", f"attr1-*-{N}-*-{data.dtype}.npy")
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(N,))
    mmap[:] = data

    data = np.zeros(shape=(N, 2))
    mmap_path = os.path.join(root, "labels", f"attr2-*-{N}x2-*-{data.dtype}.npy")
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(N, 2))
    mmap[:] = data

    data = np.ones(shape=(N, 17, 2))
    mmap_path = os.path.join(root, "labels", f"keypoints-*-{N}x17x2-*-{data.dtype}.npy")
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(N, 17, 2))
    mmap[:] = data

    for p in paths:
        image = (255 * np.ones((64, 64, 3))).astype(np.uint8)
        im = Image.fromarray(image)
        im.save(p)

    with open(os.path.join(root, "meta.yaml"), "w+") as mfile:
        mfile.write(
            """
description: |
    # Test Dataset

    This is a dataset which loads images.
    All paths to the images are in the label `image`.

    ## Content
    image: images

loader_kwargs:
    image:
        support: "-1->1"
        """
        )

    return os.path.abspath(root)


def _teardown(test_data_root):
    if test_data_root == ".":
        raise ValueError("Are you sure you want to delete this directory?")

    os.system(f"rm -rf {test_data_root}")


def test_meta_dset():
    N = 100
    try:
        root = _setup(".", N)

        M = MetaDataset(root)
        M.expand = True
        M.show()

        assert len(M) == N

        for k in ["attr1", "attr2", "image_", "keypoints"]:
            assert k in M.labels
            assert len(M.labels[k]) == N

        d = M[0]
        ref = {
            "image": np.ones(shape=(64, 64, 3)),
            "index_": 0,
            "labels_": {
                "image_": os.path.join(root, "images", "000.png"),
                "attr1": 0,
                "attr2": np.zeros((2)),
                "keypoints": np.ones((17, 2)),
            },
        }

        def tester(key, val):
            assert np.all(val == retrieve(ref, key))

        walk(d, tester, pass_key=True)

        assert hasattr(M, "meta")

    finally:
        _teardown(root)

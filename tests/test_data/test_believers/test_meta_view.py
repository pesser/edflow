import pytest
import numpy as np
import os

from edflow.data.believers.meta_view import MetaViewDataset
from edflow.util import walk, retrieve


def _setup(root, N=100, V=25):
    from PIL import Image

    super_root = os.path.join(root, "METAVIEW__test_data__METAVIEW")
    super_root = os.path.abspath(super_root)
    root = os.path.join(super_root, "base")
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

    view_root = os.path.join(super_root, "mview")

    os.makedirs(os.path.join(view_root, "labels", "views"), exist_ok=True)

    # view 1
    data = np.arange(V).astype(int)
    mmap_path = os.path.join(
        view_root, "labels", "views", f"simple-*-{V}-*-{data.dtype}.npy"
    )
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(V,))
    mmap[:] = data

    # view 2
    data = np.zeros(shape=(V, 5, 3)).astype(int)
    mmap_path = os.path.join(
        view_root, "labels", "views", f"complex-*-{V}x5x3-*-{data.dtype}.npy"
    )
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(V, 5, 3))
    mmap[:] = data

    # view 3
    data = np.arange(V).astype(int)
    mmap_path = os.path.join(view_root, "labels", f"simple-*-{V}-*-{data.dtype}.npy")
    mmap = np.memmap(mmap_path, dtype=data.dtype, mode="w+", shape=(V,))
    mmap[:] = data

    with open(os.path.join(view_root, "meta.yaml"), "w+") as mfile:
        mfile.write(
            """
description: |
    # Test Dataset

    This is a view dataset which loads images from a base.

base_dset: edflow.data.believers.meta.MetaDataset
base_kwargs:
    root: {}

views:
    simple1: simple
    simple: views/simple
    complex:
        - views/complex
        - views/simple
        """.format(
                root
            )
        )

    return super_root, root, view_root


def _teardown(test_data_root):
    if test_data_root == ".":
        raise ValueError("Are you sure you want to delete this directory?")

    os.system(f"rm -rf {test_data_root}")


def test_meta_view_dset():
    N = 100
    V = 25
    try:
        super_root, base_root, view_root = _setup(".", N, V)

        M = MetaViewDataset(view_root)
        M.append_labels = False
        M.show()

        assert len(M) == V

        for kk in ["simple1", "simple", "complex"]:
            assert kk in M.labels
            if kk == "complex":
                for i in range(2):
                    for k in ["attr1", "attr2", "image_", "keypoints"]:
                        assert k in M.labels[kk][i]
                        assert len(M.labels[kk][i][k]) == V
            else:
                for k in ["attr1", "attr2", "image_", "keypoints"]:
                    assert k in M.labels[kk]
                    assert len(M.labels[kk][k]) == V

        d = M[0]
        # For ex 0 this is the same for both complex and simple
        single_ref = {
            "image": np.ones(shape=(64, 64, 3)),
            "index_": 0,
        }

        ref_simple = single_ref
        ref_complex = [[single_ref] * 3] * 20

        ref = {
            "simple1": ref_simple,
            "simple": ref_simple,
            "complex": [ref_complex, ref_simple],
            "index_": 0,
        }

        def tester(key, val):
            assert np.all(val == retrieve(ref, key))

        walk(d, tester, pass_key=True)

        assert hasattr(M, "meta")

    finally:
        _teardown(super_root)

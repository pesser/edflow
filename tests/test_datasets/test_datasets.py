import pytest
import os


class TestCelebaDataset:
    def test(self):
        from edflow.datasets.celeba import CelebA

        dset = CelebA()
        example = dset.get_example(0)
        assert all(
            [
                k in example.keys()
                for k in ["fname", "partition", "identity", "attributes", "image"]
            ]
        )
        assert len(dset) == 162770


class TestCelebaWildDataset:
    def test(self):
        from edflow.datasets.celeba import CelebAWild

        dset = CelebAWild()
        example = dset.get_example(0)
        assert all(
            [
                k in example.keys()
                for k in ["fname", "partition", "identity", "attributes", "image"]
            ]
        )
        assert len(dset) == 162770


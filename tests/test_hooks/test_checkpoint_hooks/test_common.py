import pytest
import os
import time


class Test_WaitForCheckpointHook(object):
    def setup_tmpdir(self, tmpdir):
        subdirs_to_create = ["code", "train", "eval", "ablation"]
        for subdir in subdirs_to_create:
            subdir_path = os.path.join(tmpdir, "logs", "trained_model", subdir)
            if subdir != "code":
                os.makedirs(subdir_path, exist_ok=True)
        sub_dir = os.path.join(tmpdir, "logs", "trained_model", "train", "checkpoints")
        os.makedirs(sub_dir, exist_ok=True)
        checkpoint_fnames = ["model.ckpt-0", "model.ckpt-100"]
        for checkpoint_fname in checkpoint_fnames:
            checkpoint_path = os.path.join(sub_dir, checkpoint_fname)
            time.sleep(2)  # make sure they are sorted in time
            self.make_dummy_checkpoint(checkpoint_path)

    def make_dummy_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "w"):
            pass
        with open(".".join([checkpoint_path, "index"]), "w"):
            pass

    def test_look(self, tmpdir):
        from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook

        self.setup_tmpdir(tmpdir)
        found_checkpoints = []

        def add_checkpoint(checkpoint_path):
            found_checkpoints.append(checkpoint_path)

        checkpoint_root = os.path.join(
            tmpdir, "logs", "trained_model", "train", "checkpoints"
        )
        expected_checkpoints = os.listdir(checkpoint_root)
        expected_checkpoints = list(
            filter(lambda x: ".index" in x, expected_checkpoints)
        )
        expected_checkpoints = list(
            map(lambda x: x.strip(".index"), expected_checkpoints)
        )
        expected_checkpoints = set(expected_checkpoints)
        waiter = WaitForCheckpointHook(
            checkpoint_root=checkpoint_root,
            callback=add_checkpoint,
            eval_all=True,
            interval=1,
            add_sec=1,
        )
        for _ in range(len(expected_checkpoints)):
            waiter.look()
        found_checkpoints = list(map(lambda x: os.path.basename(x), found_checkpoints))
        found_checkpoints = set(found_checkpoints)

        assert found_checkpoints.difference(expected_checkpoints) == set([])

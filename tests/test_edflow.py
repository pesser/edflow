import pytest
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.iterators.batches import DatasetMixin
import tensorflow as tf
import numpy as np
import subprocess
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import subprocess
import os, time
import errno, sys


class Model(object):
    def __init__(self, config):
        self.config = config


class Iterator_checkpoint(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        """ iterator for testing that the provided checkpoint is model.ckpt-0 """

    def initialize(self, checkpoint_path=None):
        assert "model.ckpt-0" in checkpoint_path

    def iterate(self, batch_iterator):
        return None


class Iterator_checkpoint_latest(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        """ iterator for testing that the provided checkpoint is model.ckpt-100 """

    def initialize(self, checkpoint_path=None):
        assert "model.ckpt-100" in checkpoint_path, checkpoint_path

    def iterate(self, batch_iterator):
        return None


class Iterator_no_checkpoint(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        """ iterator for testing that the provided checkpoint is None """

    def initialize(self, checkpoint_path=None):
        assert checkpoint_path is None

    def iterate(self, batch_iterator):
        return None


class Iterator4(TFBaseEvaluator):
    def initialize(self, checkpoint_path=None):
        assert "model.ckpt-0" in checkpoint_path
        assert not self.config["eval_all"]
        assert not self.config["eval_forever"]

    def iterate(self, batch_iterator):
        return None


class Dataset(DatasetMixin):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 1

    def get_example(self, i):
        return {"foo": 0}


class LLDataset(DatasetMixin):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 1

    def get_example(self, i):
        def fn():
            return 0

        return {"foo": fn}


def fullname(o):
    """Get string to specify class in edflow config."""
    module = o.__module__
    return module + "." + o.__name__


def run_edflow_cmdline(command, cwd):
    """Just make sure example runs without errors."""
    env = os.environ.copy()
    if not "CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(
        command,
        shell=True,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        timeout=60,
    )


class Test_eval(object):
    def setup_tmpdir(self, tmpdir):
        subdirs = ["code", "train", "eval", "ablation"]
        for sub in subdirs:
            path = os.path.join(tmpdir, "logs", "trained_model", sub)
            if sub != "code":
                os.makedirs(path, exist_ok=True)
        sub_dir = os.path.join(tmpdir, "logs", "trained_model", "train", "checkpoints")
        os.makedirs(sub_dir, exist_ok=True)
        checkpoints = ["model.ckpt-0", "model.ckpt-100"]
        for c in checkpoints:
            checkpoint_path = os.path.join(sub_dir, c)
            time.sleep(2)  # make sure they are sorted in time
            self.make_dummy_checkpoint(checkpoint_path)

    def make_dummy_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "w"):
            pass
        with open(".".join([checkpoint_path, "index"]), "w"):
            pass

    def test_1(self, tmpdir):
        """
        Test evaluation with providing a checkpoint and writing evaluation into new test_inference folder.
        This should load the checkpoint logs/train/checkpoints/model.ckpt-0

        effectively runs
            edflow -b config.yaml -c logs/train/checkpoints/model.ckpt-0 -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/

        Note: This used to copy everything under {tmpdir}/test but this
        conflicts with other python packages called test so it was changed to
        tmptest.
        -------
        """

        self.setup_tmpdir(tmpdir)
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator_checkpoint)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        config["n_processes"] = 1
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        print(config)
        command = [
            "edflow",
            "-c",
            os.path.join(
                "logs", "trained_model", "train", "checkpoints", "model.ckpt-0"
            ),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        log_dirs = os.listdir(os.path.join(tmpdir, "logs"))
        assert any(list(filter(lambda x: "test_inference" in x, log_dirs)))

    def test_2(self, tmpdir):
        """
        Tests evaluation with providing a checkpoint and writing evaluation into the project folder.
        This should load the checkpoint logs/trained_model/train/checkpoints/model.ckpt-0

        effectively runs
            edflow -b config.yaml -c logs/trained_model/train/checkpoints/model.ckpt-0
            -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator_checkpoint)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        command = [
            "edflow",
            "-c",
            os.path.join(
                "logs", "trained_model", "train", "checkpoints", "model.ckpt-0"
            ),
            "-b",
            "config.yaml",
            "-p",
            os.path.join("logs", "trained_model"),
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

    def test_3(self, tmpdir):
        """
        Tests evaluation without providing a checkpoint. This should load the latest checkpoint.

        effectively runs
            edflow -b config.yaml -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        # command = "edflow -b train.yaml -n test"
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator_checkpoint_latest)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        command = [
            "edflow",
            "-p",
            os.path.join("logs", "trained_model"),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

    def test_4(self, tmpdir):
        """Tests evaluation with
        1. providing a checkpoint
        2. and using eval_all=True and eval_forever=True.

        This should disable overwrite eval_all and eval_forever to ``False``, and then load the specified checkpoint

        effectively runs
            edflow -b config.yaml -c logs/trained_model/train/checkpoints/model.ckpt-0
            -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        # command = "edflow -b eval.yaml train.yaml -n test"
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator4)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        config["eval_all"] = True
        config["eval_forever"] = True
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        command = [
            "edflow",
            "-c",
            os.path.join(
                "logs", "trained_model", "train", "checkpoints", "model.ckpt-0"
            ),
            "-b",
            "config.yaml",
            "-p",
            os.path.join("logs", "trained_model"),
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

    def test_5(self, tmpdir):
        """Tests evaluation with
        1. providing a project
        1. using eval_all=True and eval_forever=True.

        This should NOT load any checkpoint.

        effectively runs
            edflow -b config.yaml -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        # command = "edflow -b train.yaml -n test"
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator_no_checkpoint)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        config["eval_all"] = True
        config["eval_forever"] = False
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        command = [
            "edflow",
            "-p",
            os.path.join("logs", "trained_model"),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

    def test_6(self, tmpdir):
        """Tests evaluation with
        1. a late loading dataset, which returns a function

        effectively runs test_5 but with a late loading dataset. If test_5
        passes, the data has been correctly loaded.
        -------
        """
        self.setup_tmpdir(tmpdir)
        # command = "edflow -b train.yaml -n test"
        config = dict()
        config["model"] = "tmptest." + fullname(Model)
        config["iterator"] = "tmptest." + fullname(Iterator_no_checkpoint)
        config["datasets"] = {
            "train": "tmptest." + fullname(Dataset),
            "validation": "tmptest." + fullname(Dataset),
        }
        config["batch_size"] = 16
        config["num_steps"] = 100
        config["eval_all"] = True
        config["eval_forever"] = False
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tmptest"))
        command = [
            "edflow",
            "-p",
            os.path.join("logs", "trained_model"),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = " ".join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

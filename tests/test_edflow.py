import pytest
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.iterators.batches import DatasetMixin
import tensorflow as tf
import numpy as np
import subprocess
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import subprocess
import os
import errno, sys


class Model(object):
    def __init__(self, config):
        self.config = config


class Iterator1(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        """ iterator for testing that the provided checkpoint is model.ckpt-0 """

    def initialize(self, checkpoint_path=None):
        assert "model.ckpt-0" in checkpoint_path

    def iterate(self, batch_iterator):
        return None


class Iterator2(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        """ iterator for testing that the provided checkpoint is None """

    def initialize(self, checkpoint_path=None):
        assert checkpoint_path == None

    def iterate(self, batch_iterator):
        return None


class Dataset(DatasetMixin):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 1

    def get_example(self, i):
        return {"foo": 0}


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
            self.make_dummy_checkpoint(checkpoint_path)

    def make_dummy_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "w"):
            pass

    def test_1(self, tmpdir):
        """
        Test evaluation with providing a checkpoint and writing evaluation into new test_inference folder.
        This should load the checkpoint logs/train/checkpoints/model.ckpt-0

        effectively runs
            edflow -e config.yaml -b config.yaml -c logs/train/checkpoints/model.ckpt-0 -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/
        -------
        """
        self.setup_tmpdir(tmpdir)
        config = dict()
        config["model"] = "tests." + fullname(Model)
        config["iterator"] = "tests." + fullname(Iterator1)
        config["dataset"] = "tests." + fullname(Dataset)
        config["batch_size"] = 16
        config["num_steps"] = 100
        config["n_processes"] = 1
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tests"))
        command = [
            "edflow",
            "-e",
            "config.yaml",
            "-c",
            os.path.join(
                "logs", "trained_model", "train", "checkpoints", "model.ckpt-0"
            ),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = ' '.join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        log_dirs = os.listdir(os.path.join(tmpdir, "logs"))
        assert any(list(filter(lambda x: "test_inference" in x, log_dirs)))

    def test_2(self, tmpdir):
        """
        Tests evaluation with providing a checkpoint and writing evaluation into the project folder.
        This should load the checkpoint logs/trained_model/train/checkpoints/model.ckpt-0

        effectively runs
            edflow -e config.yaml -b config.yaml -c logs/trained_model/train/checkpoints/model.ckpt-0
            -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        config = dict()
        config["model"] = "tests." + fullname(Model)
        config["iterator"] = "tests." + fullname(Iterator1)
        config["dataset"] = "tests." + fullname(Dataset)
        config["batch_size"] = 16
        config["num_steps"] = 100
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tests"))
        command = [
            "edflow",
            "-e",
            "config.yaml",
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
        command = ' '.join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

    def test_3(self, tmpdir):
        """
        Tests evaluation without providing a checkpoint. This should NOT load any checkpoint.

        effectively runs
            edflow -e config.yaml -b config.yaml -p logs/trained_model -n test_inference

        and then checks if an evaluation folder "test_inference" was created in logs/trained_model/eval
        -------
        """
        self.setup_tmpdir(tmpdir)
        # command = "edflow -e eval.yaml -b train.yaml -n test"
        config = dict()
        config["model"] = "tests." + fullname(Model)
        config["iterator"] = "tests." + fullname(Iterator2)
        config["dataset"] = "tests." + fullname(Dataset)
        config["batch_size"] = 16
        config["num_steps"] = 100
        import yaml

        with open(os.path.join(tmpdir, "config.yaml"), "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        import shutil

        shutil.copytree(os.path.split(__file__)[0], os.path.join(tmpdir, "tests"))
        command = [
            "edflow",
            "-e",
            "config.yaml",
            "-p",
            os.path.join("logs", "trained_model"),
            "-b",
            "config.yaml",
            "-n",
            "test_inference",
        ]
        command = ' '.join(command)
        run_edflow_cmdline(command, cwd=tmpdir)

        # check if correct folder was created
        eval_dirs = os.listdir(os.path.join(tmpdir, "logs", "trained_model", "eval"))
        assert any(list(filter(lambda x: "test_inference" in x, eval_dirs)))

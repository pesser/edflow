import subprocess, os
from edflow.main import _train
from edflow.custom_logging import init_project


def fullname(o):
    """Get string to specify class in edflow config."""
    module = o.__module__
    return module + '.' + o.__name__


def run_edflow_cmdline(command):
    """Just make sure example runs without errors."""
    env = os.environ.copy()
    if not "CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(command, shell = True, check = True, env = env)


def run_edflow(name, config):
    """Run edflow directly from config."""
    P = init_project("logs", code_root=None, postfix=name)
    _train(config, P.root)

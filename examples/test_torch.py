import subprocess, os


def test_mnist():
    """Just make sure example runs without errors."""
    env = os.environ.copy()
    if not "CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        "edflow -b template_pytorch/config.yaml -t --max_batcher_per_epoch --num_epochs 1",
        shell=True,
        check=True,
    )

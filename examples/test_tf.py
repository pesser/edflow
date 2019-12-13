import subprocess, os


def test_mnist():
    """Just make sure example runs without errors."""
    env = os.environ.copy()
    if not "CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(
        "edflow -b mnist_tf/train.yaml -n testrun --num_steps 11 -t",
        shell=True,
        check=True,
        env=env,
    )

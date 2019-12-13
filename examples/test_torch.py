import subprocess


def test_mnist():
    """Just make sure example runs without errors."""
    subprocess.run(
        "edflow -b mnist_pytorch/mnist_config.yaml -n testrun --num_steps 11 -t",
        shell=True,
        check=True,
    )

import subprocess


def test_mnist():
    """Just make sure example runs without errors."""
    subprocess.run('edflow -t mnist_pytorch/mnist_config.yaml -n testrun -o "num_steps: 11"',
            shell = True, check = True)

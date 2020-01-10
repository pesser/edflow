import subprocess
import pytest


def get_model_csv(stdout):
    token = "MODEL_OUTPUT_ROOT "
    stdout = stdout.replace("\\n", "\n")
    for line in stdout.split("\n"):
        if token in line:
            _, path = line.split(token)
            path = path.strip()
            print(path)
            return path


def test_eval():
    """Just make sure example runs without errors."""
    output = subprocess.check_output(
        "edflow -b eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11 -t",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    output = subprocess.run(
        "edeval -m {} -c empty:eval_hook.model.empty_callback".format(csv_name),
        shell=True,
        check=True,
    )


def test_eval_cbs_and_kwargs_from_config():
    """Just make sure example runs and raises no errors. Here additional
    callbacks are supplied via the config file."""
    output = subprocess.check_output(
        "edflow -b eval_hook/mnist_config_cb.yaml -n eval_testrun --num_steps 11 -t",
        shell=True,
    )


def test_eval_wrong_cb_format():
    """Just make sure example runs and raises an expected error."""
    output = subprocess.check_output(
        "edflow -b eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11 -t",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    with pytest.raises(subprocess.CalledProcessError):
        output = subprocess.run(
            "edeval -m {} -c eval_hook.model.empty_callback".format(csv_name),
            shell=True,
            check=True,
        )


def test_eval_with_additional_kwargs():
    """Make sure example runs without errors while adding more kwargs.

    1. edflow -b eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11 -t
    2. edeval -m mata_folder -c empty:eval_hook.model.empty_callback --batch_size 16
    3. edeval -m mata_folder -c empty:eval_hook.model.empty_callback --not_in_there TEST
    """

    output = subprocess.check_output(
        "edflow -b eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11 -t",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    output = subprocess.run(
        "edeval -m {} -c empty:eval_hook.model.empty_callback --batch_size 16".format(
            csv_name
        ),
        shell=True,
        check=True,
    )

    output = subprocess.run(
        "edeval -m {} -c empty:eval_hook.model.empty_callback --not_in_there TEST".format(
            csv_name
        ),
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    test_eval()

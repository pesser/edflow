import subprocess
import pytest


def get_model_csv(stdout):
    token = "MODEL_OUPUT_CSV "
    stdout = stdout.replace("\\n", "\n")
    for line in stdout.split("\n"):
        if token in line:
            print(len(token) + len("]: MODEL_OUPUT_CSV "))
            csv_name = line[len(token) + len("]: MODEL_OUPUT_CSV ") :]
            print(csv_name)
            return csv_name


def test_eval():
    """Just make sure example runs without errors."""
    output = subprocess.check_output(
        "edflow -t eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    output = subprocess.run(
        "edeval -c {} -cb empty:eval_hook.model.empty_callback".format(csv_name),
        shell=True,
        check=True,
    )


def test_eval_cbs_and_kwargs_from_config():
    """Just make sure example runs and raises no errors. Here additional
    callbacks are supplied via the config file."""
    output = subprocess.check_output(
        "edflow -t eval_hook/mnist_config_cb.yaml -n eval_testrun --num_steps 11",
        shell=True,
    )


def test_eval_wrong_cb_format():
    """Just make sure example runs and raises an expected error."""
    output = subprocess.check_output(
        "edflow -t eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    with pytest.raises(subprocess.CalledProcessError):
        output = subprocess.run(
            "edeval -c {} -cb eval_hook.model.empty_callback".format(csv_name),
            shell=True,
            check=True,
        )


def test_eval_with_additional_kwargs():
    """Make sure example runs without errors while adding more kwargs.

    1. edflow -t eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11
    2. edeval -c csv_name -cb empty:eval_hook.model.empty_callback --batch_size 16
    3. edeval -c csv_name -cb empty:eval_hook.model.empty_callback --not_in_there TEST
    """

    output = subprocess.check_output(
        "edflow -t eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    output = subprocess.run(
        "edeval -c {} -cb empty:eval_hook.model.empty_callback --batch_size 16".format(
            csv_name
        ),
        shell=True,
        check=True,
    )

    output = subprocess.run(
        "edeval -c {} -cb empty:eval_hook.model.empty_callback --not_in_there TEST".format(
            csv_name
        ),
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    test_eval()

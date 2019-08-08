import subprocess


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
        "edeval -c {} -cb eval_hook.model.empty_callback".format(csv_name),
        shell=True,
        check=True,
    )


def test_eval_with_additional_kwargs():
    """Just make sure example runs without errors while adding more kwargs."""
    output = subprocess.check_output(
        "edflow -t eval_hook/mnist_config.yaml -n eval_testrun --num_steps 11",
        shell=True,
    )

    csv_name = get_model_csv(str(output))

    output = subprocess.run(
        "edeval -c {} -cb eval_hook.model.empty_callback --batch_size 16".format(
            csv_name
        ),
        shell=True,
        check=True,
    )

    output = subprocess.run(
        "edeval -c {} -cb eval_hook.model.empty_callback --not_in_there TEST".format(
            csv_name
        ),
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    test_eval()

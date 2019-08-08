import ast
from edflow.util import set_value, walk, retrieve


def update_config(config, additional_kwargs):
    """additional_kwargs are added in order of the keys' length, e.g. 'a'
    is overriden by 'a/b'."""
    keys = sorted(additional_kwargs.keys())
    for k in keys:
        set_value(config, k, additional_kwargs[k])

    def replace(k):
        if isinstance(k, str) and k[0] == "{" and k[-1] == "}":
            k_ = k[1:-1].strip()
            return retrieve(config, k_, default=k)
        else:
            return k

    walk(config, replace, inplace=True)


def parse_unknown_args(unknown):
    kwargs = {}
    for i in range(len(unknown)):
        key = unknown[i]
        if key[0] == "-" or key[:2] == "--":
            # Make sure that keys are only passed once
            if key in kwargs:
                raise ValueError("Double Argument: {} is passed twice".format(key))

            # kwarg value
            value = unknown[i + 1]

            # Try to transform the string into a default python class and if
            # that is not possible keep it as string.
            try:
                value = ast.literal_eval(value)
            except Exception as e:
                pass

            # Strip '-' or '--' from key
            while key[0] == "-":
                key = key[1:]

            # Store key key pairs
            kwargs[key] = value

    return kwargs


if __name__ == "__main__":
    import argparse

    A = argparse.ArgumentParser()

    A.add_argument("--a", default="a", type=str)

    passed = [
        "--a",
        "c",
        "--b",
        "b",
        "--c",
        "12.5",
        "--d",
        "True",
        "--e",
        "[14, 15]",
        "--f",
        "a.b.c",
        "--g",
        "a/b/c",
        "--i",
        "1",
        "-j",
        "2",
        "-k",
        "3.",
        "-l",
        "abc",
        "-m",
        "{'asd': 3.5}",
    ]

    print(passed)
    args, unknown = A.parse_known_args(passed)

    unknown = parse_unknown_args(unknown)

    print(args)
    for k, v in unknown.items():
        print(k, v, type(v))

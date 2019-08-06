import pytest

from edflow.config import parse_unknown_args, update_config


def test_basic_parsing():
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
        "--abc/def",
        "1.0",
        "--abc/def/ghi",
        "2.0",
        "--abc/jkl",
        "3.0",
        "--xyz/0",
        "4.0",
    ]

    print(passed)
    args, unknown = A.parse_known_args(passed)

    unknown = parse_unknown_args(unknown)

    assert not "a" in unknown
    ref = {
        "b": "b",
        "c": 12.5,
        "d": True,
        "e": [14, 15],
        "f": "a.b.c",
        "g": "a/b/c",
        "i": 1,
        "j": 2,
        "k": 3.0,
        "l": "abc",
        "m": {"asd": 3.5},
        "abc/def": 1.0,
        "abc/def/ghi": 2.0,
        "abc/jkl": 3.0,
        "xyz/0": 4.0,
    }
    assert ref == unknown


def test_update_config():
    config = dict()
    updates = {"a/b": 1.0, "a": 2}
    update_config(config, updates)
    ref = {"a": {"b": 1.0}}
    assert config == ref

    config = {"a": {"x": 0}}
    updates = {"a/y": 1}
    update_config(config, updates)
    ref = {"a": {"x": 0, "y": 1}}
    assert config == ref

    config = {"a": {"x": 0}}
    updates = {"a/x": 1}
    update_config(config, updates)
    ref = {"a": {"x": 1}}
    assert config == ref

    config = {"a": {"x": 0}}
    updates = {"a/0": 1}
    update_config(config, updates)
    ref = {"a": {"x": 0, 0: 1}}
    assert config == ref

    config = {"a": {"x": 0}}
    updates = {"a/x/1": 1}
    update_config(config, updates)
    ref = {"a": {"x": [None, 1]}}
    assert config == ref


def test_config_format():
    config = dict()
    updates = {"a/b": 1.0, "x": "{a/b}"}
    update_config(config, updates)
    print(config)
    ref = {"a": {"b": 1.0}, "x": 1.0}
    assert config == ref

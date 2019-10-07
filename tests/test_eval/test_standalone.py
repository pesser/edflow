from edflow.eval.pipeline import standalone_eval_csv_file
from edflow.eval.pipeline import load_callbacks
from edflow.eval.pipeline import cbargs2cbdict, config2cbdict, apply_callbacks
import os

import pytest


def empty(root, data_in, data_out, config, **kwargs):
    return kwargs


def test_args2dict():
    args = ["a:b.c.d", "b:f.g.a"]
    ref = {"a": "b.c.d", "b": "f.g.a"}

    result = cbargs2cbdict(args)

    assert ref == result


def test_conf2dict():
    config = {
        "eval_pipeline": {
            "callbacks": {"a": "b.c.d", "b": "f.g.a"},
            "callback_kwargs": {"a": {"k1": 1}},
        }
    }
    ref_cb = {"a": "b.c.d", "b": "f.g.a"}
    ref_kwargs = {"a": {"k1": 1}}

    cbs, kwargs = config2cbdict(config)

    assert ref_cb == cbs
    assert ref_kwargs == kwargs

    config = {"eval_pipeline": {"callbacks": {"a": "b.c.d", "b": "f.g.a"}}}
    ref_cb = {"a": "b.c.d", "b": "f.g.a"}
    ref_kwargs = {}

    cbs, kwargs = config2cbdict(config)

    assert ref_cb == cbs
    assert ref_kwargs == kwargs

    config = {}
    ref_cb = {}
    ref_kwargs = {}

    cbs, kwargs = config2cbdict(config)

    assert ref_cb == cbs
    assert ref_kwargs == kwargs


def test_load_callbacks():
    cb_dict = {"a": "test_standalone.empty"}

    cbs = load_callbacks(cb_dict)

    assert cbs == {"a": empty}


def test_apply_callbacks():
    cbs = {"a": empty}

    results = apply_callbacks(cbs, "", None, None, {}, {})

    assert results == {"a": {}}

    cbs = {"a": empty}

    results = apply_callbacks(cbs, "", None, None, {}, {"a": {"k1": 1}})

    assert results == {"a": {"k1": 1}}


def test_standalone_no_cbs():
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "model_output.csv")
    )
    cbs = {}

    result = standalone_eval_csv_file(csv_path, cbs, {}, None)

    assert result == {}


def test_standalone_w_cbs():
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "model_output.csv")
    )
    cbs = {"a": "test_standalone.empty"}

    result = standalone_eval_csv_file(csv_path, cbs, {}, None)

    assert result == {"a": {}}


def test_standalone_w_cbs_and_kwargs():
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "model_output.csv")
    )
    cbs = {"a": "test_standalone.empty"}

    kwargs = {"eval_pipeline": {"callback_kwargs": {"a": {"k1": 1}}}}

    result = standalone_eval_csv_file(csv_path, cbs, kwargs, None)

    assert result == {"a": {"k1": 1}}


def test_standalone_w_cbs_and_kwargs_and_config():
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "model_output.csv")
    )
    extra_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "extra_config.yaml")
    )
    cbs = {"a": "test_standalone.empty"}

    kwargs = {"eval_pipeline": {"callback_kwargs": {"a": {"k1": 1}}}}

    result = standalone_eval_csv_file(csv_path, cbs, kwargs, extra_path)

    assert result == {"a": {"k1": 2, "k2": 1}}

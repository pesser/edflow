import pytest

from edflow.util import set_value, retrieve, walk, set_default


# ================= set_value ====================


def test_set_value_fail():
    with pytest.raises(Exception):
        dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
        set_value("a/g", 3, dol)  # should raise


def test_set_value():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    ref = {"a": [3, 2], "b": {"c": {"d": 1}, "e": 2}}

    set_value("a/0", 3, dol)
    assert dol == ref

    ref = {"a": [3, 2], "b": {"c": {"d": 1}, "e": 3}}

    set_value("b/e", 3, dol)
    assert dol == ref

    set_value("a/1/f", 3, dol)

    ref = {"a": [3, {"f": 3}], "b": {"c": {"d": 1}, "e": 3}}
    assert dol == ref


def test_append_to_list():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    set_value("a/2", 3, dol)
    ref = {"a": [1, 2, 3], "b": {"c": {"d": 1}, "e": 2}}
    assert dol == ref

    set_value("a/5", 6, dol)
    ref = {"a": [1, 2, 3, None, None, 6], "b": {"c": {"d": 1}, "e": 2}}
    assert dol == ref


def test_add_key():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    set_value("f", 3, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}, "f": 3}
    assert dol == ref

    set_value("b/1", 3, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2, 1: 3}, "f": 3}
    assert dol == ref


def test_fancy_overwriting():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    set_value("e/f", 3, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": 3}}
    assert ref == dol

    set_value("e/f/1/g", 3, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": [None, {"g": 3}]}}
    assert ref == dol


def test_top_is_dict():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    set_value("h", 4, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4}
    assert ref == dol

    set_value("i/j/k", 4, dol)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4, "i": {"j": {"k": 4}}}
    assert ref == dol

    set_value("j/0/k", 4, dol)
    ref = {
        "a": [1, 2],
        "b": {"c": {"d": 1}},
        "e": 2,
        "h": 4,
        "i": {"j": {"k": 4}},
        "j": [{"k": 4}],
    }
    assert ref == dol


def test_top_is_list():
    dol = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}, 2, 3]

    set_value("0/k", 4, dol)
    ref = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "k": 4}, 2, 3]
    assert ref == dol

    set_value("0", 1, dol)
    ref = [1, 2, 3]
    assert ref == dol


# ==================== retrieve ==================


def test_retrieve():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    val = retrieve("a", dol)
    ref = [1, 2]
    assert val == ref

    val = retrieve("a/0", dol)
    ref = 1
    assert val == ref

    val = retrieve("b/c/d", dol)
    ref = 1
    assert val == ref


def test_retrieve_default():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc"

    val = retrieve("f", dol, default="abc")
    assert val == ref

    val = retrieve("a/4", dol, default="abc")
    assert val == ref

    val = retrieve("b/c/e", dol, default="abc")
    assert val == ref


def test_retrieve_fail():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        val = retrieve("f", dol)

        val = retrieve("a/4", dol)

        val = retrieve("b/c/e", dol)


def test_retrieve_pass_success():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc", False

    val = retrieve("f", dol, default="abc", pass_success=True)
    assert val == ref

    val = retrieve("a/4", dol, default="abc", pass_success=True)
    assert val == ref

    val = retrieve("b/c/e", dol, default="abc", pass_success=True)
    assert val == ref

    ref = [1, 2], True
    val = retrieve("a", dol, default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve("a/0", dol, default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve("b/c/d", dol, default="abc", pass_success=True)
    assert val == ref


def test_retrieve_pass_success_fail():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        retrieve("f", dol, pass_success=True)
        retrieve("a/4", dol, pass_success=True)
        retrieve("b/c/e", dol, pass_success=True)

    ref = [1, 2], True
    val = retrieve("a", dol, pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve("a/0", dol, pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve("b/c/d", dol, pass_success=True)
    assert val == ref


# ====================== walk ====================


def test_walk():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": [-1, -2], "b": {"c": {"d": -1}}, "e": -2}

    def fn(leaf):
        return -leaf

    val = walk(dol, fn)

    assert val == ref


def test_walk_inplace():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": [-1, -2], "b": {"c": {"d": -1}}, "e": -2}

    def fn(leaf):
        return -leaf

    walk(dol, fn, inplace=True)

    assert dol == ref


def test_walk_pass_key():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": [-1, -2], "b": {"c": {"d": -1}}, "e": -2}

    def fn(key, leaf):
        return -leaf

    val = walk(dol, fn, pass_key=True)

    assert val == ref


def test_walk_pass_key_inplace():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": [-1, -2], "b": {"c": {"d": -1}}, "e": -2}

    def fn(key, leaf):
        return -leaf

    walk(dol, fn, inplace=True, pass_key=True)

    assert dol == ref


# =================== set_default ================


def test_set_default_key_contained():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    val = set_default(dol, "a", "new")

    assert dol == ref
    assert val == [1, 2]


def test_set_default_key_not_contained():
    dol = {"b": {"c": {"d": 1}}, "e": 2}
    ref = {"a": "new", "b": {"c": {"d": 1}}, "e": 2}

    val = set_default(dol, "a", "new")

    assert dol == ref
    assert val == "new"

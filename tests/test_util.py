import pytest
import copy
from edflow.util import (
    set_value,
    retrieve,
    walk,
    set_default,
    contains_key,
    KeyNotFoundError,
)
from edflow import util
from itertools import product

# ================= set_value ====================


def test_pop_value_from_key():
    collection = {"a": [1, 2]}
    key = "a"
    popped_value = util.pop_value_from_key(collection, key)
    expected_value = [1, 2]
    assert expected_value == popped_value


def pytest_generate_tests(metafunc):
    # called once per each test class
    # http://doc.pytest.org/en/latest/example/parametrize.html
    if metafunc.cls is not None:
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = metafunc.cls.argnames
        metafunc.parametrize(argnames, funcarglist)


def make_collection():
    collection = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    return collection


class Test_pop_from_nested_structure:
    argnames = ("collection", "key", "expected_value")
    params = {
        "test_pop_from_nested_structure": [
            (make_collection(), "a", [1, 2]),
            (make_collection(), "b/c/d", 1),
            (make_collection(), "a/0", 1),
        ],
        "test_default": [
            (make_collection(), "f", "abc"),
            (make_collection(), "a/4", "abc"),
            (make_collection(), "b/c/e", "abc"),
        ],
        "test_raise_keyNotFoundError": [
            (make_collection(), "f", None),
            (make_collection(), "a/4", None),
            (make_collection(), "b/c/e", None),
        ],
        "test_pass_success": [
            (make_collection(), "f", ("abc", False)),
            (make_collection(), "a/4", ("abc", False)),
            (make_collection(), "b/c/e", ("abc", False)),
            (make_collection(), "a", ([1, 2], True)),
            (make_collection(), "a/0", (1, True)),
            (make_collection(), "b/c/d", (1, True)),
        ],
        "test_raise_keyNotFoundError_pass_success": [
            (make_collection(), "f", None),
            (make_collection(), "a/4", None),
            (make_collection(), "b/c/e", None),
        ],
        "test_pass_sucess_default": [
            (make_collection(), "a", ([1, 2], True)),
            (make_collection(), "a/0", (1, True)),
            (make_collection(), "b/c/d", (1, True)),
        ],
    }

    def test_pop_from_nested_structure(self, collection, key, expected_value):
        popped_value = util.pop_from_nested_structure(collection, key)
        assert expected_value == popped_value

    def test_default(self, collection, key, expected_value):
        popped_value = util.pop_from_nested_structure(collection, key, default="abc")
        assert expected_value == popped_value

    def test_raise_keyNotFoundError(self, collection, key, expected_value):
        with pytest.raises(KeyNotFoundError) as exc_info:
            util.pop_from_nested_structure(collection, key)

    def test_pass_success(self, collection, key, expected_value):
        popped_value = util.pop_from_nested_structure(
            collection, key, default="abc", pass_success=True
        )
        assert expected_value == popped_value

    def test_raise_keyNotFoundError_pass_success(self, collection, key, expected_value):
        with pytest.raises(KeyNotFoundError) as exc_info:
            util.pop_from_nested_structure(collection, key, pass_success=True)

    def test_pass_sucess_default(self, collection, key, expected_value):
        popped_value = util.pop_from_nested_structure(
            collection, key, default="abc", pass_success=True
        )
        assert expected_value == popped_value


def test_keyNotFoundError():
    with pytest.raises(KeyNotFoundError) as exc_info:
        raise KeyNotFoundError("test")

    with pytest.raises(KeyNotFoundError) as exc_info:
        try:
            a = {"a": "b"}
            a.pop("c")
        except (KeyError, IndexError) as e:
            raise KeyNotFoundError(e)


def test_set_value_fail():
    with pytest.raises(Exception):
        dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
        set_value(dol, "a/g", 3)  # should raise


def test_set_value():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    ref = {"a": [3, 2], "b": {"c": {"d": 1}, "e": 2}}

    set_value(dol, "a/0", 3)
    assert dol == ref

    ref = {"a": [3, 2], "b": {"c": {"d": 1}, "e": 3}}

    set_value(dol, "b/e", 3)
    assert dol == ref

    set_value(dol, "a/1/f", 3)

    ref = {"a": [3, {"f": 3}], "b": {"c": {"d": 1}, "e": 3}}
    assert dol == ref


def test_append_to_list():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    set_value(dol, "a/2", 3)
    ref = {"a": [1, 2, 3], "b": {"c": {"d": 1}, "e": 2}}
    assert dol == ref

    set_value(dol, "a/5", 6)
    ref = {"a": [1, 2, 3, None, None, 6], "b": {"c": {"d": 1}, "e": 2}}
    assert dol == ref


def test_add_key():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
    set_value(dol, "f", 3)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}, "f": 3}
    assert dol == ref

    set_value(dol, "b/1", 3)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2, 1: 3}, "f": 3}
    assert dol == ref


def test_fancy_overwriting():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    set_value(dol, "e/f", 3)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": 3}}
    assert ref == dol

    set_value(dol, "e/f/1/g", 3)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": [None, {"g": 3}]}}
    assert ref == dol


def test_top_is_dict():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    set_value(dol, "h", 4)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4}
    assert ref == dol

    set_value(dol, "i/j/k", 4)
    ref = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4, "i": {"j": {"k": 4}}}
    assert ref == dol

    set_value(dol, "j/0/k", 4)
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

    set_value(dol, "0/k", 4)
    ref = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "k": 4}, 2, 3]
    assert ref == dol

    set_value(dol, "0", 1)
    ref = [1, 2, 3]
    assert ref == dol


# ==================== retrieve ==================


def test_retrieve():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    val = retrieve(dol, "a")
    ref = [1, 2]
    assert val == ref

    val = retrieve(dol, "a/0")
    ref = 1
    assert val == ref

    val = retrieve(dol, "b/c/d")
    ref = 1
    assert val == ref


def test_retrieve_default():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc"

    val = retrieve(dol, "f", default="abc")
    assert val == ref

    val = retrieve(dol, "a/4", default="abc")
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc")
    assert val == ref


def test_retrieve_fail():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        val = retrieve(dol, "f")

    with pytest.raises(Exception):
        val = retrieve(dol, "a/4")

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/e")


def test_retrieve_pass_success():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc", False

    val = retrieve(dol, "f", default="abc", pass_success=True)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", pass_success=True)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", pass_success=True)
    assert val == ref

    ref = [1, 2], True
    val = retrieve(dol, "a", default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", default="abc", pass_success=True)
    assert val == ref


def test_retrieve_pass_success_fail():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        retrieve(dol, "f", pass_success=True)
    with pytest.raises(Exception):
        retrieve(dol, "a/4", pass_success=True)
    with pytest.raises(Exception):
        retrieve(dol, "b/c/e", pass_success=True)

    ref = [1, 2], True
    val = retrieve(dol, "a", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", pass_success=True)
    assert val == ref


# -------------------- retrieve with expand=False ------------------


def test_retrieve_ef():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    val = retrieve(dol, "a", expand=False)
    ref = [1, 2]
    assert val == ref

    val = retrieve(dol, "a/0", expand=False)
    ref = 1
    assert val == ref

    val = retrieve(dol, "b/c/d", expand=False)
    ref = 1
    assert val == ref


def test_retrieve_default_ef():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc"

    val = retrieve(dol, "f", default="abc", expand=False)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", expand=False)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", expand=False)
    assert val == ref


def test_retrieve_fail_ef():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        val = retrieve(dol, "f", expand=False)

    with pytest.raises(Exception):
        val = retrieve(dol, "a/4", expand=False)

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/e", expand=False)


def test_retrieve_pass_success_ef():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    ref = "abc", False

    val = retrieve(dol, "f", default="abc", pass_success=True, expand=False)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", pass_success=True, expand=False)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = [1, 2], True
    val = retrieve(dol, "a", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", default="abc", pass_success=True, expand=False)
    assert val == ref


def test_retrieve_pass_success_fail_ef():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    with pytest.raises(Exception):
        retrieve(dol, "f", pass_success=True, expand=False)
    with pytest.raises(Exception):
        retrieve(dol, "a/4", pass_success=True, expand=False)
    with pytest.raises(Exception):
        retrieve(dol, "b/c/e", pass_success=True, expand=False)

    ref = [1, 2], True
    val = retrieve(dol, "a", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", pass_success=True, expand=False)
    assert val == ref


# -------------------- retrieve with callable leaves ------------------


def nested_leave():
    return {"d": 1}


def callable_leave():
    return {"c": nested_leave}


def test_retrieve_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    val = retrieve(dol, "a")
    ref = [1, 2]
    assert val == ref

    val = retrieve(dol, "a/0")
    ref = 1
    assert val == ref

    val = retrieve(dol, "b/c/d")
    ref = 1
    assert val == ref

    # test in-place modification
    ref_dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    assert dol == ref_dol


def test_retrieve_default_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    ref = "abc"

    val = retrieve(dol, "f", default="abc")
    assert val == ref

    val = retrieve(dol, "a/4", default="abc")
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc")
    assert val == ref


def test_retrieve_fail_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    with pytest.raises(Exception):
        val = retrieve(dol, "f")

    with pytest.raises(Exception):
        val = retrieve(dol, "a/4")

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/e")


def test_retrieve_pass_success_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    ref = "abc", False

    val = retrieve(dol, "f", default="abc", pass_success=True)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", pass_success=True)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", pass_success=True)
    assert val == ref

    ref = [1, 2], True
    val = retrieve(dol, "a", default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", default="abc", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", default="abc", pass_success=True)
    assert val == ref


def test_retrieve_pass_success_fail_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    with pytest.raises(Exception):
        retrieve(dol, "f", pass_success=True)
    with pytest.raises(Exception):
        retrieve(dol, "a/4", pass_success=True)
    with pytest.raises(Exception):
        retrieve(dol, "b/c/e", pass_success=True)

    ref = [1, 2], True
    val = retrieve(dol, "a", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", pass_success=True)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "b/c/d", pass_success=True)
    assert val == ref


# -------------------- retrieve with callable and expand=False ------------------


def test_retrieve_ef_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    val = retrieve(dol, "a", expand=False)
    ref = [1, 2]
    assert val == ref

    val = retrieve(dol, "a/0", expand=False)
    ref = 1
    assert val == ref

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/d", expand=False)


def test_retrieve_default_ef_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    ref = "abc"

    val = retrieve(dol, "f", default="abc", expand=False)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", expand=False)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", expand=False)
    assert val == ref


def test_retrieve_fail_ef_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    with pytest.raises(Exception):
        val = retrieve(dol, "f", expand=False)

    with pytest.raises(Exception):
        val = retrieve(dol, "a/4", expand=False)

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/e", expand=False)


def test_retrieve_pass_success_ef_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    ref = "abc", False

    val = retrieve(dol, "f", default="abc", pass_success=True, expand=False)
    assert val == ref

    val = retrieve(dol, "a/4", default="abc", pass_success=True, expand=False)
    assert val == ref

    val = retrieve(dol, "b/c/e", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = [1, 2], True
    val = retrieve(dol, "a", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", default="abc", pass_success=True, expand=False)
    assert val == ref

    ref = "abc", False
    val = retrieve(dol, "b/c/d", default="abc", pass_success=True, expand=False)
    assert val == ref


def test_retrieve_pass_success_fail_ef_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}

    with pytest.raises(Exception):
        retrieve(dol, "f", pass_success=True, expand=False)
    with pytest.raises(Exception):
        retrieve(dol, "a/4", pass_success=True, expand=False)
    with pytest.raises(Exception):
        retrieve(dol, "b/c/e", pass_success=True, expand=False)

    ref = [1, 2], True
    val = retrieve(dol, "a", pass_success=True, expand=False)
    assert val == ref

    ref = 1, True
    val = retrieve(dol, "a/0", pass_success=True, expand=False)
    assert val == ref

    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/d", pass_success=True, expand=False)


def failing_leave():
    raise Exception()
    return {"c": nested_leave}


class CustomException(Exception):
    pass


def custom_leave():
    raise CustomException()
    return {"c": nested_leave}


def test_retrieve_propagates_exception():
    dol = {"a": [1, 2], "b": failing_leave, "e": 2}
    with pytest.raises(Exception):
        val = retrieve(dol, "b/c/d", default=0)

    dol = {"a": [1, 2], "b": custom_leave, "e": 2}
    with pytest.raises(CustomException):
        val = retrieve(dol, "b/c/d", default=0)


def test_retrieve_callable_leaves():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    val = retrieve(dol, "b")

    # make sure expansion is returned
    assert val == callable_leave()

    # make sure expansion was done in-place
    assert dol["b"] == callable_leave()

    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    val = retrieve(dol, "b/c")
    # make sure expansion is returned
    assert val == nested_leave()
    # make sure expansion was done in-place
    assert dol["b"]["c"] == nested_leave()

    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    val = retrieve(dol, "b/c/d")
    assert val == 1


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


# =================== set_default ================


def test_contains_key():
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
    assert contains_key(dol, "a")
    assert contains_key(dol, "b/c/d")
    assert not contains_key(dol, "b/c/f")
    assert not contains_key(dol, "f")


def test_contains_key_callable():
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}
    assert contains_key(dol, "a", expand=True)
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}  # reset
    assert contains_key(dol, "a", expand=False)
    assert contains_key(dol, "b/c/d", expand=True)
    assert contains_key(dol, "b/c/d", expand=False)  # now its expanded
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}  # reset
    assert not contains_key(dol, "b/c/d", expand=False)
    assert not contains_key(dol, "b/c/f", expand=True)
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}  # reset
    assert not contains_key(dol, "b/c/f", expand=False)
    assert not contains_key(dol, "f", expand=True)
    dol = {"a": [1, 2], "b": callable_leave, "e": 2}  # reset
    assert not contains_key(dol, "f", expand=False)

"""Some Utility functions, that make yur life easier but don't fit in any
better catorgory than util."""

import numpy as np
import os
import pickle
from typing import *
import importlib

try:
    from IPython import get_ipython
    from IPython.display import display, Markdown

    __COULD_HAVE_IPYTHON__ = True
except ImportError:
    __COULD_HAVE_IPYTHON__ = False


def get_str_from_obj(obj):
    return obj.__module__ + "." + obj.__name__


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def linear_var(step, start, end, start_value, end_value, clip_min=0.0, clip_max=1.0):
    r"""
    Linear from :math:`(a, \alpha)` to :math:`(b, \beta)`, i.e.
    :math:`y = (\beta - \alpha)/(b - a) * (x - a) + \alpha`

    Parameters
    ----------
    step : int
        :math:`x`
    start : float
        :math:`a`
    end : float
        :math:`b`
    start_value : float
        :math:`\alpha`
    end_value : float
        :math:`\beta`
    clip_min : float
        Minimal value returned.
    clip_max : float
        Maximum value returned.

    Returns
    -------
    :math:`y` : float

    """
    linear = (end_value - start_value) / (end - start) * (
        float(step) - start
    ) + start_value
    return float(np.clip(linear, clip_min, clip_max))


def walk(
    dict_or_list,
    fn,
    inplace=False,
    pass_key=False,
    prev_key="",
    splitval="/",
    walk_np_arrays=False,
):  # noqa
    """
    Walk a nested list and/or dict recursively and call fn on all non
    list or dict objects.

    Example
    -------

    .. code-block:: python

        dol = {'a': [1, 2], 'b': {'c': 3, 'd': 4}}

        def fn(val):
            return val**2

        result = walk(dol, fn)
        print(result)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}
        print(dol)  # {'a': [1, 2], 'b': {'c': 3, 'd': 4}}

        result = walk(dol, fn, inplace=True)
        print(result)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}
        print(dol)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}

    Parameters
    ----------
    dict_or_list : dict or list
        Possibly nested list or dictionary.
    fn : Callable
        Applied to each leave of the nested list_dict-object.
    inplace : bool
        If False, a new object with the same structure
        and the results of fn at the leaves is created. If True the leaves
        are replaced by the results of fn.
    pass_key : bool
        Also passes the key or index of the leave element to
        ``fn``.
    prev_key : str
        If ``pass_key == True`` keys of parent nodes are passed
        to calls of ``walk`` on child nodes to accumulate the keys.
    splitval : str
        String used to join keys if :attr:`pass_key` is ``True``.
    walk_np_arrays : bool
        If ``True``, numpy arrays are intepreted as list, ie not as leaves.

    Returns
    -------
    The resulting nested list-dict-object with the results of
    fn at its leaves. : dict or list
    """

    instance_test = (list, dict)
    list_test = (list,)

    if walk_np_arrays:
        instance_test += (np.ndarray,)
        list_test += (np.ndarray,)

    if not pass_key:

        def call(value):
            if isinstance(value, instance_test):
                return walk(value, fn, inplace, walk_np_arrays=walk_np_arrays)
            else:
                return fn(value)

    else:

        def call(key, value):
            if prev_key != "":
                key = splitval.join([prev_key, key])

            if isinstance(value, instance_test):
                return walk(
                    value,
                    fn,
                    inplace,
                    pass_key=True,
                    prev_key=key,
                    splitval=splitval,
                    walk_np_arrays=walk_np_arrays,
                )
            else:
                return fn(key, value)

    if isinstance(dict_or_list, list_test):
        results = []
        for i, val in strenumerate(dict_or_list):
            result = call(i, val) if pass_key else call(val)
            results += [result]
            if inplace:
                dict_or_list[int(i)] = result
    elif isinstance(dict_or_list, dict):
        results = {}
        for key, val in dict_or_list.items():
            result = call(key, val) if pass_key else call(val)
            results[key] = result
            if inplace:
                dict_or_list[key] = result
    else:
        if not inplace:
            if not pass_key:
                results = fn(dict_or_list)
            else:
                results = fn(prev_key, dict_or_list)
        else:
            if not pass_key:
                dict_or_list = fn(dict_or_list)
            else:
                dict_or_list = fn(prev_key, dict_or_list)

    if inplace:
        results = dict_or_list

    return results


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


def pop_keypath(
    current_item: Union[callable, list, dict],
    key: str,
    splitval: str = "/",
    default: object = None,
    expand: bool = True,
    pass_success: bool = False,
):
    """Given a nested list or dict structure, pop the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        current_item : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    visited = []
    parent = None
    last_key = None
    try:
        for key in keys:
            if callable(current_item):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                else:
                    current_item = current_item()
                    parent[last_key] = current_item

            last_key = key
            parent = current_item

            try:
                if isinstance(current_item, dict):
                    current_item = current_item[key]
                else:
                    current_item = current_item[int(key)]
            except (KeyError, IndexError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(current_item):
            current_item = current_item()
            parent[last_key] = current_item

        if isinstance(parent, list):
            parent[int(last_key)] = None
        else:
            del parent[last_key]

    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            current_item = default
            success = False
    if pass_success:
        return current_item, success
    else:
        return current_item


def get_value_from_key(collection: Union[list, dict], key: str):
    """Get value from collection given key
    """
    if isinstance(collection, dict):
        value = collection[key]
    else:
        value = collection[int(key)]
    return value


def pop_value_from_key(collection: Union[list, dict], key: str):
    """Pop item from collection given key

    Parameters
    ----------
    collection : Union[list, dict]
    key

    Returns
    -------

    """
    if isinstance(collection, dict):
        value = collection.pop(key)
    else:
        value = collection.pop(int(key))
    return value


def set_default(list_or_dict, key, default, splitval="/"):
    """Combines :func:`retrieve` and :func:`set_value` to create the
    behaviour of pythons ``dict.setdefault``: If ``key`` is found in
    ``list_or_dict``, return its value, otherwise return ``default`` and add it
    to ``list_or_dict`` at ``key``.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.  splitval (str): String that
            defines the delimiter between keys of the different depth levels in
            `key`.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        default : object
            Value to be returned if ``key`` not in list_or_dict and set to be
            at ``key`` in this case.
        splitval : str
            String that defines the delimiter between keys of the different
            depth levels in :attr:`key`.

    Returns
    -------
        The retrieved value or if the :attr:`key` is not found returns
        ``default``.
    """

    try:
        ret_val = retrieve(list_or_dict, key, splitval, None)
    except KeyNotFoundError as e:
        set_value(list_or_dict, key, default, splitval)
        ret_val = default

    return ret_val


def set_value(list_or_dict, key, val, splitval="/"):
    """Sets a value in a possibly nested list or dict object.

    Parameters
    ----------

    key : str
        ``key/to/value``, path like string describing all keys necessary to
        consider to get to the desired value. List indices can also be passed
        here.
    value : object
        Anything you want to put behind :attr:`key`
    list_or_dict : list or dict
        Possibly nested list or dictionary.
    splitval : str
        String that defines the delimiter between keys of the different depth
        levels in :attr:`key`.


    Examples
    --------

    .. code-block:: python

        dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}

        # Change existing entry
        set_value(dol, "a/0", 3)
        # {'a': [3, 2], 'b': {'c': {'d': 1}, 'e': 2}}}

        set_value(dol, "b/e", 3)
        # {"a": [3, 2], "b": {"c": {"d": 1}, "e": 3}}

        set_value(dol, "a/1/f", 3)
        # {"a": [3, {"f": 3}], "b": {"c": {"d": 1}, "e": 3}}

        # Append to list
        dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}

        set_value(dol, "a/2", 3)
        # {"a": [1, 2, 3], "b": {"c": {"d": 1}, "e": 2}}

        set_value(dol, "a/5", 6)
        # {"a": [1, 2, 3, None, None, 6], "b": {"c": {"d": 1}, "e": 2}}

        # Add key
        dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}
        set_value(dol, "f", 3)
        # {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}, "f": 3}

        set_value(dol, "b/1", 3)
        # {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2, 1: 3}, "f": 3}

        # Raises Error:
        # Appending key to list
        # set_value(dol, 'a/g', 3)  # should raise

        # Fancy Overwriting
        dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

        set_value(dol, "e/f", 3)
        # {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": 3}}

        set_value(dol, "e/f/1/g", 3)
        # {"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": [None, {"g": 3}]}}

        # Toplevel new key
        dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}
        set_value(dol, "h", 4)
        # {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4}

        set_value(dol, "i/j/k", 4)
        # {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4, "i": {"j": {"k": 4}}}

        set_value(dol, "j/0/k", 4)
        # {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4, "i": {"j": {"k": 4}}, "j": [{"k": 4}], }

        # Toplevel is list new key
        dol = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}, 2, 3]

        set_value(dol, "0/k", 4)
        # [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "k": 4}, 2, 3]

        set_value(dol, "0", 1)
        # [1, 2, 3]

    """

    # Split into single keys and convert to int if possible
    keys = []
    for k in key.split(splitval):
        try:
            newkey = int(k)
        except:
            newkey = k
        keys.append(newkey)
    next_keys = keys[1:] + [None]

    is_leaf = [False for k in keys]
    is_leaf[-1] = True

    next_is_leaf = is_leaf[1:] + [None]

    parent = None
    last_key = None
    for key, leaf, next_key, next_leaf in zip(keys, is_leaf, next_keys, next_is_leaf):

        if isinstance(key, str):
            # list_or_dict must be a dict
            if isinstance(list_or_dict, list):
                # Not possible
                raise ValueError("Trying to add a key to a list")
            elif not isinstance(list_or_dict, dict) or key not in list_or_dict:
                # Replace value based on next key -> This is only met if
                list_or_dict[key] = {} if isinstance(next_key, str) else []

        else:
            # list_or_dict must be list
            if isinstance(list_or_dict, list):
                if key >= len(list_or_dict):
                    # Append to list and pad with None
                    n_add = key - len(list_or_dict) + 1
                    list_or_dict += [None] * n_add
            elif not isinstance(list_or_dict, dict):
                if parent is None:
                    # We are at top level
                    list_or_dict = [None] * (key + 1)
                else:
                    parent[last_key] = [None] * (key + 1)

        if leaf:
            list_or_dict[key] = val
        else:
            if not isinstance(list_or_dict[key], (dict, list)):
                # Replacement condition met
                list_or_dict[key] = (
                    {} if isinstance(next_key, str) else [None] * (next_key + 1)
                )

            parent = list_or_dict
            last_key = key
            list_or_dict = list_or_dict[key]


def contains_key(nested_thing, key, splitval="/", expand=True):
    """
    Tests if the path like key can find an object in the nested_thing.
    """
    try:
        retrieve(nested_thing, key, splitval=splitval, expand=expand)
        return True
    except Exception:
        return False


def update(to_update, to_update_with, splitval="/", expand=True):
    def _update(key, value):
        set_value(to_update, key, value, splitval=splitval)

    walk(to_update_with, _update, splitval=splitval, pass_key=True)


def get_leaf_names(nested_thing):
    class LeafGetter:
        def __call__(self, key, value):
            if not hasattr(self, "keys"):
                self.keys = []

            self.keys += [key]

    LG = LeafGetter()

    walk(nested_thing, LG, pass_key=True)

    return LG.keys


def strenumerate(iterable):
    """Works just as enumerate, but the returned index is a string.

    Parameters
    ----------
    iterable : Iterable
        An (guess what) iterable object.
    """

    for i, val in enumerate(iterable):
        yield str(i), val


def cached_function(fn):
    """a very rough cache for function calls. Highly experimental. Only
    active if activated with environment variable."""
    # secret activation code
    if not os.environ.get("EDFLOW_CACHED_FUNC", 0) == "42":
        return fn
    cache_dir = os.path.join(os.environ.get("HOME"), "var", "edflow_cached_func")
    os.makedirs(cache_dir, exist_ok=True)

    def wrapped(*args, **kwargs):
        fnhash = fn.__name__
        callargs = (args, kwargs)
        callhash = str(len(pickle.dumps(callargs)))
        fullhash = fnhash + callhash
        pfname = fullhash + ".p"
        ppath = os.path.join(cache_dir, pfname)
        if not os.path.exists(ppath):
            # compute
            print("Computing {}".format(ppath))
            result = fn(*args, **kwargs)
            # and cache
            with open(ppath, "wb") as f:
                pickle.dump(result, f)
            print("Cached {}".format(ppath))
        else:
            # load from cache
            with open(ppath, "rb") as f:
                result = pickle.load(f)
        return result

    return wrapped


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class Printer(object):
    """For usage with walk: collects strings for printing"""

    def __init__(self, string_fn):
        self.str = ""
        self.string_fn = string_fn

    def __call__(self, key, obj):
        self.str += self.string_fn(key, obj) + "\n"

    def __str__(self):
        return self.str


class TablePrinter(object):
    """For usage with walk: Collects string to put in a table."""

    def __init__(self, string_fn, names=None, jupyter_style=False):
        if names is None:
            self.vals = []
            self.has_header = False
        else:
            self.vals = [names]
            self.has_header = True
        self.string_fn = string_fn
        self.jupyter_style = jupyter_style

    def __call__(self, key, obj):
        self.vals += [list(self.string_fn(key, obj))]

    def __str__(self):
        # get width of table:
        col_widths = [0] * len(self.vals[0])
        for val in self.vals:
            for i, entry in enumerate(val):
                col_widths[i] = max(col_widths[i], len(entry) + 2)

        form = "|"
        for cw in col_widths:
            form += " {: >" + str(cw) + "} |"
        form += "\n"

        ref_line = form.format(*self.vals[0])
        sep = "-" * (len(ref_line) - 1)
        hsep = "=" * (len(ref_line) - 1)

        chars = np.array(list(ref_line))
        crossings = np.where(chars == "|")[0]
        if not self.jupyter_style:
            for c in crossings:
                sep = sep[:c] + "+" + sep[c + 1 :]
                hsep = hsep[:c] + "+" + hsep[c + 1 :]
        else:
            for c in crossings:
                hsep = hsep[:c] + "|" + hsep[c + 1 :]
        sep += "\n"
        hsep += "\n"

        if self.jupyter_style:
            sep = ""
            hsep = hsep.replace("=", "-")

        table_str = sep
        for i, val in enumerate(self.vals):
            table_str += form.format(*val)
            if self.has_header and i == 0:
                table_str += hsep
            else:
                table_str += sep

        return table_str


def pprint_str(nested_thing, heuristics=None):
    """Formats nested objects as string and tries to give relevant information.

    Parameters
    ----------
    nested_thing : dict or list
        Some nested object.
    heuristics : Callable
        If given this should produce the string, which
        is printed as description of a leaf object.
    """

    if heuristics is None:

        def heuristics(key, obj):
            if isinstance(obj, np.ndarray):
                return "{}: np array - {}".format(key, obj.shape)
            else:
                return "{}: {} - {}".format(key, type(obj), obj)

    P = Printer(heuristics)

    walk(nested_thing, P, pass_key=True)

    return str(P)


def pprint(nested_thing, heuristics=None):
    """Prints nested objects and tries to give relevant information.

    Parameters
    ----------
    nested_thing : dict or list
        Some nested object.
    heuristics : Callable
        If given this should produce the string, which
        is printed as description of a leaf object.
    """
    print(pprint_str(nested_thing, heuristics))


def pp2mkdtable(nested_thing, jupyter_style=False):
    """Turns a formatted string into a markdown table."""

    def heuristics(key, obj):
        if hasattr(obj, "shape"):
            s = str(obj) if obj.shape == () else str(obj.shape)
            return key, str(obj.__class__.__name__), s
        elif hasattr(obj, "size"):
            return key, str(obj.__class__.__name__), str(obj.size())
        else:
            return key, str(obj.__class__.__name__), str(obj)

    P = TablePrinter(
        heuristics, names=["Name", "Type", "Content"], jupyter_style=jupyter_style
    )

    walk(nested_thing, P, pass_key=True)

    return str(P)


def edprint(nested_thing):
    """Prints every leaf variable in :attr:`nested_thing` in the form of a 
    table.

    Parameters
    ----------
    nested_thing : dict or list
        Some nested object.
    """

    if (
        __COULD_HAVE_IPYTHON__
        and hasattr(get_ipython(), "config")
        and "IPKernelApp" in get_ipython().config
    ):
        display(Markdown(pp2mkdtable(nested_thing, True)))
    else:
        print(pp2mkdtable(nested_thing, False))


class NoModel(object):
    def __init__(self, config):
        self.config = config


if __name__ == "__main__":
    from edflow.data.util import plot_datum

    image = np.ones([100, 100, 3])
    nested = {
        "step": 1,
        "stuff": {"a": 1, "b": [1, 2, 3]},
        "more": [{"c": 1}, 2, [3, 4]],
        "image": image,
    }

    def fn(val):
        print(val)
        return -val

    new = walk(nested, fn)

    print(nested)
    print(new)

    pprint(nested)

    print(pp2mkdtable(nested))

    plot_datum(nested)

    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}

    print(dol)
    print("set_value(dol, 'a/0', 3)")
    set_value(dol, "a/0", 3)

    print(dol)
    print({"a": [3, 2], "b": {"c": {"d": 1}, "e": 2}})
    # {'a': [3, 2], 'b': {'c': {'d': 1}, 'e': 2}}}
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'b/e', 3)")
    set_value(dol, "b/e", 3)

    print(dol)
    print({"a": [3, 2], "b": {"c": {"d": 1}, "e": 3}})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'a/1/f', 3)")
    set_value(dol, "a/1/f", 3)

    print(dol)
    print({"a": [3, {"f": 3}], "b": {"c": {"d": 1}, "e": 3}})
    print("-" * 10)

    # Append to list
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}

    print(dol)
    print("set_value(dol, 'a/2', 3)")
    set_value(dol, "a/2", 3)

    print(dol)
    print({"a": [1, 2, 3], "b": {"c": {"d": 1}, "e": 2}})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'a/5', 6)")
    set_value(dol, "a/5", 6)

    print(dol)
    print({"a": [1, 2, 3, None, None, 6], "b": {"c": {"d": 1}, "e": 2}})
    print("-" * 10)

    # Add key
    dol = {"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}}

    print(dol)
    print("set_value(dol, 'f', 3)")
    set_value(dol, "f", 3)

    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}, "e": 2}, "f": 3})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'b/1', 3)")
    set_value(dol, "b/1", 3)

    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}, "e": 2, 1: 3}, "f": 3})
    print("-" * 10)

    # Raises Error:
    # Appending key to list
    print("# print(set_value(dol, 'a/g', 3))  # should raise")
    # print(set_value(dol, 'a/g', 3))  # should raise

    # Appending key to leaf
    print(dol)
    print("set_value(dol, 'f/a', 3)")
    set_value(dol, "f/a", 3)
    print(dol)
    print("-" * 10)

    # Fancy Overwriting
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    print(dol)
    print("set_value(dol, 'e/f', 3)")
    set_value(dol, "e/f", 3)

    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": 3}})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'e/f/1/g', 3)")
    set_value(dol, "e/f/1/g", 3)

    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}}, "e": {"f": [None, {"g": 3}]}})
    print("-" * 10)

    # Toplevel new key
    dol = {"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}

    print(dol)
    print("set_value(dol, 'h', 4)")
    set_value(dol, "h", 4)
    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'i/j/k', 4)")
    set_value(dol, "i/j/k", 4)
    print(dol)
    print({"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "h": 4, "i": {"j": {"k": 4}}})
    print("-" * 10)

    print(dol)
    print("set_value(dol, 'j/0/k', 4)")
    set_value(dol, "j/0/k", 4)
    print(dol)
    print(
        {
            "a": [1, 2],
            "b": {"c": {"d": 1}},
            "e": 2,
            "h": 4,
            "i": {"j": {"k": 4}},
            "j": [{"k": 4}],
        }
    )
    print("-" * 10)

    # Toplevel is list new key
    dol = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2}, 2, 3]

    print(dol)
    print("set_value(dol, '0/k', 4)")
    set_value(dol, "0/k", 4)
    print(dol)
    ref = [{"a": [1, 2], "b": {"c": {"d": 1}}, "e": 2, "k": 4}, 2, 3]
    print(ref == dol)
    print(ref)
    print("-" * 10)

    print(dol)
    print("set_value(dol, '0', 1)")
    set_value(dol, "0", 1)
    print(dol)
    ref = [1, 2, 3]
    print(ref == dol)
    print(ref)
    print("-" * 10)

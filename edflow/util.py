'''Some Utility functions, that make yur life easier but don't fit in any
better catorgory than util.'''

import tensorflow as tf
import os, pickle


def make_linear_var(step,
                    start, end,
                    start_value, end_value,
                    clip_min=0.0, clip_max=1.0):
    r"""Linear from :math:`(a, \alpha)` to :math:`(b, \beta)`, i.e.
    :math:`y = (\beta - \alpha)/(b - a) * (x - a) + \alpha`

    Args:
        step (tf.Tensor): :math:`x`
        start: :math:`a`
        end: :math:`b`
        start_value: :math:`\alpha`
        end_value: :math:`\beta`
        clip_min: Minimal value returned.
        clip_max: Maximum value returned.

    Returns:
        tf.Tensor: :math:`y`
    """
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def walk(dict_or_list, fn, inplace=False):
    '''Walk a nested list and/or dict recursively and call fn on all non
    list or dict objects.

    Example:

    :codeblock: python

    dol = {'a': [1, 2], 'b': {'c': 3, 'd': 4}}

    def fn(val):
        return val**2

    result = walk(dol, fn)
    print(result)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}
    print(dol)  # {'a': [1, 2], 'b': {'c': 3, 'd': 4}}

    result = walk(dol, fn, inplace=True)
    print(result)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}
    print(dol)  # {'a': [1, 4], 'b': {'c': 9, 'd': 16}}


    Args:
        dict_or_list (dict or list): Possibly nested list or dictionary.
        fn (Callable): Applied to each leave of the nested list_dict-object.
        inplace (bool): If False, a new object with the same structure
            and the results of fn at the leaves is created. If True the leaves
            are replaced by the results of fn.

    Returns:
        dict or list: The resulting nested list-dict-object with the results of
            fn at its leaves.
    '''

    def call(value):
        if isinstance(value, (list, dict)):
            return walk(value, fn)
        else:
            return fn(value)

    if not inplace:
        if isinstance(dict_or_list, list):
            results = [call(val) for val in dict_or_list]
        elif isinstance(dict_or_list, dict):
            results = {k: call(val) for k, val in dict_or_list.items()}
        else:
            results = fn(dict_or_list)
    else:
        if isinstance(dict_or_list, list):
            for i, val in enumerate(dict_or_list):
                dict_or_list[i] = call(val)
        elif isinstance(dict_or_list, dict):
            for k, val in dict_or_list.items():
                dict_or_list[k] = call(val)
        else:
            dict_or_list = fn(dict_or_list)

        results = dict_or_list

    return results


def retrieve(key, list_or_dict, splitval='/'):
    '''Given a nested list or dict return the desired value at key.

    Args:
        key (str): key/to/value, path like string describing all keys
            necessary to consider to get to the desired value. List indices
            can also be passed here.
        list_or_dict (list or dict): Possibly nested list or dictionary.
        splitval (str): String that defines the delimiter between keys of the
            different depth levels in `key`.

    Returns:
        The desired value :)
    '''

    keys = key.split(splitval)

    for key in keys:
        if isinstance(list_or_dict, dict):
            list_or_dict = list_or_dict[key]
        else:
            list_or_dict = list_or_dict[int(key)]

    return list_or_dict


def cached_function(fn):
    """a very rough cache for function calls. Highly experimental. Only
    active if activated with environment variable."""
    if not os.environ.get("EDFLOW_CACHED_FUNC", 0) == "42": # secret activation code
        return fn
    cache_dir = os.path.join(os.environ.get("HOME"), "var", "edflow_cached_func")
    os.makedirs(cache_dir, exist_ok = True)
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


if __name__ == '__main__':
    nested = {'step': 1,
              'stuff': {'a': 1, 'b': [1,2,3]},
              'more': [{'c': 1}, 2, [3, 4]]}

    def fn(val):
        print(val)
        return -val

    new = walk(nested, fn)

    print(nested)
    print(new)

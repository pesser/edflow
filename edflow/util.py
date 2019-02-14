'''Some Utility functions, that make yur life easier but don't fit in any
better catorgory than util.'''

import numpy as np
import tensorflow as tf
import os, pickle


def make_linear_var(step,
                    start, end,
                    start_value, end_value,
                    clip_min=None, clip_max=None):
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
    if clip_min is None:
        clip_min = min(start_value, end_value)
    if clip_max is None:
        clip_max = max(start_value, end_value)
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def make_exponential_var(step,
                         start, end,
                         start_value, end_value,
                         decay):
    r"""Exponential from :math:`(a, \alpha)` to :math:`(b, \beta)` with decay
    rate decay.

    Args:
        step (tf.Tensor): :math:`x`
        start: :math:`a`
        end: :math:`b`
        start_value: :math:`\alpha`
        end_value: :math:`\beta`
        decay: Decay rate

    Returns:
        tf.Tensor: :math:`y`
    """
    startstep = start
    endstep = (np.log(end_value) - np.log(start_value))/np.log(decay)
    stepper = make_linear_var(step, start, end, startstep, endstep)
    return tf.math.pow(decay, stepper)*start_value


def walk(dict_or_list, fn, inplace=False, pass_key=False, prev_key=''):
    '''Walk a nested list and/or dict recursively and call fn on all non
    list or dict objects.

    Example:

    ..codeblock:: python

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
        pass_key (bool): Also passes the key or index of the leave element to
            ``fn``.
        prev_key (str): If ``pass_key == True`` keys of parent nodes are passed
            to calls of ``walk`` on child nodes to accumulate the keys.

    Returns:
        dict or list: The resulting nested list-dict-object with the results of
            fn at its leaves.
    '''

    if not pass_key:
        def call(value):
            if isinstance(value, (list, dict)):
                return walk(value, fn, inplace)
            else:
                return fn(value)
    else:
        def call(key, value):
            key = os.path.join(prev_key, key)
            if isinstance(value, (list, dict)):
                return walk(value, fn, inplace, pass_key=True, prev_key=key)
            else:
                return fn(key, value)

    if isinstance(dict_or_list, list):
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

    try:
        visited = []
        for key in keys:
            if isinstance(list_or_dict, dict):
                list_or_dict = list_or_dict[key]
            else:
                list_or_dict = list_or_dict[int(key)]
            visited += [key]
    except Exception as e:
        print('Key not found: {}, seen: {}'.format(keys, visited))
        raise e

    return list_or_dict


def contains_key(nested_thing, key, splitval='/'):
    '''Tests if the path like key can find an object in the nested_thing.
    Has the same signature as :function:`retrieve`.'''
    try:
        retrieve(nested_thing, key, splitval)
        return True
    except:
        return False


def strenumerate(iterable):
    '''Works just as enumerate, but the returned index is a string.

    Args:
        iterable (Iterable): An (guess what) iterable object.
    '''

    for i, val in enumerate(iterable):
        yield str(i), val


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

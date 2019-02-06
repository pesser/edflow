'''Some Utility functions, that make yur life easier but don't fit in any
better catorgory than util.'''

import numpy as np
import tensorflow as tf
import os, pickle
import cv2


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


def pprint(nested_thing, heuristics=None):
    '''Prints nested objects and tries to give relevant information.
    
    Args:
        nested_thing (dict or list): Some nested object.
        heuristics (Callable): If given this should produce the string, which
            is printed as description of a leaf object.
    '''

    if heuristics is None:
        def heuristics(key, obj):
            if isinstance(obj, np.ndarray):
                return '{}: np array - {}'.format(key, obj.shape)
            else:
                return '{}: {} - {}'.format(key, type(obj), obj)

    class Printer(object):
        def __init__(self, string_fn):
            self.str = ''
            self.string_fn = string_fn

        def __call__(self, key, obj):
            self.str += self.string_fn(key, obj) + '\n'

        def __str__(self):
            return self.str

    P = Printer(heuristics)

    walk(nested_thing, P, pass_key=True)

    print(P)


def flow2hsv(flow):                                                             
    '''Given a Flowmap of shape ``[W, H, 2]`` calculates an hsv image,
    showing the relative magnitude and direction of the optical flow.

    Args:
        flow (np.array): Optical flow with shape ``[W, H, 2]``.
    Returns:
        np.array: Containing the hsv data.
    '''

    # prepare array - value is always at max
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 2] = 255

    # magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # make it colorful
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return hsv

def hsv2rgb(hsv):
    '''color space conversion hsv -> rgb. simple wrapper for nice name.'''
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def flow2rgb(flow):
    '''converts a flow field to an rgb color image.

    Args:
        flow (np.array): optical flow with shape ``[W, H, 2]``.
    Returns:
        np.array: Containing the rgb data. Color indicates orientation,
            intensity indicates magnitude.
    '''

    return hsv2rgb(flow2hsv(flow))


def plot_datum(nested_thing,
               savename='datum.png',
               heuristics=None,
               plt_functions=None):
    '''Plots all data in the nested_thing as best as can.

    If heuristics is given, this determines how each leaf datum is converted
    to something plottable.

    Args:
        nested_thing (dict or list): Some nested object.
        savename (str): ``Path/to/the/plot.png``.
        heuristics (Callable): If given this should produce a string specifying
            the kind of data of the leaf. If ``None`` determinde automatically.
        plt_functions (dict of Callables): Maps a ``kind`` to a function which
            can plot it. Each callable must be able to receive a the key, the
            leaf object and the Axes to plot it in.
    '''

    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def im_fn(key, im, ax):
        if im.shape[-1] == 1:
            im = np.squeeze(im)

        ax.imshow(im)
        ax.set_ylabel(key, rotation=0)

    def heatmap_fn(key, im, ax):
        '''Assumes that heatmap shape is [H, W, N]'''
        im = np.mean(im, axis=-1)
        im_fn(key, im, ax)

    def flow_fn(key, im, ax):
        im = flow2rgb(im)
        im_fn(key, im, ax)

    def other_fn(key, obj, ax):
        text = '{}: {} - {}'.format(key, type(obj), obj)
        ax.axis('off')
        # ax.imshow(np.ones([10, 100]))
        ax.text(0, 0, text)



    plt_fns = {'image': im_fn,
               'heat': heatmap_fn,
               'flow': flow_fn,
               'other': other_fn}

    if plt_functions is not None:
        plt_fns.update(plt_functions)

    if heuristics is None:
        def heuristics(key, obj):
            if isinstance(obj, np.ndarray):
                if len(obj.shape) > 3 or len(obj.shape) < 2:
                    # This is no image -> Maybe later implement sequence fn
                    return 'other'
                else:
                    if obj.shape[-1] == 3:
                        return 'image'
                    elif obj.shape[-1] == 2:
                        return 'flow'
                    else:
                        return 'heat'
            return 'other'

    class Plotter(object):
        def __init__(self, kind_fn, savename):
            self.kind_fn = kind_fn
            self.savename = savename
            self.buffer = []

        def __call__(self, key, obj):
            kind = self.kind_fn(key, obj)
            self.buffer += [[kind, key, obj]]

        def plot(self):
            n_pl = len(self.buffer)

            f = plt.figure(figsize=(5, 2*n_pl))

            gs = gridspec.GridSpec(n_pl, 1)

            for i, [kind, key, obj] in enumerate(self.buffer):
                ax = f.add_subplot(gs[i])
                plt_fns[kind](key, obj, ax)

            f.savefig(self.savename)

        def __str__(self):
            self.plot()
            return 'Saved Plot at {}'.format(self.savename)

    P = Plotter(heuristics, savename)

    walk(nested_thing, P, pass_key=True)

    print(P)


if __name__ == '__main__':
    image = np.ones([100, 100, 3])
    nested = {'step': 1,
              'stuff': {'a': 1, 'b': [1,2,3]},
              'more': [{'c': 1}, 2, [3, 4]],
              'image': image}

    def fn(val):
        print(val)
        return -val

    new = walk(nested, fn)

    print(nested)
    print(new)

    pprint(nested)

    plot_datum(nested)

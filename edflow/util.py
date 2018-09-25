import tensorflow as tf


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


def walk(dict_or_list, fn):
    '''Walk a nested list and/or dict recursively and call fn on all non
    list or dict objects.'''

    def call(value):
        if isinstance(value, (list, dict)):
            return walk(value, fn)
        else:
            return fn(value)

    if isinstance(dict_or_list, list):
        results = [call(val) for val in dict_or_list]
    elif isinstance(dict_or_list, dict):
        results = {k: call(val) for k, val in dict_or_list.items()}
    else:
        results = fn(dict_or_list)

    return results


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

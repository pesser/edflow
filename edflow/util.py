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

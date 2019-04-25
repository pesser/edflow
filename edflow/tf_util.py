import tensorflow as tf


def make_linear_var(
    step, start, end, start_value, end_value, clip_min=None, clip_max=None
):
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
    linear = (end_value - start_value) / (end - start) * (
        tf.cast(step, tf.float32) - start
    ) + start_value
    return tf.clip_by_value(linear, clip_min, clip_max)


def make_exponential_var(step, start, end, start_value, end_value, decay):
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
    endstep = (np.log(end_value) - np.log(start_value)) / np.log(decay)
    stepper = make_linear_var(step, start, end, startstep, endstep)
    return tf.math.pow(decay, stepper) * start_value

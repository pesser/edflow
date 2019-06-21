import tensorflow as tf
import numpy as np


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


def make_var(step, var_type, options):
    """

    # usage within trainer
    grad_weight = make_var(step=self.global_step,
    var_type=self.config["grad_weight"]["var_type"],
    options=self.config["grad_weight"]["options"])

    # within yaml file
    grad_weight:
      var_type: linear
      options:
        start:      50000
        end:        60000
        start_value:  0.0
        end_value: 1.0
        clip_min: 1.0e-6
        clip_max: 1.0

    Parameters
    ----------
    step
    var_type
    options

    Returns
    -------

    """
    switch = {"linear": make_linear_var, "exponential": make_exponential_var}
    return switch[var_type](step=step, **options)

import tensorflow as tf
import numpy as np


def make_linear_var(
    step, start, end, start_value, end_value, clip_min=None, clip_max=None, **kwargs
):
    r"""
    Linear from :math:`(a, \alpha)` to :math:`(b, \beta)`, i.e.
    :math:`y = (\beta - \alpha)/(b - a) * (x - a) + \alpha`

    Parameters
    ----------
    step : tf.Tensor
        :math:`x`
    start : int
        :math:`a`
    end : int
        :math:`b`
    start_value : float
        :math:`\alpha`
    end_value : float
        :math:`\beta`
    clip_min : int
        Minimal value returned.
    clip_max : int
        Maximum value returned.

    Returns
    -------
    :math:`y` : tf.Tensor
    """

    if clip_min is None:
        clip_min = min(start_value, end_value)
    if clip_max is None:
        clip_max = max(start_value, end_value)
    delta_value = end_value - start_value
    delta_step = end - start
    linear = (
        delta_value / delta_step * (tf.cast(step, tf.float32) - start) + start_value
    )
    return tf.clip_by_value(linear, clip_min, clip_max)


def make_periodic_step(step, start_step: int, period_duration_in_steps: int, **kwargs):
    """
    Returns step within the unit period cycle specified

    Parameters
    ----------
    step: tf.Tensor
        step variable
    start_step : int
        an offset parameter specifying when the first period begins
    period_duration_in_steps : int
        period duration of step

    Returns
    -------
    unit_step
        step within unit cycle period
    """
    step = tf.to_float(step)
    unit_step = step - start_step
    unit_step = tf.clip_by_value(unit_step, 0.0, unit_step)
    unit_step = tf.mod(unit_step, period_duration_in_steps)
    return unit_step


def make_exponential_var(step, start, end, start_value, end_value, decay, **kwargs):
    r"""
    Exponential from :math:`(a, \alpha)` to :math:`(b, \beta)` with decay
    rate decay.

    Parameters
    ----------
    step : tf.Tensor
        :math:`x`
    start : int
        :math:`a`
    end : int
        :math:`b`
    start_value : float
        :math:`\alpha`
    end_value : float
        :math:`\beta`
    decay : int
        Decay rate

    Returns
    -------
    :math:`y` : tf.Tensor

    """

    startstep = start
    endstep = (np.log(end_value) - np.log(start_value)) / np.log(decay)
    stepper = make_linear_var(step, start, end, startstep, endstep)
    return tf.math.pow(decay, stepper) * start_value


def make_staircase_var(
    step,
    start,
    start_value,
    step_size,
    stair_factor,
    clip_min=0.0,
    clip_max=1.0,
    **kwargs
):
    r"""
    Parameters
    ----------
    step : tf.Tensor
        :math:`x`
    start : int
        :math:`a`
    start_value : float
        :math:`\alpha`
    step_size: int
        after how many steps the value should be changed
    stair_factor: float
        factor that the value is multiplied with at every 'step_size' steps
    clip_min : int
        Minimal value returned.
    clip_max : int
        Maximum value returned.

    Returns
    -------
    :math:`y` : tf.Tensor

    """
    stair_case = (
        stair_factor ** ((tf.cast(step, tf.float32) - start) // step_size) * start_value
    )
    stair_case_clipped = tf.clip_by_value(stair_case, clip_min, clip_max)
    return stair_case_clipped


def make_periodic_wrapper(step_function):
    """
    A wrapper to wrap the step variable of a step function into a periodic step variable.
    Parameters
    ----------
    step_function: callable
        the step function where to exchange the step variable with a periodic step variable

    Returns
    -------
    a function with periodic steps

    """

    def make_periodic_xxx_var(step, **kwargs):
        new_step = make_periodic_step(step, **kwargs)
        return step_function(new_step, **kwargs)

    return make_periodic_xxx_var


def make_var(step, var_type, options):
    r"""
    Example
    -------
    usage within trainer

    .. code-block:: python

        grad_weight = make_var(step=self.global_step,
        var_type=self.config["grad_weight"]["var_type"],
        options=self.config["grad_weight"]["options"])

    within yaml file

    .. code-block:: yaml

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
    step: tf.Tensor
        scalar tensor variable
    var_type: str
        a string from ["linear", "exponential", "staircase"]
    options: dict
        keyword arguments passed to specific 'make_xxx_var' function

    Returns
    -------
    :math:`y` : tf.Tensor
    """
    switch = {
        "linear": make_linear_var,
        "exponential": make_exponential_var,
        "staircase": make_staircase_var,
        "periodic_linear": make_periodic_wrapper(make_linear_var),
        "periodic_staircase": make_periodic_wrapper(make_staircase_var),
    }
    return switch[var_type](step=step, **options)


if __name__ == "__main__":
    tf.enable_eager_execution()
    import seaborn as sns

    sns.set()
    from matplotlib import pyplot as plt
    import pprint

    N = 10000
    t = tf.range(0, N)

    schedule_config = {
        "var_type": "linear",
        "options": {
            "start": 2000,
            "end": 3000,
            "start_value": 0.0,
            "end_value": 1.0,
            "clip_min": 1.0e-6,
            "clip_max": 1.0,
        },
    }
    pp = pprint.PrettyPrinter(indent=4)
    scheduled_variable = make_var(
        t, schedule_config["var_type"], schedule_config["options"]
    )
    textstr = pp.pformat(schedule_config["options"])
    sns.lineplot(t, scheduled_variable, label=textstr)
    ax = plt.gca()
    ax.legend(frameon=True, fontsize=16)
    plt.show()

    ###

    schedule_config = {
        "var_type": "periodic_linear",
        "options": {
            "start": 0,
            "end": 1000,
            "start_value": 0.0,
            "end_value": 1.0,
            "clip_min": 1.0e-6,
            "clip_max": 1.0,
            "start_step": 4999,
            "period_duration_in_steps": 2000,
        },
    }
    pp = pprint.PrettyPrinter(indent=4)
    scheduled_variable = make_var(
        t, schedule_config["var_type"], schedule_config["options"]
    )
    textstr = pp.pformat(schedule_config["options"])
    sns.lineplot(t, scheduled_variable, label=textstr)
    ax = plt.gca()
    ax.legend(frameon=True, fontsize=16, loc=1)
    plt.show()

    ###

    schedule_config = {
        "var_type": "periodic_staircase",
        "options": {
            "start": 0,
            "start_value": 1.0,
            "step_size": 1000,
            "stair_factor": 10,
            "clip_min": 1.0,
            "clip_max": 1.0e3,
            "start_step": 1000,
            "period_duration_in_steps": 4000,
        },
    }
    pp = pprint.PrettyPrinter(indent=4)
    scheduled_variable = make_var(
        t, schedule_config["var_type"], schedule_config["options"]
    )
    textstr = pp.pformat(schedule_config["options"])
    sns.lineplot(t, scheduled_variable, label=textstr)
    ax = plt.gca()
    ax.legend(frameon=True, fontsize=16, loc=1)
    plt.show()

    ## TODO: make this nice with some annotations

    ###

    schedule_config = {
        "var_type": "staircase",
        "options": {
            "start": 0,
            "start_value": 1.0,
            "step_size": 1000,
            "stair_factor": 10,
            "clip_min": 1.0,
            "clip_max": 1.0e3,
            "start_step": 1000,
            "period_duration_in_steps": 4000,
        },
    }
    pp = pprint.PrettyPrinter(indent=4)
    scheduled_variable = make_var(
        t, schedule_config["var_type"], schedule_config["options"]
    )
    textstr = pp.pformat(schedule_config["options"])
    sns.lineplot(t, scheduled_variable, label=textstr)
    ax = plt.gca()
    ax.legend(frameon=True, fontsize=16, loc=1)
    plt.show()

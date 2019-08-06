import tensorflow as tf
import numpy as np
import pprint


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
    step
    start_step : int
    period_duration_in_steps : int

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


def make_exponential_var(step, start, end, start_value, end_value, decay):
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
    stair_case = (
        stair_factor ** ((tf.cast(step, tf.float32) - start) // step_size) * start_value
    )
    stair_case_clipped = tf.clip_by_value(stair_case, clip_min, clip_max)
    return stair_case_clipped


def make_periodic_linear_var(step, **kwargs):
    # TODO: this can be done better with a simple decorator
    periodic_step = make_periodic_step(step, **kwargs)
    periodic_variable = make_linear_var(periodic_step, **kwargs)
    return periodic_variable


def make_periodic_staircase_var(step, **kwargs):
    # TODO: this can be done better with a simple decorator
    periodic_step = make_periodic_step(step, **kwargs)
    periodic_variable = make_staircase_var(periodic_step, **kwargs)
    return periodic_variable


def make_var(step, var_type, options):
    """
    usage within trainer

        grad_weight = make_var(step=self.global_step,
        var_type=self.config["grad_weight"]["var_type"],
        options=self.config["grad_weight"]["options"])

    within yaml file

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
    :math:`y` : tf.Tensor

    """
    switch = {
        "linear": make_linear_var,
        "exponential": make_exponential_var,
        "staircase": make_staircase_var,
        "periodic_linear": make_periodic_linear_var,
        "periodic_staircase": make_periodic_staircase_var,
    }
    return switch[var_type](step=step, **options)


if __name__ == "__main__":
    tf.enable_eager_execution()
    import seaborn as sns

    sns.set()
    from matplotlib import pyplot as plt

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

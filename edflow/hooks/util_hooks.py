from edflow.hooks.hook import Hook


class IntervalHook(Hook):
    """This hook manages a set of hooks, which it will run each time its
    interval flag is set to True."""

    def __init__(
        self,
        hooks,
        interval,
        start=None,
        stop=None,
        modify_each=None,
        modifier=lambda interval: 2 * interval,
        max_interval=None,
        get_step=None,
    ):
        """Args:
            hook (list of Hook): The set of managed hooks. Each must implement
                the methods of a :class:`Hook`.
            interval (int): The number of steps after which the managed hooks
                are run.
            start (int): If `start` is not None, the first time the hooks are
                run ist after `start` number of steps have been made.
            stop (int): If given, this hook is not evaluated anymore after
                `stop` steps.
            modify_each (int): If given, `modifier` is called on the interval
                after this many executions of thois hook. If `None` it is set
                to :attr:`interval`. In case you do not want any mofification
                you can either set :attr:`max_interval` to :attr:`interval` or
                choose the modifier to be `lambda x: x` or set
                :attr:`modify_each` to `float(inf)`.
            modifier (Callable): See `modify_each`.
            max_interval (int): If given, the modifier can only increase the
                interval up to this number of steps.
            get_step (Callable): If given, prefer over the use of batch index
                to determine run condition, e.g. to run based on global step.
        """

        self.hooks = hooks

        self.base_interval = interval

        inf = float("inf")
        self.start = start if start is not None else -1
        self.stop = stop if stop is not None else inf
        self.modival = modify_each if modify_each is not None else interval
        self.modifier = modifier
        self.max_interval = max_interval if max_interval is not None else inf
        self.get_step = get_step

        self.counter = 0

    def run_condition(self, step, is_before=False):
        if self.get_step is not None:
            step = self.get_step()
        if step > self.start and step <= self.stop:
            if step % self.base_interval == 0:
                self.counter += 1 if is_before else 0
                return True
        return False

    def maybe_modify(self, step):
        if self.counter % self.modival == 0:
            new_interval = self.modifier(self.base_interval)
            self.base_interval = min(self.max_interval, new_interval)

    def before_epoch(self, *args, **kwargs):
        """Called before each epoch."""

        for hook in self.hooks:
            hook.before_epoch(*args, **kwargs)

    def before_step(self, step, *args, **kwargs):
        """Called before each step. Can update any feeds and fetches."""

        if self.run_condition(step, True):
            for hook in self.hooks:
                hook.before_step(step, *args, **kwargs)

    def after_step(self, step, *args, **kwargs):
        """Called after each step."""

        if self.run_condition(step, False):
            for hook in self.hooks:
                hook.after_step(step, *args, **kwargs)

            self.maybe_modify(step)

    def after_epoch(self, *args, **kwargs):
        """Called after each epoch."""

        for hook in self.hooks:
            hook.after_epoch(*args, **kwargs)

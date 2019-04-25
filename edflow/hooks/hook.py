from edflow.custom_logging import get_default_logger


class Hook(object):
    """Base Hook to be inherited from. Hooks can be passed to
    :class:`HookedModelIterator` and will be called at fixed intervals.

    The inheriting class only needs to overwrite those methods below, which are
    of interest.

    In principle a hook can be used to do anything during its execution.
    It is intended to be used as an update mechanism for the standard fetches
    and feeds, passed to the session managed e.g. by a
    :class:`HookedModelIterator` and then working with the results of the
    run call to the session.

    Assuming there is one hook that is passed to a
    :class:`HookedModelIterator` its methods will be called in the following
    fashion:

    .. code-block:: python

        for epoch in epochs:
            hook.before_epoch(epoch)
            for i, batch in enumerate(batches):
                fetches, feeds = some_function(batch)
                hook.before_step(i, fetches, feeds)  # change fetches & feeds

                results = session.run(fetches, feed_dict=feeds)

                hook.after_step(i, results)
            hook.after_epoch(epoch)
    """

    def before_epoch(self, epoch):
        """Called before each epoch.

        Args:
            epoch (int): Index of epoch that just started.
        """

        pass

    def before_step(self, step, fetches, feeds, batch):
        """Called before each step. Can update any feeds and fetches.

        Args:
            step (int): Current training step.
            fetches (list or dict): Fetches for the next session.run call.
            feeds (dict): Data used at this step.
            batch (list or dict): All data available at this step.
        """

        pass

    def after_step(self, step, last_results):
        """Called after each step.

        Args:
            step (int): Current training step.
            last_results (list): Results from last time this hook was called.
        """

        pass

    def after_epoch(self, epoch):
        """Called after each epoch.

        Args:
            epoch (int): Index of epoch that just ended.
        """

        pass

    def at_exception(self, exception):
        """Called when an exception is raised.

        Args:
            exception (Exception): The exception which is being raised. Will
                be raised again after all :method:`at_eception` calls have
                been handled.
        """

        pass


def match_frequency(global_hook_frequency, local_hook_frequency):
    r"""Given the global frequency at which hooks are evaluated matches the
    local frequency at which a hook wants to be evaluated s.t. it will
    be at least the global frequency or an integer multiple of it.

    Args:
        global_hook_frequency (int): Step frequency :math:`f_g` at which hooks
            are called.
        local_hook_frequency (int): Step frequency :math:`f_l` at which hook
            wants to be called.

    Returns:
        int: Matching new frequency :math:`f` for the hook, with
            :math:`f =
            \begin{cases}
            f_g &\text{if}\; f_l \leq f_g \\
            \left\lfloor \frac{f_l}{f_g}\right\rfloor\cdot f_g &\text{else}
            \end{cases}`.
    """

    n = max(1, local_hook_frequency // global_hook_frequency)
    return n * global_hook_frequency

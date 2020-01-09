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

    Parameters
    ----------

    Returns
    -------

    """

    def before_epoch(self, epoch):
        """Called before each epoch.

        Parameters
        ----------
        epoch : int
            Index of epoch that just started.

        Returns
        -------

        """

        pass

    def before_step(self, step, fetches, feeds, batch):
        """Called before each step. Can update any feeds and fetches.

        Parameters
        ----------
        step : int
            Current training step.
        fetches : list or dict
            Fetches for the next session.run call.
        feeds : dict
            Data used at this step.
        batch : list or dict
            All data available at this step.

        Returns
        -------

        """

        pass

    def after_step(self, step, last_results):
        """Called after each step.

        Parameters
        ----------
        step : int
            Current training step.
        last_results : list
            Results from last time this hook was called.

        Returns
        -------

        """

        pass

    def after_epoch(self, epoch):
        """Called after each epoch.

        Parameters
        ----------
        epoch : int
            Index of epoch that just ended.

        Returns
        -------

        """

        pass

    def at_exception(self, exception):
        """Called when an exception is raised.

        Parameters
        ----------
        exception :


        Returns
        -------

        Raises
        ------
        be
            raised again after all
        been
            handled

        """

        pass

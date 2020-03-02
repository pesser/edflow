import signal, sys, math
from tqdm import tqdm, trange

from edflow.custom_logging import get_logger
from edflow.util import walk


class ShutdownRequest(Exception):
    """Raised when we receive a SIGTERM signal to shut down. Allows hooks to
    perform final actions such as writing a last checkpoint."""

    pass


class PyHookedModelIterator(object):
    """Implements a similar interface as the :class:`HookedModelIterator` to
    train framework independent models."""

    def __init__(
        self,
        config,
        root,
        model,
        datasets,
        hook_freq=100,
        num_epochs=100,
        hooks=[],
        bar_position=0,
        nogpu=False,
        desc="",
    ):
        """Constructor.

        Parameters
        ----------
        model : object
	    Model class.
        num_epochs : int
	    Number of times to iterate over the data.
        hooks : list
	    List containing :class:`Hook` instances.
        hook_freq : int
	    Frequency at which hooks are evaluated.
        bar_position : int
	    Used by tqdm to place bars at the right
            position when using multiple Iterators in parallel.
        """
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

        self.config = config
        self.root = root

        self.model = model
        self.datasets = datasets
        # backwards compatibility
        self.dataset = datasets["train"]
        self.validation_dataset = datasets["validation"]

        self.num_epochs = num_epochs

        self.hooks = hooks
        self.epoch_hooks = list()

        self.hook_freq = hook_freq

        self.bar_pos = bar_position * 2
        self.desc = desc

        self.logger = get_logger(type(self).__name__)

        self._global_step = 0
        self._batch_step = 0
        self._epoch_step = 0
        self._split = None

    def get_split(self, *args, **kwargs):
        """Get the current split that is processed."""
        return self._split

    def get_global_step(self, *args, **kwargs):
        """Get the global step. The global step corresponds to the number of
        steps the model was trained for. It is updated in each step during
        training but not during evaluation."""
        return self._global_step

    def set_global_step(self, step):
        """Set the global step. Should be done when restoring a model from a
        checkpoint."""
        self._global_step = step

    def get_batch_step(self, *args, **kwargs):
        """Batch index of current run."""
        return self._batch_step

    def get_epoch_step(self, *args, **kwargs):
        """Epoch index of current run."""
        return self._epoch_step

    def reset_global_step(self):
        self.set_global_step(0)

    def increment_global_step(self, *args, **kwargs):
        if not self.config.get("test_mode", False):
            self._global_step += 1
        return self._global_step

    def make_feeds(self, batch):
        # copy of batches
        feeds = walk(batch, lambda val: val)
        return feeds

    def _handle_sigterm(self, signum, frame):
        e = ShutdownRequest()
        self._handle_exception(e)
        sys.exit(0)

    def _handle_exception(self, e):
        for hook in self.hooks:
            hook.at_exception(e)

    def iterate(self, batches):
        """Iterates over the data supplied and feeds it to the model.

        Parameters
        ----------
        batch_iterator : Iterable
	    Iterable returning training data.
        batch_iterator_validation : Iterable
	    Iterable returning validation data or None
        """

        try:
            self._iterate(batches)
        except Exception as e:
            self._handle_exception(e)
            raise e

    def _iterate(self, batches):
        """Iterates over the data supplied and feeds it to the model.

        Parameters
        ----------
        batch_iterator : Iterable
	    Iterable returning training data.
        """

        step_ops = self.step_ops()
        epoch_hooks_only = self.config.get("test_mode", False)

        pos = self.bar_pos
        base = self.desc + " - " if self.desc != "" else ""
        desc_epoch = base + "Epoch"
        desc_batch = base + "Batch"

        # TODO use val freq
        validation_frequency = self.config.get(
            "val_freq", self.config.get("log_freq", -1)
        )
        batches_per_epoch = 0 if epoch_hooks_only else len(batches["train"])
        if "max_batches_per_epoch" in self.config:
            batches_per_epoch = min(
                batches_per_epoch, self.config["max_batches_per_epoch"]
            )
        num_epochs = 1 if epoch_hooks_only else self.num_epochs
        start_epoch = (
            0 if epoch_hooks_only else (self.get_global_step() // batches_per_epoch)
        )
        start_step = (
            0 if epoch_hooks_only else (self.get_global_step() % batches_per_epoch)
        )
        for epoch_step in trange(
            start_epoch,
            num_epochs,
            initial=start_epoch,
            total=num_epochs,
            desc=desc_epoch,
            position=pos,
            dynamic_ncols=True,
            leave=False,
        ):
            self._epoch_step = epoch_step

            ############# run one batch on each split until new epoch or max steps
            batches["train"].reset()
            self.run_hooks(epoch_step, before=True)

            for batch_step in trange(
                start_step,
                batches_per_epoch,
                initial=start_step,
                total=batches_per_epoch,
                desc=desc_batch,
                position=pos + 1,
                dynamic_ncols=True,
                leave=False,
            ):
                self._batch_step = batch_step

                def lazy_split_op(split):
                    def split_op():
                        self._split = split
                        batch = next(batches[split])
                        feeds = self.make_feeds(batch)
                        fetches = step_ops
                        self.run_hooks(batch_step, fetches, feeds, batch, before=True)
                        return self.run(fetches, feed_dict=feeds)

                    return split_op

                results = {"global_step": self.get_global_step()}
                for split in batches:
                    results[split] = lazy_split_op(split)
                self.run_hooks(batch_step, results=results, before=False)
                del results

                self.increment_global_step()

                if self.get_global_step() >= self.config.get("num_steps", float("inf")):
                    break
            self.run_hooks(epoch_step, before=False)
            start_step = 0

            ############# run one epoch on each split
            # only continue a split as long as someone is retrieving results
            for split in batches:
                batches[split].reset()
                self.run_hooks(epoch_step, before=True, epoch_hooks=True)

                tqdm_iterator = trange(
                    len(batches[split]),
                    desc=split,
                    position=pos + 1,
                    dynamic_ncols=True,
                    leave=False,
                )
                for batch_step in tqdm_iterator:
                    self._batch_step = batch_step

                    active = False

                    def lazy_split_op(split):
                        def split_op():
                            nonlocal active
                            active = True
                            self._split = split
                            batch = next(batches[split])
                            feeds = self.make_feeds(batch)
                            fetches = step_ops
                            self.run_hooks(
                                batch_step,
                                fetches,
                                feeds,
                                batch,
                                before=True,
                                epoch_hooks=True,
                            )
                            return self.run(fetches, feed_dict=feeds)

                        return split_op

                    results = {
                        "global_step": self.get_global_step(),
                        split: lazy_split_op(split),
                    }
                    self.run_hooks(
                        batch_step, results=results, before=False, epoch_hooks=True
                    )
                    del results

                    if batches[split].is_new_epoch or not active:
                        tqdm_iterator.update()
                        tqdm_iterator.close()
                        self.logger.info("Done with {}".format(split))
                        break
                self.run_hooks(epoch_step, before=False, epoch_hooks=True)

            if self.get_global_step() >= self.config.get("num_steps", float("inf")):
                break

    def run(self, fetches, feed_dict):
        """Runs all fetch ops and stores the results.

        Parameters
        ----------
        fetches : dict
	    name: Callable pairs.
        feed_dict : dict
	    Passed as kwargs to all fetch ops

        Returns
        -------
        dict
            name: results pairs.
        """

        def fn(fetch_fn):
            return fetch_fn(self.model, **feed_dict)

        results = walk(fetches, fn)

        return results

    def run_hooks(
        self,
        index,
        fetches=None,
        feeds=None,
        batch=None,
        results=None,
        before=True,
        epoch_hooks=False,
    ):
        """Run all hooks and manage their stuff. The passed arguments determine
        which method of the hooks is called.

        Parameters
        ----------
        index : int
	    Current epoch or batch index. This is not necessarily
            the global training step.
        fetches : list or dict
	    Fetches for the next session.run call.
        feeds : dict
	    Feeds for the next session.run call.
        results : same as fetches
	    Results from the last session.run call.
        before : bool
	    If not obvious determines if the before or after
            methods of the hooks should be called.

        Returns
        -------
        test : same as fetches
	    Updated fetches.
        test : dict
	    Updated feeds
        """

        is_step = fetches is not None and feeds is not None
        is_step = is_step or results is not None

        condition = self._global_step % self.hook_freq == 0 or not is_step

        hooks = self.hooks if not epoch_hooks else self.epoch_hooks
        if condition:
            for hook in hooks:
                if before:
                    if is_step:
                        hook.before_step(index, fetches, feeds, batch)
                    else:
                        hook.before_epoch(index)
                else:
                    if is_step:
                        hook.after_step(index, results)
                    else:
                        hook.after_epoch(index)

    def step_ops(self):
        """Defines ops that are called at each step.

        Returns
        -------
            The operation run at each step."""

        raise NotImplementedError()

    def initialize(self, checkpoint_path=None):
        pass

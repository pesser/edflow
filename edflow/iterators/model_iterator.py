import tensorflow as tf
from tqdm import tqdm, trange

from edflow.custom_logging import get_logger
from edflow.util import walk


class HookedModelIterator(object):
    '''DEPRECATED: use PyHookedModelIterator or TFHookedModelIterator!
    Base Trainer class containing useful methods to inherit.'''

    def __init__(self,
                 model,
                 num_epochs,
                 hooks=[],
                 hook_freq=100,
                 bar_position=0,
                 gpu_mem_growth=False,
                 gpu_mem_fraction=None,
                 nogpu=False):
        '''Constructor.

        Args:
            model (object): Model class. Must have an attribute ``inputs`` for
                placeholders and an attribute ``outputs`` for output tensors.
            num_epochs (int): Number of times to iterate over the data.
            hooks (list): List containing :class:`Hook` instances.
            hook_freq (int): Frequency at which hooks are evaluated.
            bar_position (int): Used by tqdm to place bars at the right
                position when using multiple Iterators in parallel.
            gpu_mem_growth (bool): tf.Session.ConfigProto parameter
            gpu_mem_fraction (bool): tf.Session.ConfigProto parameter
        '''

        self.model = model
        self.num_epochs = num_epochs

        self.hooks = hooks
        self.hook_freq = hook_freq

        self.bar_pos = bar_position * 2

        self.logger = get_logger(type(self).__name__)

        self.global_step = tf.train.get_or_create_global_step()

        sess_config = tf.ConfigProto()
        if nogpu:
            sess_config.device_count["GPU"] = 0
        sess_config.gpu_options.allow_growth = gpu_mem_growth
        if gpu_mem_fraction is not None:
            sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
        self.session = tf.Session(config=sess_config)

        self.inputs = {}
        self.outputs = {}

    def iterate(self, batch_iterator):
        '''Iterates over the data supplied and feeds it to the model.

        Args:
            batch_iterator (Iterable): Iterable returning training data.
        '''

        inputs, outputs = self.get_interfaces()

        step_ops = self.step_ops()

        with self.session.as_default():
            self.session.run(self.get_init_op())

            pos = self.bar_pos
            for ep in trange(self.num_epochs, desc='Epoch', position=pos):
                self.run_hooks(ep, before=True)

                pos = self.bar_pos + 1
                iterator = tqdm(batch_iterator, desc='Batch', position=pos)
                for bi, batch in enumerate(iterator):

                    fetches = {'global_step': self.global_step,
                               'step_ops': step_ops}
                    feeds = {pl: batch[name] for name, pl in inputs.items()}

                    self.run_hooks(bi,
                                   fetches,
                                   feeds,
                                   batch,
                                   before=True)

                    results = self.session.run(fetches, feed_dict=feeds)

                    self.run_hooks(bi, results=results, before=False)
                    if batch_iterator.is_new_epoch:
                        batch_iterator.reset()
                        break
                self.run_hooks(ep, before=False)

    def run_hooks(self,
                  index,
                  fetches=None,
                  feeds=None,
                  batch=None,
                  results=None,
                  before=True):
        '''Run all hooks and manage their stuff. The passed arguments determine
        which method of the hooks is called.

        Args:
            index (int): Current epoch or batch index. This is not necessarily
                the global training step.
            fetches (list or dict): Fetches for the next session.run call.
            feeds (dict): Feeds for the next session.run call.
            results (same as fetches): Results from the last session.run call.
            before (bool): If not obvious determines if the before_ or after_
                methods of the hooks should be called.

        Return:
            If before:
            same as fetches: Updated fetches.
            dict: Updated feeds
        '''

        is_step = fetches is not None and feeds is not None
        is_step = is_step or results is not None

        if index % self.hook_freq == 0 or not is_step:
            for hook in self.hooks:
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


    def get_interfaces(self):
        '''Get model in- and outputs, as well as iterator ins and outs.
        Make sure to match the ordering of the elements in the data to those
        of the input placeholders.'''

        self.inputs.update(self.model.inputs)
        self.outputs.update(self.model.outputs)

        return self.inputs, self.outputs

    def step_ops(self):
        '''Defines ops that are called at each step.

        Returns:
            The operation run at each step.'''

        raise NotImplementedError()

    def get_init_op(self):
        '''Defines the initialization op. Defaults to
        tf.global_variables_initializer(). Should probably be overwritten by
        the inheriting class.'''

        self.logger.warning('Used default initialization from '
                            'tf.global_variables_initializer()')
        return tf.global_variables_initializer()


class PyHookedModelIterator(object):
    '''Implements a similar interface as the :class:`HookedModelIterator` to
    train framework independent models.'''

    def __init__(self,
                 config,
                 root,
                 model,
                 hook_freq=100,
                 num_epochs=100,
                 hooks=[],
                 bar_position=0,
                 nogpu=False,
                 desc=''):
        '''Constructor.

        Args:
            model (object): Model class. Must have an attribute ``inputs`` for
                placeholders and an attribute ``outputs`` for output tensors.
            num_epochs (int): Number of times to iterate over the data.
            hooks (list): List containing :class:`Hook` instances.
            hook_freq (int): Frequency at which hooks are evaluated.
            bar_position (int): Used by tqdm to place bars at the right
                position when using multiple Iterators in parallel.
        '''
        self.config = config
        self.root = root

        self.model = model
        self.num_epochs = num_epochs

        self.hooks = hooks
        self.hook_freq = hook_freq

        self.bar_pos = bar_position * 2
        self.desc = desc

        self.logger = get_logger(type(self).__name__)

        self._global_step = 0

    def get_global_step(self, *args, **kwargs):
        return self._global_step

    def set_global_step(self, step):
        self._global_step = step

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

    def iterate(self, batch_iterator):
        '''Iterates over the data supplied and feeds it to the model.

        Args:
            batch_iterator (Iterable): Iterable returning training data.
        '''

        try:
            self._iterate(batch_iterator)
        except Exception as e:
            for hook in self.hooks:
                hook.at_exception(e)

            raise e

    def _iterate(self, batch_iterator):
        '''Iterates over the data supplied and feeds it to the model.

        Args:
            batch_iterator (Iterable): Iterable returning training data.
        '''

        step_ops = self.step_ops()

        pos = self.bar_pos
        base = self.desc + ' - ' if self.desc != '' else ''
        desc_e = base + 'Epoch'
        desc_b = base + 'Batch'

        for ep in trange(self.num_epochs,
                         desc=desc_e,
                         position=pos,
                         dynamic_ncols=True):
            self.run_hooks(ep, before=True)

            pos = self.bar_pos + 1
            iterator = tqdm(batch_iterator,
                            desc=desc_b,
                            position=pos,
                            dynamic_ncols=True)
            for bi, batch in enumerate(iterator):
                fetches = {'global_step': self.get_global_step,
                           'step_ops': step_ops}

                feeds = self.make_feeds(batch)

                self.run_hooks(bi,
                               fetches,
                               feeds,
                               batch,
                               before=True)

                results = self.run(fetches, feed_dict=feeds)

                self.run_hooks(bi, results=results, before=False)

                self.increment_global_step()

                if (batch_iterator.is_new_epoch
                        or self.get_global_step() >= self.config.get("num_steps", float('inf'))):
                    self.logger.info('Done with epoch')
                    self.logger.info('is_new_epoch: {}'
                                     .format(batch_iterator.is_new_epoch))
                    gs = self.get_global_step()
                    ns = self.config.get('num_steps', float('inf'))
                    self.logger.info('gs > ns: {} ({} >= {})'
                                     .format(gs >= ns, gs, ns))
                    batch_iterator.reset()
                    break
            self.run_hooks(ep, before=False)

    def run(self, fetches, feed_dict):
        '''Runs all fetch ops and stores the results.

        Args:
            fetches (dict): name: Callable pairs.
            feed_dict (dict): Passed as kwargs to all fetch ops

        Returns:
            dict: name: results pairs.
        '''

        def fn(fetch_fn):
            return fetch_fn(self.model, **feed_dict)

        results = walk(fetches, fn)

        return results

    def run_hooks(self,
                  index,
                  fetches=None,
                  feeds=None,
                  batch=None,
                  results=None,
                  before=True):
        '''Run all hooks and manage their stuff. The passed arguments determine
        which method of the hooks is called.

        Args:
            index (int): Current epoch or batch index. This is not necessarily
                the global training step.
            fetches (list or dict): Fetches for the next session.run call.
            feeds (dict): Feeds for the next session.run call.
            results (same as fetches): Results from the last session.run call.
            before (bool): If not obvious determines if the before_ or after_
                methods of the hooks should be called.

        Return:
            If before:
            same as fetches: Updated fetches.
            dict: Updated feeds
        '''

        is_step = fetches is not None and feeds is not None
        is_step = is_step or results is not None

        condition = self._global_step % self.hook_freq == 0 or not is_step

        if condition:
            for hook in self.hooks:
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
        '''Defines ops that are called at each step.

        Returns:
            The operation run at each step.'''

        raise NotImplementedError()


class TFHookedModelIterator(PyHookedModelIterator):
    def make_feeds(self, batch):
        feeds = {pl: batch[name] for name, pl in self.model.inputs.items() if name in batch}
        return feeds

    def run(self, fetches, feed_dict):
        get_global_step = fetches.pop("global_step")
        results = self.session.run(fetches, feed_dict = feed_dict)
        results["global_step"] = get_global_step()
        return results

    def iterate(self, batch_iterator):
        with self.session.as_default():
            super().iterate(batch_iterator)

    @property
    def session(self):
        # session that is initialized the first time it is needed
        if hasattr(self, "_session"):
            return self._session
        sess_config = tf.ConfigProto()
        if self.config.get("nogpu", False):
            self.logger.info("Hiding GPUs.")
            sess_config.device_count["GPU"] = 0
        sess_config.gpu_options.allow_growth = self.config.get("gpu_mem_growth", False)
        gpu_mem_fraction = self.config.get("gpu_mem_fraction", None)
        if gpu_mem_fraction is not None:
            self.logger.info('Setting GPU MEM Fraction to {}'.format(gpu_mem_fraction))
            sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
        self._session = tf.Session(config=sess_config)
        return self._session

import tensorflow as tf
try:
    import torch
except ModuleNotFoundError:
    print("Warning: Could not import torch.")
import time
import os, re
import numpy as np
from collections import OrderedDict, namedtuple

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.project_manager import ProjectManager
from edflow.util import retrieve


# Values storable as npz
SAVABLES = (np.ndarray, np.int64, int, float, np.float)

P = ProjectManager()


class WaitForCheckpointHook(Hook):
    '''Waits until a new checkpoint is created, then lets the Iterator
    continue.'''

    def __init__(self,
                 checkpoint_root,
                 filter_cond=lambda c: True,
                 interval=5,
                 add_sec=5,
                 callback=None):
        '''Args:
            checkpoint_root (str): Path to look for checkpoints.
            filter_cond (Callable): A function used to filter files, to only
                get the checkpoints that are wanted.
            interval (float): Number of seconds after which to check for a new
                checkpoint again.
            add_sec (float): Number of seconds to wait, after a checkpoint is
                found, to avoid race conditions, if the checkpoint is still
                being written at the time it's meant to be read.
            callback (Callable): Callback called with path of found
                checkpoint.
        '''

        self.root = checkpoint_root
        self.fcond = filter_cond
        self.sleep_interval = interval
        self.additional_wait = add_sec
        self.callback = callback

        self.logger = get_logger(self)

        self.latest_checkpoint = None

    def look(self):
        '''Loop until a new checkpoint is found.'''
        self.logger.info("Waiting for new checkpoint.")
        while True:
            time.sleep(self.sleep_interval)

            latest_checkpoint = get_latest_checkpoint(self.root, self.fcond)
            if latest_checkpoint is not None and latest_checkpoint != self.latest_checkpoint:
                self.latest_checkpoint = latest_checkpoint
                time.sleep(self.additional_wait)
                self.logger.info("Found new checkpoint: {}".format(latest_checkpoint))
                if self.callback is not None:
                    self.callback(latest_checkpoint)
                break

    def before_epoch(self, ep):
        self.look()


def get_latest_checkpoint(checkpoint_root, filter_cond=lambda c: True):
    '''Return path to name of latest checkpoint in checkpoint_root dir.

    Args:
        checkpoint_root (str): Path to where the checkpoints live.
        filter_cond (Callable): A function used to filter files, to only
            get the checkpoints that are wanted.

    Returns:
        str: path of the latest checkpoint. Note that for tensorflow
            checkpoints this is not an existing file, but
            path{.index,.meta,data*} should be
    '''
    ckpt_root = checkpoint_root

    all_files = sorted(os.listdir(ckpt_root))

    # get actual files belonging to checkpoint as well as normalized name as
    # used by tf.Saver.restore
    checkpoint_files = list()
    checkpoint_names = list()
    for f in all_files:
        if f.endswith(".ckpt"):
            checkpoint_files.append(f)
            checkpoint_names.append(f)
        else:
            # check if filename matches tensorflow checkpoint of form
            # name.ckpt-300.index and continue with name.ckpt-300
            match = re.match(r"(.+\.ckpt-[0-9]+)\.index$", f)
            if match:
                checkpoint_files.append(f)
                checkpoint_names.append(match.group(1))

    # convert to list of pairs [name, timestamp of file] to retrieve latest
    checkpoints = []
    for file_, name in zip(checkpoint_files, checkpoint_names):
        file_ = os.path.join(ckpt_root, file_)
        name = os.path.join(ckpt_root, name)
        try:
            mt = os.path.getmtime(file_)
        except FileNotFoundError:
            # checkpoint was deleted, make it infinitely old
            mt = -float("inf")
        checkpoints += [[name, mt]]
    checkpoints = [ckpt for ckpt in checkpoints if filter_cond(ckpt[0])]

    if len(checkpoints) > 0:
        checkpoints = sorted(checkpoints, key=lambda pt: -pt[1])
        latest = checkpoints[0][0]
    else:
        latest = None

    return latest


class RestoreModelHook(Hook):
    '''Restores a TensorFlow model from a checkpoint at each epoch. Can also
    be used as a functor.'''

    def __init__(self,
                 variables,
                 checkpoint_path,
                 filter_cond=lambda c: True,
                 global_step_setter=None):
        '''Args:
            variables (list): tf.Variable to be loaded from the checkpoint.
            checkpoint_path (str): Directory in which the checkpoints are
                stored or explicit checkpoint. Ignored if used as functor.
            filter_cond (Callable): A function used to filter files, to only
                get the checkpoints that are wanted. Ignored if used as
                functor.
            global_step_setter (Callable): Callback to set global_step.
        '''
        self.root = checkpoint_path
        self.fcond = filter_cond
        self.setstep = global_step_setter

        self.logger = get_logger(self)

        self.saver = tf.train.Saver(variables)

    @property
    def session(self):
        if not hasattr(self, "_session"):
            self._session = tf.get_default_session()
        return self._session

    def before_epoch(self, ep):
        #checkpoint = tf.train.latest_checkpoint(self.root)
        checkpoint = get_latest_checkpoint(self.root, self.fcond)
        self(checkpoint)

    def __call__(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        self.logger.info("Restored model from {}".format(checkpoint))
        global_step = self.parse_global_step(checkpoint)
        self.logger.info("Global step: {}".format(global_step))
        if self.setstep is not None:
            self.setstep(global_step)

    @staticmethod
    def parse_global_step(checkpoint):
        global_step = int(checkpoint.rsplit("-", 1)[1])
        return global_step



# Simple renaming for consistency
# Todo: Make the Restore op part of the model (issue #2)
# https://bitbucket.org/jhaux/edflow/issues/2/make-a-general-model-restore-hook
RestoreTFModelHook = RestoreModelHook


# TODO Test filtering for multiple models
# TODO Set Global Step
class RestorePytorchModelHook(Hook):
    '''Restores a PyTorch model from a checkpoint at each epoch. Can also be
    used as a functor.'''

    def __init__(self,
                 model,
                 checkpoint_path,
                 filter_cond=lambda c: True,
                 global_step_setter=None):
        '''Args:
            model (torch.nn.Module): Model to initialize
            checkpoint_path (str): Directory in which the checkpoints are
                stored or explicit checkpoint. Ignored if used as functor.
            filter_cond (Callable): A function used to filter files, to only
                get the checkpoints that are wanted. Ignored if used as
                functor.
            global_step_setter (Callable): Function, that the retrieved global
                step can be passed to.
        '''
        self.root = checkpoint_path
        self.fcond = filter_cond

        self.logger = get_logger(self)

        self.model = model
        self.global_step_setter = global_step_setter

    def before_epoch(self, ep):
        checkpoint = get_latest_checkpoint(self.root, self.fcond)
        self(checkpoint)

    def __call__(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))
        self.logger.info("Restored model from {}".format(checkpoint))

        epoch, step = self.parse_checkpoint(checkpoint)

        if self.global_step_setter is not None:
            self.global_step_setter(step)
        self.logger.info("Epoch: {}, Global step: {}"
                         .format(epoch, step))

    def parse_global_step(self, checkpoint):
        return self.parse_checkpoint(checkpoint)[0]

    @staticmethod
    def parse_checkpoint(checkpoint):
        e_s = os.path.basename(checkpoint).split('.')[0].split('-')
        if len(e_s) > 1:
            epoch = e_s[0]
            step = e_s[1].split('_')[0]
        else:
            epoch = 0
            step = e_s[0].split('_')[0]

        return int(epoch), int(step)


def strenumerate(*args, **kwargs):
    '''Same as enumerate, but yields str(index).'''
    for i, v in enumerate(*args, **kwargs):
        yield str(i), v


def make_iterator(list_or_dict):
    '''Make an iterator that yields key value pairs.'''

    if isinstance(list_or_dict, (dict, OrderedDict)):
        return list_or_dict.items()
    elif isinstance(list_or_dict, (list, tuple)):
        return strenumerate(list_or_dict)
    else:
        msg = 'results must be list or dict but is '
        msg += '{} '.format(type(list_or_dict))
        raise ValueError(msg)


def dict_repr(some_dict, pre='', level=0):
    '''Makes a nice representation of a nested dict.'''

    outstr = ''
    n = 1
    N = len(some_dict)
    for k, v in some_dict.items():
        corner = '├╴ ' if n < N else '└╴ '
        straight = '│  ' if n < N else '   '

        if isinstance(v, dict):
            outstr += pre + '{}{}\n'.format(corner, k)
            outstr += dict_repr(v, pre+straight, level+1)
        else:
            outstr += pre + '{}{}: {}\n'.format(corner, k, type(v))

        n += 1
    return outstr


class CollectorHook(Hook):
    '''Collects data. Supposed to be used as base class.'''

    def __init__(self):
        self.collected_data = {}

        self.logger = get_logger(self, 'latest_eval')

    def after_step(self, step, results):
        self.stack_results(results, self.collected_data)

    def stack_results(self, new_data, all_data):
        '''Given the current collected data append the new results along the
        batch dimension.

        Args:
            new_data (list or dict): data to append.
            all_data (list or dict): data to append to.
        '''

        iterator = make_iterator(new_data)

        for key, value in iterator:
            if isinstance(value, SAVABLES):
                if len(value.shape) == 0:
                    value = np.reshape(value, [1])
                # Leave branch
                if key in all_data:
                    all_data[key] = np.concatenate([all_data[key], value])
                else:
                    all_data[key] = value
            else:
                if key not in all_data:
                    all_data[key] = {}

                self.stack_results(value, all_data[key])


class StoreArraysHook(CollectorHook):
    '''Collects lots of data, stacks them and then stores them.'''

    def __init__(self, save_root):
        '''Collect all outputs of step op and store them as npz.'''
        super().__init__()
        self.root = save_root

    def after_epoch(self, epoch):
        data = self.collected_data
        self.logger.info('Collected Data:\n'+dict_repr(data))

        global_step = data['global_step'][0]

        # Flatten results dictionary for easy storage
        self.flat_dict = {}
        self.flatten_results(data, '', self.flat_dict)
        self.logger.info('Stored Data:\n'+dict_repr(self.flat_dict))

        name = '{:0>6d}_results'.format(global_step)
        name = os.path.join(self.root, name)
        np.savez_compressed(name, **self.flat_dict)

    def flatten_results(self, results, prefix, store_dict):
        '''Recursively walk over the results dictionary and stack the data.

        Args:
            results (dict or list): Containing results.
            prefix (str): Prepended to name when storing.
            store_dict (dict): Flat storage dictionary.
        '''

        iterator = make_iterator(results)

        for name, value in iterator:
            save_name = '{}_{}'.format(prefix, name) if prefix != '' else name
            if isinstance(value, SAVABLES):
                store_dict[save_name] = value
            else:
                self.flatten_results(value, save_name, store_dict)


MetricTuple = namedtuple('MetricTuple', 'input_names output_names metric name')


def test_valid_metrictuple(metric_tuple):
    '''Checks if all inputs are correct.'''
    in_names = metric_tuple.input_names
    out_names = metric_tuple.output_names

    if not isinstance(in_names, dict):
        raise ValueError('input_names must be a dict')
    if not isinstance(out_names, dict):
        raise ValueError('output_names must be a dict')
    if not callable(metric_tuple.metric):
        raise ValueError('metric must be callable')
    if not isinstance(metric_tuple.name, str):
        raise ValueError('name must be a string')

    if not all([isinstance(i, str) for i in in_names.values()]):
        raise ValueError('All entries in input_names must be strings')
    if not all([isinstance(o, str) for o in out_names.values()]):
        raise ValueError('All entries in output_names must be strings')

    identical_names = set(in_names.values()) & set(out_names.values())
    if len(identical_names) > 0:
        raise ValueError('All names must be unique. '
                         'Found {}'.format(identical_names))

    # enough checking already :)


class MetricHook(Hook):
    '''Applies a set of given metrics to the calculated data.'''

    def __init__(self, metrics, save_root, consider_only_first=None):
        '''Args:
            metrics (list): List of ``MetricTuple``s of the form
                ``(input names, output names, metric, name)``.
                - ``input names`` are the keys corresponding to the feeds of
                    interest, e.g. an original image.
                - ``output names`` are the keys corresponding to the values
                    in the results dict.
                - ``metric`` is a ``Callable`` that accepts all inputs and
                    outputs keys as keyword arguments
                - ``name`` is a
                If nested feeds or results are expected the names can be
                passed as "path" like ``'key1_key2'`` returning
                ``dict[key1][key2]``.
            save_root (str): Path to where the results are stored.
            consider_only_first (int): Metric is only evaluated on the first
                `consider_only_first` examples.
        '''

        self.metrics = metrics

        self.root = save_root
        self.logger = get_logger(self, 'latest_eval')

        self.max_step = consider_only_first

        self.storage_dict = {}
        self.metric_results = {}
        for m in metrics:
            test_valid_metrictuple(m)

        self.tb_saver = tf.summary.FileWriter(self.root)

    def before_epoch(self, epoch):
        self.count = 0
        for m in self.metrics:
            self.metric_results[m.name] = []

    def before_step(self, step, fetches, feeds, batch):
        if self.max_step is not None and self.count >= self.max_step:
            return

        for in_names, out_names, metric, m_name in self.metrics:
            self.storage_dict[m_name] = {}
            for kwargs_name, name in in_names.items():
                val = retrieve(name, batch)
                self.storage_dict[m_name][kwargs_name] = val

    def after_step(self, step, results):
        if self.max_step is not None and self.count >= self.max_step:
            return

        for in_names, out_names, metric, m_name in self.metrics:
            for kwargs_name, name in out_names.items():
                val = retrieve(name, results)
                self.storage_dict[m_name][kwargs_name] = val
            m_res = metric(**self.storage_dict[m_name])
            self.metric_results[m_name] += [m_res]

        self.global_step = results['global_step']
        self.count += 1

    def after_epoch(self, epoch):
        self.logger.info("Metrics at epoch {}:".format(epoch))

        mean_results = {}
        for name, result in self.metric_results.items():
            self.logger.info('name: {}'.format(name))
            self.logger.info('result: {}'.format(result))
            results = np.concatenate(result)
            mean = np.mean(results, axis=0)
            var = np.std(results, axis=0)
            mean_results[name] = np.array([mean, var])
            self.logger.info("{}: {} +- {}".format(name, mean, var))

            summary = tf.Summary()
            summary_mean = mean if len(mean.shape) == 0 else mean[0]
            summary.value.add(tag=name, simple_value=summary_mean)
            self.tb_saver.add_summary(summary, self.global_step)
            self.tb_saver.flush()

        name = '{:0>6d}_metrics'.format(self.global_step)
        name = os.path.join(self.root, name)
        np.savez_compressed(name, **mean_results)


def get_checkpoint_files(checkpoint_root):
    '''Return {global_step: [files,...]}.

    Args:
        checkpoint_root (str): Path to where the checkpoints live.
    '''
    ckpt_root = checkpoint_root
    files = []
    checkpoints = []
    global_steps = []
    all_files = os.listdir(ckpt_root)
    for p in all_files:
        p = os.path.join(ckpt_root, p)
        if '.ckpt' in p:
            name, ext = os.path.splitext(p)
            if not ext == ".ckpt":
                normalized = name
                global_step = RestoreTFModelHook.parse_global_step(normalized)
            else:
                normalized = p
                global_step = RestorePytorchModelHook.parse_global_step(normalized)
            files.append(p)
            checkpoints.append(normalized)
            global_steps.append(global_step)
    stepmap = dict()
    for step in np.unique(global_steps):
        stepmap[step] = list()
    for step, file_ in zip(global_steps, files):
        stepmap[step].append(file_)

    return stepmap


class KeepBestCheckpoints(Hook):
    '''Waits until a new checkpoint is created, then lets the Iterator
    continue.'''

    def __init__(self,
                 checkpoint_root,
                 metric_template,
                 metric_key,
                 n_keep = 5):
        '''Args:
            checkpoint_root (str): Path to look for checkpoints.
            metric_template (str): Format string to find metric file.
            metric_key (str): Key to use from metric file.
            n_keep (int): Maximum number of checkpoints to keep.
        '''

        self.root = checkpoint_root
        self.metric_template = metric_template
        self.metric_key = metric_key
        self.n_keep = n_keep

        self.logger = get_logger(self)

    def get_loss(self, step):
        try:
            loss = np.load(self.metric_template.format(step))[self.metric_key][0]
        except FileNotFoundError:
            loss = None
        return loss

    def after_epoch(self, ep):
        checkpoint_files = get_checkpoint_files(self.root)
        steps = sorted(checkpoint_files.keys())
        losses = [self.get_loss(step) for step in steps]
        valid = [i for i in range(len(steps)) if losses[i] is not None]
        steps = [steps[i] for i in valid]
        losses = [losses[i] for i in valid]

        loss_steps = sorted(zip(losses, steps), key = lambda x: x[0])
        steps = [s for _, s in loss_steps]
        remove_steps = steps[self.n_keep:]
        remove_files = list()
        for step in remove_steps:
            remove_files += checkpoint_files[step]
        
        self.logger.info("Removing files:")
        self.logger.info(remove_files)
        for file_ in remove_files:
            os.remove(file_)

        best_ls = loss_steps[0]
        self.logger.info("Current best: {} = {} @ global step {}".format(
            self.metric_key, best_ls[0], best_ls[1]))

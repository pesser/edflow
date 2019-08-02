import time
import os
import re
import pickle
import numpy as np
from collections import OrderedDict, namedtuple

from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger
from edflow.project_manager import ProjectManager
from edflow.util import retrieve


# Values storable as npz
SAVABLES = (np.ndarray, np.int64, int, float, np.float)

P = ProjectManager()


def get_latest_checkpoint(checkpoint_root, filter_cond=lambda c: True):
    """Return path to name of latest checkpoint in checkpoint_root dir.

    Args:
        checkpoint_root (str): Path to where the checkpoints live.
        filter_cond (Callable): A function used to filter files, to only
            get the checkpoints that are wanted.

    Returns:
        str: path of the latest checkpoint. Note that for tensorflow
            checkpoints this is not an existing file, but
            path{.index,.meta,data*} should be
    """
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
        elif f.endswith(".index"):
            # check if filename matches tensorflow index file of form
            # name.ckpt-300.index and continue with name.ckpt-300
            checkpoint_files.append(f)
            checkpoint_names.append(f[: -len(".index")])

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


class WaitForCheckpointHook(Hook):
    """Waits until a new checkpoint is created, then lets the Iterator
    continue."""

    def __init__(
        self,
        checkpoint_root,
        filter_cond=lambda c: True,
        interval=5,
        add_sec=5,
        callback=None,
        eval_all=False,
    ):
        """Args:
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
            eval_all (bool): Accept all instead of just latest checkpoint.
        """

        self.root = checkpoint_root
        self._fcond = filter_cond
        self.sleep_interval = interval
        self.additional_wait = add_sec
        self.callback = callback
        self.eval_all = eval_all

        self.logger = get_logger(self)

        self.known_checkpoints = set()

    def fcond(self, c):
        cond = self._fcond(c)
        if self.eval_all:
            cond = cond and c not in self.known_checkpoints
        return cond

    def look(self):
        """Loop until a new checkpoint is found."""
        self.logger.info("Waiting for new checkpoint.")
        while True:
            latest_checkpoint = get_latest_checkpoint(self.root, self.fcond)
            if (
                latest_checkpoint is not None
                and latest_checkpoint not in self.known_checkpoints
            ):
                self.known_checkpoints.add(latest_checkpoint)
                time.sleep(self.additional_wait)
                self.logger.info("Found new checkpoint: {}".format(latest_checkpoint))
                if self.callback is not None:
                    self.callback(latest_checkpoint)
                break
            time.sleep(self.sleep_interval)

    def before_epoch(self, ep):
        self.look()


def strenumerate(*args, **kwargs):
    """Same as enumerate, but yields str(index)."""
    for i, v in enumerate(*args, **kwargs):
        yield str(i), v


def make_iterator(list_or_dict):
    """Make an iterator that yields key value pairs."""

    if isinstance(list_or_dict, (dict, OrderedDict)):
        return list_or_dict.items()
    elif isinstance(list_or_dict, (list, tuple)):
        return strenumerate(list_or_dict)
    else:
        msg = "results must be list or dict but is "
        msg += "{} ".format(type(list_or_dict))
        raise ValueError(msg)


def dict_repr(some_dict, pre="", level=0):
    """Makes a nice representation of a nested dict."""

    outstr = ""
    n = 1
    N = len(some_dict)
    for k, v in some_dict.items():
        corner = "├╴ " if n < N else "└╴ "
        straight = "│  " if n < N else "   "

        if isinstance(v, dict):
            outstr += pre + "{}{}\n".format(corner, k)
            outstr += dict_repr(v, pre + straight, level + 1)
        else:
            outstr += pre + "{}{}: {}\n".format(corner, k, type(v))

        n += 1
    return outstr


class CollectorHook(Hook):
    """Collects data. Supposed to be used as base class."""

    def __init__(self):
        self.collected_data = {}

        self.logger = get_logger(self, "latest_eval")

    def after_step(self, step, results):
        self.stack_results(results, self.collected_data)

    def stack_results(self, new_data, all_data):
        """Given the current collected data append the new results along the
        batch dimension.

        Args:
            new_data (list or dict): data to append.
            all_data (list or dict): data to append to.
        """

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
    """Collects lots of data, stacks them and then stores them."""

    def __init__(self, save_root):
        """Collect all outputs of step op and store them as npz."""
        super().__init__()
        self.root = save_root

    def after_epoch(self, epoch):
        data = self.collected_data
        self.logger.info("Collected Data:\n" + dict_repr(data))

        global_step = data["global_step"][0]

        # Flatten results dictionary for easy storage
        self.flat_dict = {}
        self.flatten_results(data, "", self.flat_dict)
        self.logger.info("Stored Data:\n" + dict_repr(self.flat_dict))

        name = "{:0>6d}_results".format(global_step)
        name = os.path.join(self.root, name)
        np.savez_compressed(name, **self.flat_dict)

    def flatten_results(self, results, prefix, store_dict):
        """Recursively walk over the results dictionary and stack the data.

        Args:
            results (dict or list): Containing results.
            prefix (str): Prepended to name when storing.
            store_dict (dict): Flat storage dictionary.
        """

        iterator = make_iterator(results)

        for name, value in iterator:
            save_name = "{}_{}".format(prefix, name) if prefix != "" else name
            if isinstance(value, SAVABLES):
                store_dict[save_name] = value
            else:
                self.flatten_results(value, save_name, store_dict)


MetricTuple = namedtuple("MetricTuple", "input_names output_names metric name")


def test_valid_metrictuple(metric_tuple):
    """Checks if all inputs are correct."""
    in_names = metric_tuple.input_names
    out_names = metric_tuple.output_names

    if not isinstance(in_names, dict):
        raise ValueError("input_names must be a dict")
    if not isinstance(out_names, dict):
        raise ValueError("output_names must be a dict")
    if not callable(metric_tuple.metric):
        raise ValueError("metric must be callable")
    if not isinstance(metric_tuple.name, str):
        raise ValueError("name must be a string")

    if not all([isinstance(i, str) for i in in_names.values()]):
        raise ValueError("All entries in input_names must be strings")
    if not all([isinstance(o, str) for o in out_names.values()]):
        raise ValueError("All entries in output_names must be strings")

    identical_names = set(in_names.values()) & set(out_names.values())
    if len(identical_names) > 0:
        raise ValueError(
            "All names must be unique. " "Found {}".format(identical_names)
        )

    # enough checking already :)


def torch_parse_global_step(checkpoint):
    e_s = os.path.basename(checkpoint).split(".")[0].split("-")
    if len(e_s) > 1:
        epoch = e_s[0]
        step = e_s[1].split("_")[0]
    else:
        epoch = 0
        step = e_s[0].split("_")[0]

    epoch, step = int(epoch), int(step)
    return step


def tf_parse_global_step(checkpoint):
    global_step = int(checkpoint.rsplit("-", 1)[1])
    return global_step


def get_checkpoint_files(checkpoint_root):
    """Return {global_step: [files,...]}.

    Args:
        checkpoint_root (str): Path to where the checkpoints live.
    """
    ckpt_root = checkpoint_root
    files = []
    checkpoints = []
    global_steps = []
    all_files = os.listdir(ckpt_root)
    for p in all_files:
        p = os.path.join(ckpt_root, p)
        if ".ckpt" in p:
            name, ext = os.path.splitext(p)
            if not ext == ".ckpt":
                normalized = name
                global_step = tf_parse_global_step(normalized)
            else:
                normalized = p
                global_step = torch_parse_global_step(normalized)
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
    """Tries to find a metric for all checkpoints and keeps the n_keep best
    checkpoints and the latest checkpoint."""

    def __init__(
        self,
        checkpoint_root,
        metric_template,
        metric_key,
        n_keep=5,
        lower_is_better=True,
    ):
        """Args:
            checkpoint_root (str): Path to look for checkpoints.
            metric_template (str): Format string to find metric file.
            metric_key (str): Key to use from metric file.
            n_keep (int): Maximum number of checkpoints to keep.
        """

        self.root = checkpoint_root
        self.metric_template = metric_template
        self.metric_key = metric_key
        self.n_keep = n_keep
        self.lower_is_better = lower_is_better

        self.logger = get_logger(self)

    def get_loss(self, step):
        path = self.metric_template.format(step)
        try:
            if path.endswith(".npz"):
                loss = np.load(path)[self.metric_key][0]
            else:
                with open(path, "rb") as f:
                    loss = pickle.load(f)[self.metric_key][0]
            if not self.lower_is_better:
                loss = -1.0 * loss
        except FileNotFoundError:
            self.logger.debug("Could not find {}".format(path))
            loss = None
        return loss

    def after_epoch(self, ep):
        checkpoint_files = get_checkpoint_files(self.root)
        steps = sorted(checkpoint_files.keys())
        losses = [self.get_loss(step) for step in steps]
        valid = [i for i in range(len(steps)) if losses[i] is not None]
        steps = [steps[i] for i in valid]
        losses = [losses[i] for i in valid]

        latest_step = max(steps)

        loss_steps = sorted(zip(losses, steps), key=lambda x: x[0])
        steps = [s for _, s in loss_steps]
        remove_steps = steps[self.n_keep :]
        remove_steps = [step for step in remove_steps if not step == latest_step]
        remove_files = list()
        for step in remove_steps:
            remove_files += checkpoint_files[step]

        self.logger.info("Removing files:")
        self.logger.info(remove_files)
        for file_ in remove_files:
            os.remove(file_)

        best_ls = loss_steps[0]
        self.logger.info(
            "Current best: {} = {} @ global step {}".format(
                self.metric_key, best_ls[0], best_ls[1]
            )
        )
        no_improvement_since = latest_step - best_ls[1]
        if no_improvement_since > 0:
            self.logger.info(
                "No improvement since {} global steps.".format(no_improvement_since)
            )

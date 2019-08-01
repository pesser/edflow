"""To produce consistent results we adopt the following pipeline:

Step 1: Evaluate model on a test dataset and write out all data of interest:
    - generated image
    - latent representations

Step 2: Load the generated data in a Datafolder using the EvalDataset

Step 3: Pass both the test Dataset and the Datafolder to the evaluation scripts

Sometime in the future:
(Step 4): Generate a report:
    - latex tables
    - paths to videos
    - plots

# Usage

The pipeline is easily setup: In you Iterator (Trainer or Evaluator) add
the EvalHook and as many callbacks as you like. You can also pass no callback
at all.

.. codeblock:: python
    from edflow.eval.pipeline import EvalHook

    from my_project.callbacks import my_callback

    class MyIterator(PyHookedModelIterator):
        def __init__(self, config, root, model, **kwargs):

            self.model = model

            self.hooks += [EvalHook(self.dataset,
                                    callbacks=[my_callback],
                                    meta=config,
                                    step_getter=self.get_global_step)]

        def eval_op(self, inputs):
            return {'generated': self.model(inputs)}

        self.step_ops(self):
            return self.eval_op


Next you run your evaluation on your data using your favourite edflow command.

.. codeblock:: bash
    edflow -n myexperiment -e the_config.yaml -p path_to_project

This will create a new evaluation folder inside your project's eval directory.
Inside this folder everything returned by your step ops is stored. In the case
above this would mean your outputs would be stored as
``generated:index.something``. But you don't need to concern yourself with
that, as the outputs can now be loaded using the :class:`EvalDataFolder`.

All you need to do is pass the EvalDataFolder the root folder in which the data
has been saved, which is the folder where you can find the
``model_outputs.csv``. Now you have all the generated data easily usable at
hand. The indices of the data in the EvalDataFolder correspond to the indices
of the data in the dataset, which was used to create the model outputs. So
you can directly compare inputs, targets etc, with the outputs of your model!

If you specified a callback, this all happens automatically. Each callback
receives 4 parameters: The ``root``, where the data lives, the two datasets
``data_in``, which was fed into the model and ``data_out``, which was generated
by the model, and the ``config``.

Should you want to run evaluations on the generated data after it has been
generated, you can run the ``edeval`` command while specifying the path
to the model outputs csv and the callbacks you want to run.

.. codeblock:: bash
    edeval -c path/to/model_outputs.csv -cb callback1 callback2
"""

import os
import numpy as np
import pandas as pd  # storing model output paths
import yaml  # metadata
from PIL import Image
import inspect
import re

from edflow.data.util import adjust_support
from edflow.util import walk
from edflow.data.dataset import DatasetMixin, CsvDataset, ProcessedDataset
from edflow.project_manager import ProjectManager as P
from edflow.hooks.hook import Hook
from edflow.custom_logging import get_logger


LOADABLE_EXTS = ["png", "npy", "txt"]


class EvalHook(Hook):
    """Stores all outputs in a reusable fashion."""

    def __init__(
        self,
        dataset,
        sub_dir_keys=[],
        label_keys=[],
        callbacks=[],
        meta=None,
        step_getter=None,
    ):
        """
        Parameters
        ==========
            dataset : DatasetMixin
                The Dataset used for creating the new data.
            sub_dir_keys : list(str)
                Keys found in :attr:`example`, which will
                be used to make a subdirectory for the stored example.
                Subdirectories are made in a nested fashion in the order of the
                list. The keys will be removed from the example dict and not be
                stored explicitly.
            label_keys : list(str)
                Keys found in :attr:`example`, which will be stored in one
                large array and later loaded as labels.
            callbacks : list(Callable)
                Called at the end of the epoch. Must
                accept root as argument as well as the generating dataset and
                the generated dataset (in that order).
            meta : object, dict
                An object containing metadata. Must be dumpable by
                ``yaml``. Usually the ``edflow`` config.
            step_getter : Callable
                Function which returns the global step as ``int``.
        """
        self.logger = get_logger(self)

        self.cbacks = callbacks
        self.logger.info("{}".format(self.cbacks))
        self.cb_names = [inspect.getmodule(c).__name__ for c in self.cbacks]
        self.sdks = sub_dir_keys
        self.lks = label_keys
        self.data_in = dataset

        self.meta = meta

        self.gs = step_getter

    def before_epoch(self, epoch):
        self.data_frame = None
        self.root = os.path.join(P.latest_eval, str(self.gs()))
        self.save_root = os.path.join(self.root, "model_outputs")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.save_root, exist_ok=True)

        self.label_arrs = None

    def before_step(self, step, fetches, feeds, batch):
        """Get dataset indices from batch."""
        self.idxs = np.array(batch["index_"], dtype=int)

    def after_step(self, step, last_results):
        # Attention -> This does not work with nested keys!
        # Fix it in post :)
        label_vals = {k: _delget(last_results["step_ops"], k) for k in self.lks}

        if self.label_arrs is None:
            self.label_arrs = {}
            for k in self.lks:
                example = label_vals[k][0]
                ex_shape = list(np.shape(example))
                shape = [len(self.data_in)] + ex_shape
                s = "x".join([str(s) for s in shape])
                dtype = d = example.dtype

                savepath = os.path.join(
                    self.save_root, "{}-*-{}-*-{}.npy".format(k, s, d)
                )
                memmap = np.memmap(savepath, shape=tuple(shape), mode="w+", dtype=dtype)
                self.label_arrs[k] = memmap

        idxs = self.idxs  # indices collected before_step

        for k in self.lks:
            # Can the inner loop be made a fancy indexing assign?
            for i, idx in enumerate(idxs):
                self.label_arrs[k][idx] = label_vals[k][i]

        path_dicts = save_output(self.save_root, last_results, idxs, self.sdks)

        if self.data_frame is None:
            columns = sorted(path_dicts[list(path_dicts.keys())[0]])
            if len(columns) == 0:
                # No load heavy logs written out
                pass
            else:
                self.data_frame = pd.DataFrame(columns=columns)

        if self.data_frame is not None:
            for idx, path_dict in path_dicts.items():
                self.data_frame.loc[idx] = path_dict

    def at_exception(self, *args, **kwargs):
        self.save_csv()

    def after_epoch(self, epoch):
        """Save csv for reuse and then start the evaluation callbacks"""
        self.save_csv()

        data_out = EvalDataFolder(self.root)

        for n, cb in zip(self.cb_names, self.cbacks):
            cb_name = "CB: {}".format(n)
            cb_name = "{a}\n{c}\n{a}".format(a="=" * len(cb_name), c=cb_name)
            self.logger.info(cb_name)
            cb(self.root, self.data_in, data_out, self.meta)

    def save_csv(self):
        csv_path = os.path.join(self.root, "model_output.csv")

        if self.data_frame is not None:
            self.data_frame = self.data_frame.sort_index()
            self.data_frame.to_csv(csv_path, index=False)
        else:
            with open(csv_path, "w+") as csv_file:
                csv_file.write("")

        add_meta_data(csv_path, self.meta)

        this_script = os.path.dirname(__file__)
        if self.cb_names:
            cbs = " ".join(self.cb_names)
        else:
            cbs = "<your callback>"

        self.logger.info("MODEL_OUPUT_CSV {}".format(csv_path))
        self.logger.info(
            "All data has been produced. You can now also run all"
            + " callbacks using the following command:\n"
            + "edeval -c {} -cb {}".format(csv_path, cbs)
        )


class EvalDataFolder(DatasetMixin):
    def __init__(self, root, show_bar=False):
        er = EvalReader(root)

        if "model_output.csv" not in root:
            csv_path = os.path.join(root, "model_output.csv")
        else:
            csv_path = root
            root = os.path.dirname(root)

        self.labels = load_labels(os.path.join(root, "model_outputs"))

        # Capture the case that only labels have been written out
        try:
            csv_data = CsvDataset(csv_path, comment="#")
            self.data = ProcessedDataset(csv_data, er)
        except pd.errors.EmptyDataError as e:
            print(e)
            exemplar_labels = self.labels[sorted(self.labels.keys())[0]]
            self.data = EmptyDataset(len(exemplar_labels), self.labels)


class EmptyDataset(DatasetMixin):
    def __init__(self, n_ex, labels={}):
        self.len = n_ex
        self.labels = labels

    def get_example(self, idx):
        return {"content": None}

    def __len__(self):
        return self.len


def load_labels(root):
    regex = re.compile(".*-\*-.*-\*-.*\.npy")

    files = os.listdir(root)
    label_files = [f for f in files if regex.match(f) is not None]

    labels = {}
    for f in label_files:
        f_ = f.strip(".npy")
        key, shape, dtype = f_.split("-*-")
        shape = tuple([int(s) for s in shape.split("x")])

        path = os.path.join(root, f)

        labels[key] = np.memmap(path, mode="r", shape=shape, dtype=dtype)

    return labels


def save_output(root, example, index, sub_dir_keys=[]):
    """Saves the ouput of some model contained in ``example`` in a reusable
    manner.

    Args:
        root (str): Storage directory
        example (dict): name: datum pairs of outputs.
        index (list(int)): dataset index corresponding to example.
        sub_dir_keys (list(str)): Keys found in :attr:`example`, which will be
            used to make a subirectory for the stored example. Subdirectories
            are made in a nested fashion in the order of the list. The keys
            will be removed from the example dict and not be stored.
            Directories are name ``key:val`` to be able to completely recover
            the keys.

    Returns:
        dict: Name: path pairs of the saved ouputs.

    .. WARNING::
        Make sure the values behind the ``sub_dir_keys`` are compatible with
        the file system you are saving data on.
    """

    example = example["step_ops"]

    sub_dirs = [""] * len(index)
    for subk in sub_dir_keys:
        sub_vals = _delget(example, subk)
        for i, sub_val in enumerate(sub_vals):
            name = "{}:{}".format(subk, sub_val)
            name = name.replace("/", "--")
            sub_dirs[i] = os.path.join(sub_dirs[i], name)

    roots = [os.path.join(root, sub_dir) for sub_dir in sub_dirs]
    for r in roots:
        os.makedirs(r, exist_ok=True)

    roots += [root]

    path_dicts = {}
    for i, [idx, root] in enumerate(zip(index, roots)):
        path_dict = {}
        for n, e in example.items():
            savename = "{}_{:0>6d}.{{}}".format(n, idx)
            path = os.path.join(root, savename)

            path = save_example(path, e[i])

            path_dict[n + "_path"] = path
        path_dicts[idx] = path_dict

    return path_dicts


def add_meta_data(path_to_csv, metadata):
    """Prepends kwargs of interest to a csv file as comments (`#`)"""

    meta_string = yaml.dump(metadata)

    commented_string = ""
    for line in meta_string.split("\n"):
        line = "# {}\n".format(line)
        commented_string += line

    with open(path_to_csv, "r+") as csv_file:
        content = csv_file.read()

    content = commented_string + content

    with open(path_to_csv, "w+") as csv_file:
        csv_file.write(content)


def read_meta_data(path_to_csv):
    """This functions assumes that the first lines of the csv are the commented
    output of a ``yaml.dump()`` call and loads its contents for further use.
    """

    with open(path_to_csv, "r") as csv_file:
        yaml_string = ""
        for line in csv_file.readlines():
            if "# " in line:
                yaml_string += line[2:] + "\n"
            else:
                break

    meta_data = yaml.load(yaml_string)

    return meta_data


def _delget(d, k):
    v = d[k]
    del d[k]
    return v


def save_example(savepath, datum):
    """Manages the writing process of a single datum: (1) Determine type,
    (2) Choos saver, (3) save.

    Args:
        savepath (str): Where to save. Must end with `.{}` to put in the
            file ending via `.format()`.
        datum (object): Some python object to save.
    """

    saver, ending = determine_saver(datum)

    savepath = savepath.format(ending)

    saver(savepath, datum)

    return savepath


def determine_saver(py_obj):
    """Applies some heuristics to save an object."""

    if isinstance(py_obj, np.ndarray):
        if isimage(py_obj):
            return image_saver, "png"
        else:
            return np_saver, "npy"

    elif isinstance(py_obj, str):
        return txt_saver, "txt"

    else:
        raise NotImplementedError(
            "There currently is not saver heuristic " + "for {}".format(type(py_obj))
        )


def load_by_heuristic(path):
    """Chooses a loader based on the file ending."""

    name, ext = os.path.splitext(path)

    if ext == ".png":
        return image_loader(path)
    elif ext == ".npy":
        return np_loader(path)
    elif ext == ".txt":
        return txt_loader(path)
    else:
        raise ValueError(
            "Cannot load file with extenstion `{}` at {}".format(ext, path)
        )


def decompose_name(name):
    try:
        splits = name.split("_")
        rest = splits[-1]
        datum_name = "_".join(splits[:-1])
        index, ending = rest.split(".")

        return int(index), datum_name, ending
    except Exception as e:
        print("Faulty name:", name)
        raise e


def is_loadable(filename):
    if "." in filename:
        name, ext = filename.split(".")
        if ext not in LOADABLE_EXTS:
            return False
        elif name.count("_") != 1:
            return False
        else:
            return True
    else:
        return False


class EvalLabeler(object):
    def __init__(self, root):
        self.root = root

        self.visited = []

    def __call__(self, path):
        """Adds the labels ``paths``, ``kind``, ``index_``, ``datum_root``"""

        ret_dict = {}

        rel = os.path.relpath(path, self.root)
        folder_structure, filename = os.path.split(rel)

        if not is_loadable(filename):
            return None

        # Do that first, the pass if index already in there.
        # Now get all the files with the same index
        index, datum_name, ending = decompose_name(filename)

        if index not in self.visited:
            self.visited += [index]
            # Get the sbfolder key val pairs
            for kv in folder_structure.split("/"):
                key, val = kv.split(":")
                val = val.replace("--", "/")
                ret_dict[key] = val

            # get all files with this index
            # We know all files must be in the folder, as we must assume, that
            # index_ is a unique key. Otherwise this whole system does not
            # work.

            all_files = os.listdir(os.path.join(self.root, folder_structure))

            def idx_filter(fname):
                n, e = os.path.splitext(fname)
                if "." in e:
                    has_nice_ending = e[1:] in LOADABLE_EXTS
                else:
                    return False
                return "{:0>6d}".format(index) in fname and has_nice_ending

            of_interest = filter(idx_filter, all_files)

            # Remember the filenames -> Not really neccessary, but makes
            # reading code cleaner (hopefully).
            for filename in of_interest:
                path = os.path.join(self.root, folder_structure, filename)
                _, datum_name, ending = decompose_name(filename)
                ret_dict["{}_{}".format(datum_name, "path")] = path

            ret_dict["datum_root"] = os.path.join(self.root, folder_structure)

            ret_dict["save_index_"] = index

            return ret_dict


class EvalReader(object):
    def __init__(self, root):
        self.root = root

    def __call__(self, **kwargs):
        """Works only with non legacy DataFolder!"""

        ret_dict = {}

        if "file_path_" in kwargs:
            del kwargs["file_path_"]

        path_keys = [f for f in list(kwargs.keys()) if "path" in f]

        for k in path_keys:
            path = kwargs[k]
            name = "_".join(os.path.basename(k).split("_")[:-1])

            ret_dict[name] = load_by_heuristic(path)

        return ret_dict


def isimage(np_arr):
    shape = np_arr.shape
    return len(shape) == 3 and shape[-1] in [1, 3, 4]


def image_saver(savepath, image):
    im_adjust = adjust_support(image, "0->255", clip=True)

    mode = "RGB" if im_adjust.shape[-1] in [1, 3] else "RGBA"

    im = Image.fromarray(im_adjust, mode)

    im.save(savepath)


def image_loader(path):
    im = np.array(Image.open(path))
    return im


def np_saver(savepath, np_arr):
    np.save(savepath, np_arr)


def np_loader(path):
    return np.load(path)


def txt_saver(savepath, string):
    with open(savepath, "a+") as f:
        f.write(string + "\n")


def txt_loader(path):
    with open(path, "r") as f:
        data = f.read()

    return data


def standalone_eval_csv_file(path_to_csv, callbacks):
    """Runs all given callbacks on the data in the :class:`EvalDataFolder`
    constructed from the given csv.abs

    Arguments:
        path_to_csv (str): Path to the csv file.
        callbacks (list(str or Callable)): Import commands used to construct
        the functions applied to the Data extracted from :attr:`path_to_csv`.

    Returns:
        The collected outputs of the callbacks.
    """

    import importlib
    from edflow.main import get_implementations_from_config

    import sys

    sys.path.append(os.getcwd())  # convenience: load implementations from cwd

    out_data = EvalDataFolder(path_to_csv)

    config = read_meta_data(path_to_csv)

    dataset_str = config["dataset"]
    impl = get_implementations_from_config(config, ["dataset"])
    in_data = impl["dataset"](config)

    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    outputs = []
    for cb in callbacks:
        if isinstance(cb, str):
            module = ".".join(cb.split(".")[:-1])
            module = importlib.import_module(module)

            cb = getattr(module, cb.split(".")[-1])

        outputs += [cb(os.path.dirname(path_to_csv), in_data, out_data, config)]

    return outputs


if __name__ == "__main__":
    import argparse

    A = argparse.ArgumentParser()

    A.add_argument(
        "-c",
        "--csv",
        default="model_output.csv",
        type=str,
        help="path to a csv-file created by the EvalHook containing"
        " samples generated by a model.",
    )
    A.add_argument(
        "-cb",
        "--callback",
        type=str,
        nargs="*",
        help="Import string to the callback functions used for the "
        "standalone evaluation.",
    )

    args = A.parse_args()

    standalone_eval_csv_file(args.csv, args.callback)

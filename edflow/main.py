import argparse
import importlib
import os
import yaml
import math
import datetime

# ignore broken pipe errors: https://www.quora.com/How-can-you-avoid-a-broken-pipe-error-on-Python
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)

import multiprocessing as mp
import traceback

from edflow.custom_logging import init_project, get_logger, LogSingleton
from edflow.project_manager import ProjectManager as P


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def get_impl(config, name):
    impl = config[name]
    module, cls = impl.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def get_implementations_from_config(config, names):
    implementations = dict((name, get_impl(config, name)) for name in names)
    return implementations


def traceable_process(fn, args, job_queue, idx):
    try:
        fn(*args)
    except Exception as e:
        trace = traceback.format_exc()
        exc = Exception(trace)
        if job_queue is not None:
            job_queue.put([idx, exc, trace])
        else:
            raise exc
    else:
        job_queue.put([idx, "Done", None])
    finally:
        job_queue.close()


def traceable_function(method, ignores=None):
    def tmethod(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            if ignores is not None:
                if not isinstance(e, tuple(ignores)):
                    traceback.print_exc()
                    raise e

    return tmethod


def traceable_method(ignores=None):
    def decorator(method):
        return traceable_function(method, ignores=ignores)

    return decorator


def _save_config(config, prefix="config"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    fname = prefix + "_" + now + ".yaml"
    path = os.path.join(P.configs, fname)
    with open(path, "w") as f:
        f.write(yaml.dump(config))
    return path


def train(args, job_queue, idx):
    traceable_process(_train, args, job_queue, idx)


def test(args, job_queue, idx):
    traceable_process(_test, args, job_queue, idx)


def _train(config, root, checkpoint=None, retrain=False):
    """Run training. Loads model, iterator and dataset according to config."""
    from edflow.iterators.batches import make_batches

    LogSingleton().set_default("train")
    logger = get_logger("train")
    logger.info("Starting Training.")

    implementations = get_implementations_from_config(
        config, ["model", "iterator", "dataset"]
    )

    # fork early to avoid taking all the crap into forked processes
    logger.info("Instantiating dataset.")
    dataset = implementations["dataset"](config=config)
    logger.info("Number of training samples: {}".format(len(dataset)))
    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    with make_batches(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
    ) as batches:
        # get them going
        logger.info("Warm up batches.")
        next(batches)
        batches.reset()
        logger.info("Reset batches.")

        if "num_steps" in config:
            # set number of epochs to perform at least num_steps steps
            steps_per_epoch = len(dataset) / config["batch_size"]
            num_epochs = config["num_steps"] / steps_per_epoch
            config["num_epochs"] = math.ceil(num_epochs)
        else:
            steps_per_epoch = len(dataset) / config["batch_size"]
            num_steps = config["num_epochs"] * steps_per_epoch
            config["num_steps"] = math.ceil(num_steps)

        logger.info("Instantiating model.")
        Model = implementations["model"](config)
        if not "hook_freq" in config:
            config["hook_freq"] = 1
        compat_kwargs = dict(
            hook_freq=config["hook_freq"], num_epochs=config["num_epochs"]
        )
        logger.info("Instantiating iterator.")
        Trainer = implementations["iterator"](
            config, root, Model, dataset=dataset, **compat_kwargs
        )

        logger.info("Initializing model.")
        if checkpoint is not None:
            Trainer.initialize(checkpoint_path=checkpoint)
        else:
            Trainer.initialize()

        if retrain:
            Trainer.reset_global_step()

        # save current config
        logger.info("Starting Training with config:\n{}".format(yaml.dump(config)))
        cpath = _save_config(config, prefix="train")
        logger.info("Saved config at {}".format(cpath))

        logger.info("Iterating.")
        Trainer.iterate(batches)


def _test(config, root, checkpoint=None, nogpu=False, bar_position=0):
    """Run tests. Loads model, iterator and dataset from config."""
    from edflow.iterators.batches import make_batches

    LogSingleton().set_default("latest_eval")
    logger = get_logger("test")
    logger.info("Starting Evaluation.")

    if "test_batch_size" in config:
        config["batch_size"] = config["test_batch_size"]
    if "test_mode" not in config:
        config["test_mode"] = True

    implementations = get_implementations_from_config(
        config, ["model", "iterator", "dataset"]
    )

    dataset = implementations["dataset"](config=config)
    logger.info("Number of testing samples: {}".format(len(dataset)))
    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    batches = make_batches(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
    )
    # get going
    next(batches)
    batches.reset()

    logger.info("Initializing model.")
    Model = implementations["model"](config)

    config["hook_freq"] = 1
    config["num_epochs"] = 1
    config["nogpu"] = nogpu
    compat_kwargs = dict(
        hook_freq=config["hook_freq"],
        bar_position=bar_position,
        nogpu=config["nogpu"],
        num_epochs=config["num_epochs"],
    )
    Evaluator = implementations["iterator"](
        config, root, Model, dataset=dataset, **compat_kwargs
    )

    logger.info("Initializing model.")
    if checkpoint is not None:
        Evaluator.initialize(checkpoint_path=checkpoint)
    else:
        Evaluator.initialize()

    # save current config
    logger.info("Starting Evaluation with config:\n{}".format(yaml.dump(config)))
    prefix = "eval"
    if bar_position > 0:
        prefix = prefix + str(bar_position)
    cpath = _save_config(config, prefix=prefix)
    logger.info("Saved config at {}".format(cpath))

    logger.info("Iterating")
    while True:
        Evaluator.iterate(batches)
        if not config.get("eval_forever", False):
            break

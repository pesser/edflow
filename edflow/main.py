import argparse
import importlib
import os
import yaml
import math
# ignore broken pipe errors: https://www.quora.com/How-can-you-avoid-a-broken-pipe-error-on-Python
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

import multiprocessing as mp
import traceback

from edflow.custom_logging import init_project, get_logger, LogSingleton


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
            job_queue.close()
        else:
            raise exc

    job_queue.put([idx, "Done", None])


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


def train(args, job_queue, idx):
    traceable_process(_train, args, job_queue, idx)


def test(args, job_queue, idx):
    traceable_process(_test, args, job_queue, idx)


def _train(config, root, checkpoint=None, retrain=False):
    """Run training. Loads model, iterator and dataset according to config."""
    from edflow.iterators.batches import make_batches

    LogSingleton().set_default("train")
    logger = get_logger("train")
    logger.info("Starting Training with config:")
    logger.info(config)

    implementations = get_implementations_from_config(
        config, ["model", "iterator", "dataset"]
    )

    # fork early to avoid taking all the crap into forked processes
    logger.info("Instantiating dataset.")
    dataset = implementations["dataset"](config=config)
    logger.info("Number of training samples: {}".format(len(dataset)))
    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    batches = make_batches(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
    )
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
    compat_kwargs = dict(hook_freq=config["hook_freq"], num_epochs=config["num_epochs"])
    logger.info("Instantiating iterator.")
    Trainer = implementations["iterator"](config, root, Model, **compat_kwargs)

    logger.info("Initializing model.")
    if checkpoint is not None:
        Trainer.initialize(checkpoint_path=checkpoint)
    else:
        Trainer.initialize()

    if retrain:
        Trainer.reset_global_step()

    logger.info("Iterating.")
    Trainer.iterate(batches)


def _test(config, root, nogpu=False, bar_position=0):
    """Run tests. Loads model, iterator and dataset from config."""
    from edflow.iterators.batches import make_batches

    LogSingleton().set_default("latest_eval")
    logger = get_logger("test")
    logger.info("Starting Evaluation with config")
    logger.info(config)

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
    # currently initialize is not called here because we assume that checkpoint
    # restoring is handled by RestoreCheckpointHook
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
    HBU_Evaluator = implementations["iterator"](config, root, Model, **compat_kwargs)

    logger.info("Iterating")
    while True:
        HBU_Evaluator.iterate(batches)
        if not config.get("eval_forever", False):
            break


def main(opt):
    with open(opt.config) as f:
        config = yaml.load(f)

    P = init_project("logs")
    logger = get_logger("main")
    logger.info(opt)
    logger.info(yaml.dump(config))
    logger.info(P)

    if opt.noeval:
        train(config, P.train, opt.checkpoint, opt.retrain)
    else:
        train_process = mp.Process(
            target=train, args=(config, P.train, opt.checkpoint, opt.retrain)
        )
        test_process = mp.Process(target=test, args=(config, P.latest_eval, True))

        processes = [train_process, test_process]

        try:
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("Terminating all processes")
            for p in processes:
                p.terminate()
        finally:
            logger.info("Finished")


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", help="path to checkpoint to restore")
    parser.add_argument(
        "--noeval", action="store_true", default=False, help="only run training"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="reset global_step to zero",
    )

    opt = parser.parse_args()
    main(opt)

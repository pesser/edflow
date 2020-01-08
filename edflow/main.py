import argparse
import importlib
import os
import yaml
import math
import datetime

from edflow.custom_logging import log, run


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


def _save_config(config, prefix="config"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fname = prefix + "_" + now + ".yaml"
    path = os.path.join(run.configs, fname)
    with open(path, "w") as f:
        f.write(yaml.dump(config))
    return path


def train(config, root, checkpoint=None, retrain=False, debug=False):
    """Run training. Loads model, iterator and dataset according to config."""
    from edflow.iterators.batches import make_batches

    log.set_log_target("train")
    logger = log.get_logger("train")
    logger.info("Starting Training.")

    implementations = get_implementations_from_config(
        config, ["model", "iterator", "dataset"]
    )
    logger.info("Instantiating dataset.")
    dataset = implementations["dataset"](config=config)
    dataset.expand = True
    logger.info("Number of training samples: {}".format(len(dataset)))
    if debug:
        logger.info("Monkey patching dataset __len__")
        type(dataset).__len__ = lambda self: 100
    if "validation_dataset" in config:
        use_validation_dataset = True
        implementations["validation_dataset"] = get_obj_from_str(
            config["validation_dataset"]
        )
        logger.info("Instantiating validation dataset.")
        validation_dataset = implementations["validation_dataset"](config=config)
        logger.info("Number of validation samples: {}".format(len(validation_dataset)))
        if debug:
            logger.info("Monkey patching validation_dataset __len__")
            type(validation_dataset).__len__ = lambda self: 100
    else:
        use_validation_dataset = False

    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    batches = make_batches(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
        error_on_timeout=config.get("error_on_timeout", False),
    )
    if use_validation_dataset:
        validation_batches = make_batches(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            n_processes=n_processes,
            n_prefetch=n_prefetch,
            error_on_timeout=config.get("error_on_timeout", False),
        )
    else:
        validation_batches = None
    try:
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
        Trainer.iterate(batches, validation_batches)
    finally:
        batches.finalize()
        if use_validation_dataset:
            validation_batches.finalize()


def test(config, root, checkpoint=None, nogpu=False, bar_position=0, debug=False):
    """Run tests. Loads model, iterator and dataset from config."""
    from edflow.iterators.batches import make_batches

    log.set_log_target("latest_eval")
    logger = log.get_logger("test")
    logger.info("Starting Evaluation.")

    if "test_batch_size" in config:
        config["batch_size"] = config["test_batch_size"]
    if "test_mode" not in config:
        config["test_mode"] = True

    implementations = get_implementations_from_config(
        config, ["model", "iterator", "dataset"]
    )

    dataset = implementations["dataset"](config=config)
    dataset.expand = True
    logger.info("Number of testing samples: {}".format(len(dataset)))
    if debug:
        logger.info("Monkey patching dataset __len__")
        type(dataset).__len__ = lambda self: 100
        logger.info("Number of testing samples: {}".format(len(dataset)))
    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    batches = make_batches(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        n_processes=n_processes,
        n_prefetch=n_prefetch,
        error_on_timeout=config.get("error_on_timeout", False),
    )

    try:
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
    finally:
        batches.finalize()

import argparse
import importlib
import os
import yaml
import math
import datetime

from edflow.custom_logging import log, run
from edflow.util import get_obj_from_str, retrieve, set_value


def _save_config(config, prefix="config"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fname = prefix + "_" + now + ".yaml"
    path = os.path.join(run.configs, fname)
    with open(path, "w") as f:
        f.write(yaml.dump(config))
    return path


# TODO: DRY --- train and test are almost the same


def train(config, root, checkpoint=None, retrain=False, debug=False):
    """Run training. Loads model, iterator and dataset according to config."""
    from edflow.iterators.batches import make_batches

    # disable integrations in debug mode
    if debug:
        if retrieve(config, "debug/disable_integrations", default=True):
            integrations = retrieve(config, "integrations", default=dict())
            for k in integrations:
                config["integrations"][k]["active"] = False
        max_steps = retrieve(config, "debug/max_steps", default=5 * 2)
        if max_steps > 0:
            config["num_steps"] = max_steps

    # backwards compatibility
    if not "datasets" in config:
        config["datasets"] = {"train": config["dataset"]}
        if "validation_dataset" in config:
            config["datasets"]["validation"] = config["validation_dataset"]

    log.set_log_target("train")
    logger = log.get_logger("train")
    logger.info("Starting Training.")

    model = get_obj_from_str(config["model"])
    iterator = get_obj_from_str(config["iterator"])
    datasets = dict(
        (split, get_obj_from_str(config["datasets"][split]))
        for split in config["datasets"]
    )

    logger.info("Instantiating datasets.")
    for split in datasets:
        datasets[split] = datasets[split](config=config)
        datasets[split].expand = True
        logger.info("{} dataset size: {}".format(split, len(datasets[split])))
        if debug:
            max_examples = retrieve(
                config, "debug/max_examples", default=5 * config["batch_size"]
            )
            if max_examples > 0:
                logger.info(
                    "Monkey patching {} dataset __len__ to {} examples".format(
                        split, max_examples
                    )
                )
                type(datasets[split]).__len__ = lambda self: max_examples

    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    logger.info("Building batches.")
    batches = dict()
    for split in datasets:
        batches[split] = make_batches(
            datasets[split],
            batch_size=config["batch_size"],
            shuffle=True,
            n_processes=n_processes,
            n_prefetch=n_prefetch,
            error_on_timeout=config.get("error_on_timeout", False),
        )
    main_split = "train"
    try:
        if "num_steps" in config:
            # set number of epochs to perform at least num_steps steps
            steps_per_epoch = len(datasets[main_split]) / config["batch_size"]
            num_epochs = config["num_steps"] / steps_per_epoch
            config["num_epochs"] = math.ceil(num_epochs)
        else:
            steps_per_epoch = len(datasets[main_split]) / config["batch_size"]
            num_steps = config["num_epochs"] * steps_per_epoch
            config["num_steps"] = math.ceil(num_steps)

        logger.info("Instantiating model.")
        model = model(config)
        if not "hook_freq" in config:
            config["hook_freq"] = 1
        compat_kwargs = dict(
            hook_freq=config["hook_freq"], num_epochs=config["num_epochs"]
        )
        logger.info("Instantiating iterator.")
        iterator = iterator(config, root, model, datasets=datasets, **compat_kwargs)

        logger.info("Initializing model.")
        if checkpoint is not None:
            iterator.initialize(checkpoint_path=checkpoint)
        else:
            iterator.initialize()

        if retrain:
            iterator.reset_global_step()

        # save current config
        logger.info("Starting Training with config:\n{}".format(yaml.dump(config)))
        cpath = _save_config(config, prefix="train")
        logger.info("Saved config at {}".format(cpath))

        logger.info("Iterating.")
        iterator.iterate(batches)
    finally:
        for split in batches:
            batches[split].finalize()


def test(config, root, checkpoint=None, nogpu=False, bar_position=0, debug=False):
    """Run tests. Loads model, iterator and dataset from config."""
    from edflow.iterators.batches import make_batches

    # backwards compatibility
    if not "datasets" in config:
        config["datasets"] = {"train": config["dataset"]}
        if "validation_dataset" in config:
            config["datasets"]["validation"] = config["validation_dataset"]

    log.set_log_target("latest_eval")
    logger = log.get_logger("test")
    logger.info("Starting Evaluation.")

    if "test_batch_size" in config:
        config["batch_size"] = config["test_batch_size"]
    if "test_mode" not in config:
        config["test_mode"] = True

    model = get_obj_from_str(config["model"])
    iterator = get_obj_from_str(config["iterator"])
    datasets = dict(
        (split, get_obj_from_str(config["datasets"][split]))
        for split in config["datasets"]
    )

    logger.info("Instantiating datasets.")
    for split in datasets:
        datasets[split] = datasets[split](config=config)
        datasets[split].expand = True
        logger.info("{} dataset size: {}".format(split, len(datasets[split])))
        if debug:
            max_examples = retrieve(
                config, "debug/max_examples", default=5 * config["batch_size"]
            )
            if max_examples > 0:
                logger.info(
                    "Monkey patching {} dataset __len__ to {} examples".format(
                        split, max_examples
                    )
                )
                type(datasets[split]).__len__ = lambda self: max_examples

    n_processes = config.get("n_data_processes", min(16, config["batch_size"]))
    n_prefetch = config.get("n_prefetch", 1)
    logger.info("Building batches.")
    batches = dict()
    for split in datasets:
        batches[split] = make_batches(
            datasets[split],
            batch_size=config["batch_size"],
            shuffle=False,
            n_processes=n_processes,
            n_prefetch=n_prefetch,
            error_on_timeout=config.get("error_on_timeout", False),
        )
    try:
        logger.info("Initializing model.")
        model = model(config)

        config["hook_freq"] = 1
        config["num_epochs"] = 1
        config["nogpu"] = nogpu
        compat_kwargs = dict(
            hook_freq=config["hook_freq"],
            bar_position=bar_position,
            nogpu=config["nogpu"],
            num_epochs=config["num_epochs"],
        )
        iterator = iterator(config, root, model, datasets=datasets, **compat_kwargs)

        logger.info("Initializing model.")
        if checkpoint is not None:
            iterator.initialize(checkpoint_path=checkpoint)
        else:
            iterator.initialize()

        # save current config
        logger.info("Starting Evaluation with config:\n{}".format(yaml.dump(config)))
        prefix = "eval"
        if bar_position > 0:
            prefix = prefix + str(bar_position)
        cpath = _save_config(config, prefix=prefix)
        logger.info("Saved config at {}".format(cpath))

        logger.info("Iterating")
        while True:
            iterator.iterate(batches)
            if not config.get("eval_forever", False):
                break
    finally:
        for split in batches:
            batches[split].finalize()

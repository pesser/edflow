import tensorflow as tf

import argparse
import importlib
import numpy as np
import glob
import os
import yaml
from tqdm import tqdm, trange

import multiprocessing as mp

from edflow.iterators.batches import make_batches
from edflow.custom_logging import init_project, get_logger
from edflow.project_manager import ProjectManager


def get_implementations_from_config(config, names):
    def get_impl(config, name):
        impl = config[name]
        module, cls = impl.rsplit(".", 1)
        return getattr(importlib.import_module(module, package=None), cls)
    implementations = dict((name, get_impl(config, name)) for name in names)
    return implementations


def train(config, root, checkpoint = None, retrain = False):
    '''Run training. Loads model, iterator and dataset according to config.'''

    logger = get_logger('train')
    logger.info('Starting Training')

    implementations = get_implementations_from_config(
            config, ["model", "iterator", "dataset"])

    # fork early to avoid taking all the crap into forked processes
    dataset = implementations["dataset"](config=config)
    logger.info("Number of training samples: {}".format(len(dataset)))
    batches = make_batches(dataset, batch_size = config["batch_size"], shuffle = True)
    # get them going
    next(batches)
    batches.reset()

    Model = implementations["model"](config)
    Trainer = implementations["iterator"](config, root, Model, hook_freq=config["hook_freq"])

    if checkpoint is not None:
        Trainer.initialize(checkpoint_path=checkpoint)
    else:
        Trainer.initialize()

    if retrain:
        Trainer.reset_global_step()

    Trainer.fit(batches)


def test(config, root, nogpu = False, bar_position = 0):
    '''Run tests. Loads model, iterator and dataset from config.'''

    logger = get_logger('test', 'latest_eval')
    if "test_batch_size" in config:
        config['batch_size'] = config['test_batch_size']

    implementations = get_implementations_from_config(
            config, ["model", "iterator", "dataset"])

    dataset = implementations["dataset"](config = config)
    logger.info("Number of testing samples: {}".format(len(dataset)))
    batches = make_batches(dataset, batch_size = config["batch_size"], shuffle = False)
    # get going
    next(batches)
    batches.reset()

    Model = implementations["model"](config)

    HBU_Evaluator = implementations["iterator"](
        config,
        root,
        Model,
        hook_freq=1,
        bar_position=bar_position,
        nogpu = nogpu)

    while True:
        HBU_Evaluator.iterate(batches)


def main(opt):
    with open(opt.config) as f:
        config = yaml.load(f)

    P = init_project('logs')
    logger = get_logger('main')
    logger.info(opt)
    logger.info(yaml.dump(config))
    logger.info(P)

    if opt.noeval:
        train(config, P.train, opt.checkpoint, opt.retrain)
    else:
        train_process = mp.Process(target=train,
                                   args=(config,
                                         P.train,
                                         opt.checkpoint,
                                         opt.retrain))
        test_process = mp.Process(target=test, args=(config, P.latest_eval, True))

        processes = [train_process, test_process]

        try:
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info('Terminating all processes')
            for p in processes:
                p.terminate()
        finally:
            logger.info('Finished')


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", help="path to checkpoint to restore")
    parser.add_argument("--noeval",
                        action="store_true",
                        default=False,
                        help="only run training")
    parser.add_argument("--retrain",
                        action="store_true",
                        default=False,
                        help="reset global_step to zero")

    opt = parser.parse_args()
    main(opt)

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


def train(config, root, checkpoint = None, retrain = False):
    '''Run training. Implementation should provide TrainDataset, TrainModel and Trainer.'''

    logger = get_logger('train')
    logger.info('Starting Training')

    implementation = config["implementation"]
    impl = importlib.import_module(implementation, package=None)

    Model = impl.TrainModel(config)
    Trainer = impl.Trainer(config, root, Model, hook_freq=config["hook_freq"])
    dataset = impl.TrainDataset(config=config)

    logger.info("Number of training samples: {}".format(len(dataset)))
    batches = make_batches(dataset, batch_size = config["batch_size"], shuffle = True)

    if checkpoint is not None:
        Trainer.initialize(checkpoint_path=checkpoint)
    else:
        Trainer.initialize()

    if retrain:
        Trainer.reset_global_step()

    Trainer.fit(batches)


def test(config, root, nogpu = False, bar_position = 0):
    '''Run tests. Implementation should provide EvalDataset, EvalModel and Evaluator.'''

    logger = get_logger('test', 'latest_eval')
    if "test_batch_size" in config:
        config['batch_size'] = config['test_batch_size']

    implementation = config["implementation"]
    impl = importlib.import_module(implementation, package=None)

    Model = impl.EvalModel(config)
    HBU_Evaluator = impl.Evaluator(config,
                                   root,
                                   Model,
                                   hook_freq=1,
                                   bar_position=bar_position,
                                   nogpu = nogpu)
    dataset = impl.EvalDataset(config = config)

    logger.info("Number of testing samples: {}".format(len(dataset)))
    batches = make_batches(dataset, batch_size = config["batch_size"], shuffle = False)

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

import tensorflow as tf

import argparse
import numpy as np
import glob
import os
import yaml
from tqdm import tqdm, trange

import multiprocessing as mp

from edflow.custom_logging import init_logging, get_logger
from edflow.main import train, test


def main(opt):
    with open(opt.config) as f:
        config = yaml.load(f)

    out_dir = init_logging('logs')
    logger = get_logger('main_training')
    logger.info(opt)
    logger.info(yaml.dump(config))

    if not opt.doeval:
        train(config, out_dir, opt.checkpoint, opt.retrain)
    else:
        train_process = mp.Process(target=train,
                                   args=(config,
                                         out_dir,
                                         opt.checkpoint,
                                         opt.retrain))
        test_process = mp.Process(target=test, args=(config, out_dir))

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
    parser.add_argument("--doeval",
                        action="store_true",
                        default=False,
                        help="only run training")
    parser.add_argument("--retrain",
                        action="store_true",
                        default=False,
                        help="reset global_step to zero")

    opt = parser.parse_args()
    main(opt)

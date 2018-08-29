
import tensorflow as tf

import argparse
import numpy as np
import glob
import os
import yaml
from tqdm import tqdm, trange

import multiprocessing as mp

from edflow.custom_logging import use_project, get_logger
from edflow.main import test


def main(opt):
    with open(opt.config) as f:
        config = yaml.load(f)

    P = use_project(opt.project)
    logger = get_logger('main_evaluate', 'latest_eval')
    logger.info(opt)
    logger.info(P)
    logger.info(yaml.dump(config))

    test(config, P.latest_eval)
    logger.info('Finished')
    print('\n'*5)


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--project", help="path to project root")

    opt = parser.parse_args()
    main(opt)

import sys, os
sys.path.append(os.getcwd()) # convenience: load implementations from cwd
import argparse, yaml
import multiprocessing as mp

import tensorflow as tf
from edflow.main import train, test
from edflow.custom_logging import init_project, use_project, get_logger


def main(opt):
    # Project manager
    if opt.project is not None:
        P = use_project(opt.project)
    else:
        # get path to implementation
        with open(opt.train) as f:
            impl = yaml.load(f)["implementation"]
        # if it looks like a package path, take its root as the code dir
        # otherwise take cwd
        path = impl.split(".")
        if len(path) > 0:
            code_root = path[0]
        else:
            code_root = "."
        P = init_project('logs', code_root = code_root)

    # Logger
    logger = get_logger('main')
    logger.info(opt)
    logger.info(P)

    # Processes
    processes = list()

    # Training
    if opt.train:
        if opt.project is not None:
            checkpoint = tf.train.latest_checkpoint(P.checkpoints)
        else:
            checkpoint = None
        with open(opt.train) as f:
            config = yaml.load(f)
        logger.info("Training config: {}".format(opt.train))
        logger.info(yaml.dump(config))
        train_process = mp.Process(target=train,
                                   args=(config,
                                         P.train,
                                         checkpoint))
        processes.append(train_process)

    # Evaluation
    opt.eval = opt.eval or list()
    for eval_idx, eval_config in enumerate(opt.eval):
        with open(eval_config) as f:
            config = yaml.load(f)
        logger.info("Evaluation config: {}".format(eval_config))
        logger.info(yaml.dump(config))
        nogpu = len(processes) > 0
        bar_position = len(processes) + eval_idx
        test_process = mp.Process(target=test, args=(config, P.latest_eval, nogpu, bar_position))
        processes.append(test_process)

    # Take off
    try:
        for p in processes:
            p.start()

    # Landing
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
    parser.add_argument("-t", "--train",
            metavar = "config.yaml", help="path to training config")
    parser.add_argument("-e", "--eval", nargs = "*",
            metavar = "config.yaml", help="path to evaluation configs")
    parser.add_argument("-p", "--project", help="path to existing project")
    opt = parser.parse_args()
    main(opt)

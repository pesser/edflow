import argparse


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n", "--name", metavar="description", help="postfix of log directory."
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=None,
    )
    parser.add_argument(
        "-t", "--train",
        type=str2bool, const=True, default=False, nargs='?',
        help="run in training mode"
    )
    parser.add_argument("-p", "--project", help="path to existing project")
    parser.add_argument("-c", "--checkpoint", help="path to existing checkpoint")
    parser.add_argument(
        "-r", "--retrain",
        type=str2bool, const=True, default=False, nargs='?',
        help="reset global step"
    )
    parser.add_argument(
        "-l",
        "--log_level",
        metavar="LEVEL",
        type=str,
        choices=["warn", "info", "debug", "critical"],
        default="info",
        help="set the logging level.",
    )
    parser.add_argument("-d", "--debug", type=str2bool, nargs='?', const=True,
            default=False, help="enable post-mortem debugging")
    parser.add_argument(
        "-w",
        "--wandb_sweep",
        nargs='?',
        const=True,
        type=str2bool,
        default=False,
        help="Process additional arguments supplied by wandb's sweep mechanism,"
        "i.e. replace dots ('.') with slashes ('/') in the argument name: "
        "--par.at.level=3 => --par/at/level 3",
    )

    return parser

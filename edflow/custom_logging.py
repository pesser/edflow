import logging
import os
from tqdm import tqdm

from edflow.project_manager import ProjectManager


class TqdmHandler(logging.StreamHandler):
    def __init__(self, pos=4):
        logging.StreamHandler.__init__(self)
        self.tqdm = tqdm(position=pos)

    def emit(self, record):
        msg = self.format(record)
        self.tqdm.write(msg)


def _init_project(out_base_dir):
    """Sets up subdirectories given a base directory and copies all scripts."""

    P = ProjectManager(out_base_dir)

    return P.root


def _get_logger(name, out_dir, pos=4, level=logging.INFO):
    """Creates a logger the way it's meant to be."""
    # init logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not len(logger.handlers) > 0:
        ch = TqdmHandler(pos)
        ch.setLevel(level)

        fh = logging.FileHandler(filename=os.path.join(out_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)

        fmt_string = "[%(levelname)s] [%(name)s]: %(message)s"
        formatter = logging.Formatter(fmt_string)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


class LogSingleton(object):
    exists = False
    default = "root"  # default directory of ProjectManager to log into

    def __init__(self, out_base_dir=None, level=logging.INFO, write_pos=4):
        if self.exists or out_base_dir is None:
            pass
        else:
            LogSingleton.out_base_dir = out_base_dir
            LogSingleton._log_dir = _init_project(self.out_base_dir)
            LogSingleton._level = level
            LogSingleton._write_pos = write_pos
            LogSingleton.exists = True
            LogSingleton.loggers = []

    def set_default(self, which):
        LogSingleton.default = which

    def get(self, name, which=None):
        """Create logger, set level.

        Args:
            name (str or object): Name of the logger. If not a string, the name
                of the given object class is used.
            which (str): subdirectory in the project folder.
        """
        which = which or LogSingleton.default

        if not isinstance(name, str):
            name = type(name).__name__

        log_dir = getattr(ProjectManager, which)
        pos = LogSingleton._write_pos
        logger = _get_logger(name, log_dir, pos, level=LogSingleton._level)
        logger.setLevel(LogSingleton._level)

        LogSingleton.loggers += [logger]

        return logger


def set_global_stdout_level(level="info"):
    level = getattr(logging, level.upper())
    print("Setting Log Level to {}".format(level))

    LogSingleton._level = level
    for logger in LogSingleton.loggers:
        logger.handlers[0].setLevel(level)


def get_default_logger():
    default_log_dir, default_logger = LogSingleton("logs").get("default")
    return default_log_dir, default_logger


def fix_abseil():
    # https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    try:
        import absl.logging

        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
    except Exception:
        pass


def init_project(base_dir, code_root=".", postfix=None):
    """Must be called at the very beginning of a script."""
    fix_abseil()
    P = ProjectManager(base_dir, code_root=code_root, postfix=postfix)
    LogSingleton(P.root)
    return P


def use_project(project_dir, postfix=None):
    """Must be called at the very beginning of a script."""
    P = ProjectManager(given_directory=project_dir, postfix=postfix)
    LogSingleton(P.root)
    return P


def get_logger(name, which=None, level="info"):
    """Creates a logger, which shares its output directory with all other
    loggers.

    Args:
        name (str): Name of the logger.
        which (str): Any subdirectory of the project.
    """

    L = LogSingleton(level=getattr(logging, level.upper()))

    if not L.exists:
        print("Warning: LogSingleton not initialized.")
        if not isinstance(name, str):
            name = type(name).__name__
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        return logger

    return L.get(name, which)

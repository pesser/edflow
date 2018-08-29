import datetime
import glob
import logging
import os
import shutil
import sys
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
    '''Sets up subdirectories given a base directory and copies all scripts.'''

    P = ProjectManager(out_base_dir)

    return P.root

def _get_logger(name, out_dir, pos=4):
    '''Creates a logger the way it's meant to be.'''
    # init logging
    logger = logging.getLogger(name)

    ch = TqdmHandler(pos)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    fh = logging.FileHandler(filename=os.path.join(out_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    formatter = logging.Formatter('[%(levelname)s] [%(name)s]: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    return logger


class LogSingleton(object):
    exists = False
    def __init__(self, out_base_dir=None, level=logging.DEBUG, write_pos=4):
        if self.exists or out_base_dir is None:
            pass
        else:
            LogSingleton.out_base_dir = out_base_dir
            LogSingleton._log_dir = _init_project(self.out_base_dir)
            LogSingleton._level = level
            LogSingleton._write_pos = write_pos
            LogSingleton.exists = True

    def get(self, name, which='train'):
        '''Create logger, set level.
        
        Args:
            name (str or object): Name of the logger. If not a string, the name
                of the given object class is used.
            which (str): subdirectory in the project folder.
        '''

        if not isinstance(name, str):
            name = type(name).__name__

        log_dir = getattr(ProjectManager, which)
        pos = LogSingleton._write_pos
        logger = _get_logger(name, log_dir, pos)
        logger.setLevel(LogSingleton._level)

        return logger


def get_default_logger():
    default_log_dir, default_logger = LogSingleton('logs').get('default')
    return default_log_dir, default_logger

def init_project(base_dir, code_root = "."):
    '''Must be called at the very beginning of a script.'''
    P = ProjectManager(base_dir, code_root = code_root)
    LogSingleton(P.root)
    return P

def use_project(project_dir):
    '''Must be called at the very beginning of a script.'''
    P = ProjectManager(given_directory=project_dir)
    LogSingleton(P.root)
    return P

def get_logger(name, which='train'):
    '''Creates a logger, which shares its output directory with all other
    loggers.
    
    Args:
        name (str): Name of the logger.
        which (str): Any subdirectory of the project.
    '''

    L = LogSingleton()

    if not L.exists:
        raise ValueError('LogSingleton not initialized. Please run '
                         'init_project.')

    return L.get(name, which)

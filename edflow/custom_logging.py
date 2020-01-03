import logging
import os, sys
import datetime
import shutil
import subprocess
from socket import gethostname
from tqdm import tqdm


class run(object):
    """
    Singleton managing all directories for a run. Useful attributes:
    - now: string representing init time
    - postfix: user specified postfix for run or eval directory
    - name: name of the run
    - git_tag: associated tag if git is used
    - resumed: if this is a resumed run
    - code_root: where code is copied from
    - code: path to copied code
    - root: path under which all outputs of the run should be stored
    - train: path to store train outputs in
    - eval: path to eval subfolders
    - latest_eval: path to store eval outputs in
    - configs: path to store configs in
    - checkpoints: path to store checkpoints in
    """

    exists = False

    @classmethod
    def init(
        cls,
        log_dir=None,
        run_dir=None,
        code_root=".",
        postfix=None,
        log_level="info",
        git=True,
    ):
        """
        Parameters
        ----------
        log_dir : str
	    Create new run directory under this directory.
        run_dir : str
	    Resume in existing run directory.
        code_root : str
	    Path to where the code lives. py and yaml files will be copied into
            run directory.
        postfix : str
	    Identifier appended to run directory if non-existent else to latest
            eval directory.
        log_level : str
	    Default log level for loggers.
        git : bool
	    If True, put code into tagged commit.
        """

        has_info = log_dir is not None or run_dir is not None
        if not cls.exists and has_info:
            cls.now = now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            cls.postfix = postfix
            cls.code_root = code_root
            if run_dir is None:
                if postfix is not None:
                    name = now + "_" + postfix
                else:
                    name = now
                cls.resumed = False
                cls.root = os.path.join(log_dir, name)
            else:
                cls.resumed = True
                cls.root = run_dir
            cls.name = os.path.split(cls.root)[1]

            # create directory structure
            cls.setup()
            cls.setup_new_eval()
            cls.exists = True

            # log run information
            log.set_log_level(log_level)
            cls.logger = get_logger("run")
            cls.logger.info(" ".join(sys.argv))
            cls.logger.info("root: {}".format(cls.root))
            cls.logger.info("hostname: {}".format(gethostname()))
            try:
                # try to match tmux target by tty.
                # only works if stdin was not messed.
                tty = os.ttyname(sys.stdin.fileno())
                tmux_target = subprocess.run(
                    [
                        "tmux list-panes -a -F"
                        + "'#{session_id}:#{window_id}.#{pane_id} #{pane_tty}'"
                        + "| grep {}".format(tty)
                    ],
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout
                tmux_target = tmux_target.split("\n")[0].split(" ")[0]
                # output can be used as tmux target, eg 'tmux a {tmux_target}'
                # to attach to the pane running the logged run
                cls.logger.info("tmux: {}".format(tmux_target))
            except Exception:
                pass
            cls.logger.info("pid: {}".format(os.getpid()))
            cls.logger.info("pgid: {}".format(os.getpgrp()))
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cls.logger.info(
                    "cuda_devices: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
                )

            # log code
            if not cls.resumed and cls.code_root is not None:
                cls.copy_code()
            if git:
                cls.git_tag = cls.git_commit()
                cls.logger.info("git_tag: {}".format(cls.git_tag))

            cls.logger.info(cls())

    @classmethod
    def setup(cls):
        """Make all the directories."""

        subdirs = ["code", "train", "eval", "configs"]
        subsubdirs = {"code": [], "train": ["checkpoints"], "eval": [], "configs": []}

        root = cls.root

        cls.repr = "Project structure:\n{}\n".format(root)

        for sub in subdirs:
            path = os.path.join(root, sub)
            setattr(cls, sub, path)
            if sub != "code":
                # Code directory will be created by copy code
                os.makedirs(path, exist_ok=True)

            cls.repr += "├╴{}\n".format(sub)

            for subsub in subsubdirs[sub]:
                path = os.path.join(root, sub, subsub)
                setattr(cls, subsub, path)
                os.makedirs(path, exist_ok=True)

                cls.repr += "  ├╴{}\n".format(subsub)

    @classmethod
    def setup_new_eval(cls):
        """Always create subfolder in eval to avoid clashes between
        evaluations."""
        name = cls.now
        if cls.postfix is not None:
            name = name + "_" + cls.postfix
        cls.latest_eval = os.path.join(cls.eval, name)
        os.makedirs(cls.latest_eval)

    @classmethod
    def copy_code(cls):
        """Copies all code to the code directory of the run."""

        src = cls.code_root
        dst = "./" + cls.code

        try:
            filtered_dirs = ["__pycache__"]

            def ignore(directory, files):
                filtered = []
                for f in files:
                    full_path = os.path.join(directory, f)
                    is_cool = False
                    if f[-3:] == ".py":
                        is_cool = True
                    elif f[-5:] == ".yaml":
                        is_cool = True
                    elif os.path.isdir(full_path):
                        if not (f.startswith(".") or f in filtered_dirs):
                            is_cool = True

                    if not is_cool:
                        filtered += [f]

                return filtered

            shutil.copytree(src, dst, symlinks=False, ignore=ignore)

        except shutil.Error as err:
            cls.logger.warning(err)

    @classmethod
    def git_commit(cls):
        # perform the following
        # CHEAD=$(git rev-parse HEAD); git add <files...>; git add -u; git commit -m "edflow ..."; git tag -a edflow_date-and-time-project -m "more"; git reset --mixed $CHEAD
        try:
            CHEAD = subprocess.run(
                ["git rev-parse HEAD"],
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            cls.logger.warning(
                "Tried to commit state of project but the current working directory does not appear to be a git repository."
            )
            tagname = "error: no git repository found"
        else:
            tagname = "{}_{}".format(cls.now, cls.postfix)
            message = "command: {}\nroot: {}".format(" ".join(sys.argv), cls.root)
            try:
                addcommand = "git add {pyfiles}; git add {yamlfiles}; git add -u".format(
                    pyfiles=os.path.join(cls.code_root, "\*.py"),
                    yamlfiles=os.path.join(cls.code_root, "\*.yaml"),
                )
                output = subprocess.run(
                    [addcommand],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
                cls.logger.debug(output)
                if (
                    subprocess.run(
                        ["git diff-index --quiet HEAD"], shell=True
                    ).returncode
                    != 0
                ):
                    # dirty working directory - add commit
                    command = "git commit -m '{commitmessage}'; git tag '{tagname}'".format(
                        commitmessage=message, tagname=tagname
                    )
                else:
                    # nothing to add - put message into annotated tag
                    command = "git tag -a '{tagname}' -m '{tagmessage}'".format(
                        tagname=tagname, tagmessage=message
                    )
                cls.logger.debug(command)
                output = subprocess.run(
                    [command],
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
                cls.logger.debug(output)
            except Exception as e:
                cls.logger.warning(
                    "Tried to commit state of project but error occured: {}".format(e)
                )
                tagname = "error: {}".format(e)
            finally:
                cls.logger.debug("git reset --mixed {}".format(CHEAD))
                output = subprocess.run(
                    ["git reset --mixed {}".format(CHEAD)],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
                cls.logger.debug(output)
        return tagname

    def __repr__(self):
        """Nice file structure representation."""
        return type(self).repr


class TqdmHandler(logging.StreamHandler):
    def __init__(self, pos=4):
        logging.StreamHandler.__init__(self)
        self.tqdm = tqdm(position=pos)

    def emit(self, record):
        # check if stderr and stdout are two different ptys.
        # this detects tampering by wandb which messes up tqdm logging.
        # fix it by writing to stderr instead of stdout.
        try:
            file_ = sys.stdout
            if not os.ttyname(sys.stdout.fileno()) == os.ttyname(sys.stderr.fileno()):
                file_ = sys.stderr
        except OSError:
            # stdout or stderr is not a pty. default to stdout.
            file_ = sys.stdout

        msg = self.format(record)
        self.tqdm.write(msg, file=file_)


class log(object):
    exists = False
    target = "root"  # default directory of ProjectManager to log into
    level = logging.INFO
    loggers = []

    @classmethod
    def set_log_target(cls, which):
        cls.target = which

    @classmethod
    def get_logger(cls, name, which=None, level=None):
        """Create logger, set level.

        Parameters
        ----------
        name : str or object
	    Name of the logger. If not a string, the name
            of the given object class is used.
        which : str
	    Subdirectory in the project folder.
        level : str
            Log level of the logger.
        """
        _fix_abseil()
        which = which or cls.target
        level = level or cls.level

        if not isinstance(name, str):
            name = type(name).__name__

        if not run.exists:
            if not isinstance(name, str):
                name = type(name).__name__
            logging.basicConfig(level=level)
            logger = logging.getLogger(name)
            logger.debug("edflow not initialized.")
            return logger

        log_dir = getattr(run, which)
        logger = cls._create_logger(name, log_dir, level=level)

        cls.loggers += [logger]

        return logger

    @classmethod
    def set_log_level(cls, level):
        level = getattr(logging, level.upper())
        log.level = level
        for logger in log.loggers:
            logger.setLevel(level)
        cls.get("log").debug("Log level set to {}".format(level))

    @staticmethod
    def _create_logger(name, out_dir, pos=4, level=logging.INFO):
        """Creates a logger with tqdm- and file-handler."""
        # init logging
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not len(logger.handlers) > 0:
            ch = TqdmHandler(pos)
            fh = logging.FileHandler(filename=os.path.join(out_dir, "log.txt"))

            fmt_string = "[%(levelname)s] [%(name)s]: %(message)s"
            formatter = logging.Formatter(fmt_string)
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(ch)
            logger.addHandler(fh)

        return logger


def _fix_abseil():
    # https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    try:
        import absl.logging

        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
    except Exception:
        pass


# backwards compatibility
LogSingleton = log
get_logger = log.get_logger

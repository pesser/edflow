import datetime
import os
import shutil
from shutil import Error


class ProjectManager(object):
    """Singelton managing all directories for one Experiment."""

    exists = False

    def __init__(self, base=None, given_directory=None, code_root=".", postfix=None):
        """Args:
            base (str): Top level directory, where all experiments live.
            given_directory (str): If not None, this will be used to get all
                relevant paths.
            code_root (str): Path to where the code lives.
        """

        self.postfix = postfix
        has_info = base is not None or given_directory is not None
        if not self.exists and has_info:
            ProjectManager.now = now = datetime.datetime.now().strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            if given_directory is None:
                if postfix is not None:
                    name = now + "_" + postfix
                else:
                    name = now
                ProjectManager.root = os.path.join(base, name)
                ProjectManager.code_root = code_root
                ProjectManager.super_root = base
            else:
                ProjectManager.root = given_directory

            self.setup()
            self.setup_new_eval()

            if given_directory is None and ProjectManager.code_root is not None:
                self.copy_code()

            ProjectManager.exists = True
        else:
            pass

    def setup(self):
        """Make all the directories."""

        subdirs = ["code", "train", "eval", "configs"]
        subsubdirs = {"code": [], "train": ["checkpoints"], "eval": [], "configs": []}

        root = ProjectManager.root

        ProjectManager.repr = "Project structure:\n{}\n".format(root)

        for sub in subdirs:
            path = os.path.join(root, sub)
            setattr(ProjectManager, sub, path)
            if sub != "code":
                # Code directory will be created by copy code
                os.makedirs(path, exist_ok=True)

            ProjectManager.repr += "├╴{}\n".format(sub)

            for subsub in subsubdirs[sub]:
                path = os.path.join(root, sub, subsub)
                setattr(ProjectManager, subsub, path)
                os.makedirs(path, exist_ok=True)

                ProjectManager.repr += "  ├╴{}\n".format(subsub)

    def setup_new_eval(self):
        """Always create subfolder in eval to avoid clashes between
        evaluations."""
        name = ProjectManager.now
        if self.postfix is not None:
            name = name + "_" + self.postfix
        ProjectManager.latest_eval = os.path.join(ProjectManager.eval, name)
        os.makedirs(ProjectManager.latest_eval)

    def copy_code(self):
        """Copies all code to the code directory of the project, for best
        possible documentation."""

        src = ProjectManager.code_root
        dst = "./" + ProjectManager.code

        print(src)
        print(dst)

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

                print(directory, filtered)
                return filtered

            shutil.copytree(src, dst, symlinks=False, ignore=ignore)

        except shutil.Error as err:
            print(err)
            pass

    def __repr__(self):
        """Nice file structure representation."""

        return ProjectManager.repr

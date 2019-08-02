import numpy as np
import os
import traceback
import yaml

from edflow.hooks.hook import Hook
from edflow.util import walk, retrieve, contains_key
from edflow.custom_logging import get_logger


class RuntimeInputHook(Hook):
    """Given a textfile reads that at each step and passes the results to
    a callback function."""

    def __init__(self, update_file, callback):
        """Args:
            update_file (str): path/to/yaml-file containing the parameters of
                interest.
            callback (Callable): Each time something changes in the update_file
                this function is called with the content of the file as
                argument.
        """

        self.logger = get_logger(self)

        self.ufile = update_file
        self.callback = callback

        self.last_updates = None

        if not os.path.exists(self.ufile):
            msg = (
                "# Automatically created file. Changes made in here will "
                "be recognized during runtime."
            )
            with open(self.ufile, "w+") as f:
                f.write(msg)

    def before_step(self, *args, **kwargs):
        """Checks if something changed and if yes runs the callback."""

        try:
            updates = yaml.full_load(open(self.ufile, "r"))

            if self.last_updates is not None:
                changes = {}

                def is_changed(key, val, changes=changes):
                    if contains_key(key, updates):
                        other_val = retrieve(updates, key)

                        change = np.any(val != other_val)
                    else:
                        # This key is new -> Changes did happen!
                        change = True
                    changes[key] = change

                self.logger.debug("Pre  CHANGES: {}".format(changes))
                walk(self.last_updates, is_changed, pass_key=True)
                self.logger.debug("Post CHANGES: {}".format(changes))

                if np.any(list(changes.values())):
                    self.callback(updates)

                    self.logger.debug("Runtime inputs received.")
                    self.logger.debug("{}".format(updates))

                    self.last_updates = updates
            else:
                if updates is not None:
                    self.callback(updates)

                    self.logger.info("Runtime inputs received.")
                    self.logger.debug("{}".format(updates))

                    self.last_updates = updates
        except Exception as e:
            self.logger.error("Something bad happend :(")
            self.logger.error("{}".format(e))
            self.logger.error(traceback.format_exc())

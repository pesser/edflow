from edflow.hooks.hook import Hook
from edflow.util import retrieve
from edflow.custom_logging import get_logger
from edflow.iterators.batches import plot_batch
import os


class LoggingHook(Hook):
    """Minimal implementation of a logging hook. Can be easily extended by
    adding handlers."""

    def __init__(self, paths, interval, root_path, name=None):
        """
        Parameters
        ----------
        paths : list(str)
            List of key-paths to logging outputs. Will be
            expanded so they can be evaluated lazily.
        interval : int
            Intervall of training steps before logging.
        root_path : str
            Path at which the logs are stored.
        name : str
            Optional name to recognize logging output.
        """
        self.paths = paths
        self.interval = interval
        self.root = root_path
        if name is not None:
            self.logger = get_logger(name)
        else:
            self.logger = get_logger(self)
        self.loggers = dict()
        for path in self.paths:
            os.makedirs(os.path.join(self.root, path), exist_ok=True)
            self.loggers[path] = get_logger(path)
        self.handlers = {
            "images": [self.log_images],
            "scalars": [self.log_scalars],
            "figures": [self.log_figures],
        }

    def __call__(self, results, step, paths):
        for path in paths:
            for k in self.handlers:
                handler_results = retrieve(results, path + "/" + k, default=dict())
                for handler in self.handlers[k]:
                    handler(handler_results, step, path=path)

    def after_step(self, batch_index, last_results):
        if batch_index % self.interval == 0:
            active = False
            step = last_results["global_step"]
            for path in self.paths:
                for k in self.handlers:
                    handler_results = retrieve(
                        last_results, path + "/" + k, default=dict()
                    )
                    if handler_results and not active:
                        self.logger.info("global_step: {}".format(step))
                        active = True
                    for handler in self.handlers[k]:
                        handler(handler_results, step, path=path)
            if active:
                self.logger.info("logging root: {}".format(self.root))

    def log_scalars(self, results, step, path):
        for name in sorted(results.keys()):
            if not path in self.loggers:
                self.loggers[path] = get_logger(path)
            self.loggers[path].info("{}: {}".format(name, results[name]))

    def log_figures(self, results, step, path):
        import matplotlib.pyplot as plt

        for name, figure in results.items():
            full_name = name + "_{:07}.png".format(step)
            save_path = os.path.join(self.root, path, full_name)
            try:
                plt.savefig(save_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(save_path)[0])
                plt.savefig(save_path)

    def log_images(self, results, step, path):
        for name, image_batch in results.items():
            full_name = name + "_{:07}.png".format(step)
            save_path = os.path.join(self.root, path, full_name)
            try:
                plot_batch(image_batch, save_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(save_path)[0])
                plot_batch(image_batch, save_path)

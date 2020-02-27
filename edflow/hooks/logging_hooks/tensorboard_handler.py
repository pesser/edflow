import numpy as np
import yaml
from edflow.iterators.batches import batch_to_canvas


def log_tensorboard_scalars(writer, results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    for k, v in results.items():
        writer.add_scalar(tag=k, scalar_value=v, global_step=step)
    writer.flush()


def log_tensorboard_images(writer, results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    for k, v in results.items():
        v = batch_to_canvas(v)
        v = ((v + 1) * 127.5).astype(np.uint8)
        writer.add_image(tag=k, img_tensor=v, global_step=step, dataformats="HWC")
    writer.flush()


def log_tensorboard_figures(writer, results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    for k, v in results.items():
        writer.add_figure(tag=k, figure=v, global_step=step)
    writer.flush()


def log_tensorboard_config(writer, config, step):
    config_string = (
        "<pre>" + yaml.dump(config) + "</pre>"
    )  # use <pre> for correct indentation
    writer.add_text("config", config_string, step)
    writer.flush()

import tensorboardX
import yaml


def log_tensorboard_scalars(writer, results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    for k, v in results.items():
        writer.add_scalar(tag=k, scalar_value=v, global_step=step)
    writer.flush()


def log_tensorboard_config(writer, config, step):
    config_string = (
        "<pre>" + yaml.dump(config) + "</pre>"
    )  # use <pre> for correct indentation
    writer.add_text("config", config_string, step)
    writer.flush()

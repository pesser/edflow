import numpy as np
import wandb
from edflow.iterators.batches import batch_to_canvas


def log_wandb(results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    wandb.log(results, step=step)


def log_wandb_images(results, step, path):
    results = dict(
        (
            path + "/" + k,
            wandb.Image(((batch_to_canvas(v) + 1) * 127.5).astype(np.uint8)),
        )
        for k, v in results.items()
    )
    wandb.log(results, step=step)

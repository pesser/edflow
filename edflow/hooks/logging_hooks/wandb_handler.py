import wandb


def log_wandb(results, step, path):
    results = dict((path + "/" + k, v) for k, v in results.items())
    wandb.log(results, step=step)

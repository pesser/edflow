import wandb

def log_wandb(results, step, prefix=""):
    if prefix:
        results = dict((prefix+"_"+k,v) for k,v in results.items())
    wandb.log(results, step=step)

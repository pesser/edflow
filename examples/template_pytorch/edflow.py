import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from edflow import TemplateIterator, get_logger


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config["n_classes"])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    @property
    def callbacks(self):
        return {"eval_op": {"acc_callback": acc_callback}}

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def step_op(self, model, **kwargs):
        # get inputs
        inputs, labels = kwargs["image"], kwargs["class"]
        inputs = torch.tensor(inputs)
        inputs = inputs.permute(0, 3, 1, 2)
        labels = torch.tensor(labels, dtype=torch.long)

        # compute loss
        outputs = model(inputs)
        loss = self.criterion(outputs, labels)
        mean_loss = torch.mean(loss)

        def train_op():
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()

        def log_op():
            acc = np.mean(
                np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy()
            )
            min_loss = np.min(loss.detach().numpy())
            max_loss = np.max(loss.detach().numpy())
            return {
                "images": {"inputs": inputs.detach().permute(0, 2, 3, 1).numpy()},
                "scalars": {
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "mean_loss": mean_loss,
                    "acc": acc,
                },
            }

        def eval_op():
            return {
                "outputs": np.array(outputs.detach().numpy()),
                "labels": {"loss": np.array(loss.detach().numpy())},
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


def acc_callback(root, data_in, data_out, config):
    from tqdm import trange

    logger = get_logger("acc_callback")
    correct = 0
    seen = 0
    # labels are loaded directly into memory
    loss1 = np.mean(data_out.labels["loss"])
    loss2 = 0.0
    for i in trange(len(data_in), leave=False):
        # data_in is the dataset that was used for evaluation
        labels = data_in[i]["class"]
        # data_out contains all the keys that were specified in the eval_op
        outputs = data_out[i]["outputs"]
        # labels are also available on each example
        loss = data_out[i]["labels_"]["loss"]

        prediction = np.argmax(outputs, axis=0)
        correct += labels == prediction
        loss2 += loss
    logger.info("Loss1: {}".format(loss1))
    logger.info("Loss2: {}".format(loss2 / len(data_in)))
    logger.info("Accuracy: {}".format(correct / len(data_in)))
    return {"scalars": {"accuracy": correct / len(data_in)}}

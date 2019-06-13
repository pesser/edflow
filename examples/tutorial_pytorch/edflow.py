import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from edflow.data.dataset import DatasetMixin
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.pytorch_hooks import PyCheckpointHook
from edflow.hooks.hook import Hook
from edflow.hooks.checkpoint_hooks.torch_checkpoint_hook import RestorePytorchModelHook
from edflow.project_manager import ProjectManager

class Dataset(DatasetMixin):
    """We just initialize the same dataset as in the tutorial and only have to
    implement __len__ and get_example."""
    def __init__(self, config):
        self.train = not config.get("test_mode", False)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=self.train,
                                                download=True, transform=transform)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        """edflow assumes  a dictionary containing numpy arrays for each
        example."""
        x, y = self.dataset[i]
        return {"x": x.numpy(), "y": y}


class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model(object):
    def __init__(self, config):
        """For illustration we read `n_classes` from the config."""
        self.net = Net(n_classes = config["n_classes"])

    def __call__(self, x):
        return self.net(torch.tensor(x))

    def parameters(self):
        return self.net.parameters()


class FinalLogHook(Hook):
    def __init__(self, iterator):
        self.iterator = iterator
        self.logger = iterator.logger
        self.correct = 0
        self.total = 0

    def after_step(self, step, last_results):
        outputs, labels = last_results["step_ops"]
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.shape[0]
        self.correct += (predicted == torch.tensor(labels)).sum().item()

        if step % 50 == 0:
            self.logger.info('Accuracy of the network on the %d step: %d %%' % (
                step, 100 * self.correct / self.total))

    def after_epoch(self, epoch):
        self.logger.info('Accuracy of the network on all test images: %d %%' % (
            100 * self.correct / self.total))



class Iterator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.running_loss = 0.0

        self.restorer = RestorePytorchModelHook(
                checkpoint_path = ProjectManager.checkpoints,
                model = self.model.net)
        # we add a hook to write checkpoints of the model each epoch or when
        # training is interrupted by ctrl-c
        if not self.config.get("test_mode", False):
            self.ckpt_hook = PyCheckpointHook(
                root_path = ProjectManager.checkpoints,
                model = self.model.net) # PyCheckpointHook expects a torch.nn.Module
            self.hooks.append(self.ckpt_hook)
        else:
            # during evaluation, restore latest checkpoint before each epoch
            self.hooks.append(self.restorer)
            self.hooks.append(FinalLogHook(self))
                

    def initialize(self, checkpoint_path = None):
        if checkpoint_path is not None:
            self.restorer(checkpoint_path)


    def step_ops(self):
        if self.config.get("test_mode", False):
            return self.test_op
        else:
            return self.train_op


    def train_op(self, model, x, y, **kwargs):
        """All ops to be run as step ops receive model as the first argument
        and keyword arguments as returned by get_example of the dataset."""

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = x, y

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, torch.tensor(labels))
        loss.backward()
        self.optimizer.step()

        # print statistics
        self.running_loss += loss.item()
        i = self.get_global_step()
        if i % 200 == 199:    # print every 200 mini-batches
            # use the logger instead of print to obtain both console output and
            # logging to the logfile in project directory
            self.logger.info('[%5d] loss: %.3f' %
                    (i + 1, self.running_loss / 200))
            self.running_loss = 0.0


    def test_op(self, model, x, y, **kwargs):
        images, labels = x, y
        outputs = self.model(images)
        return outputs, labels

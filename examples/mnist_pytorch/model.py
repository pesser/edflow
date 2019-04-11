import torch
import torch.nn as nn
import torch.nn.functional as F

from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.pytorch_hooks import DataPrepHook


class CNN(nn.Module):
    def __init__(self, config):
        """
        The Network class you are using for training.
        :param config: This parameter is mandatory for EDflow. The config file contains different
                       parameters for the model.
        """
        super(CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Iterator(PyHookedModelIterator):
    def __init__(self, config, root, model, **kwargs):
        """
        THe iterator class is the backbone of your EDflow training. It will handle your training
        loop and the parameters. In here, you specify the input and output of
        :param config:This parameter is mandatory for EDflow. The config file contains different
                       parameters for the model.
        :param root: The path where to the source code of your project. The model, iterator and dataset parameters in
                     the config are provided with respect to this path.
        :param model: The model you want to use.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(config, root, model, **kwargs)
        self.model = model
        self.config = config
        bs = config['batch_size']
        self.bs = config.get('applied_batch_size', None)
        if self.bs is None:
            self.bs = bs
        self.lr = lr = config.get('learning_rate', 1e-4)
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          betas=(0.5, 0.9))
        self.criterion = torch.nn.CrossEntropyLoss()
        DPrepH = DataPrepHook()
        self.hooks += [DPrepH]

    def initialize(self, checkpoint=None, **kwargs):
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
        self.model.cuda()

    def get_inputs(model, image, target):
        """
        Collects the input and sorts it in a dictionary accordingly.
        :param image: The image to be provided to the network.
        :param target: The target label.
        :return: Nested dictionary with the input keys and values.
        """
        return {'inputs': {
            'image': image,
            'target': target,
        }}

    def _collect_output(self, model, image):
        """
        Collects the result of the network in a dictionary.
        :param model: The model used.
        :param image: The input image.
        :return: The predicted label/class.
        """
        pred = model(image)
        output = dict()
        output["pred"] = pred
        return output

    def _collect_loss(self, input, output, criterion):
        """
        Collects the loss based on a certain criterion.
        :param input: The network input.
        :param output: The network output.
        :param criterion: A loss function.
        :return: The loss value.
        """
        target = input["target"].long()
        pred = output["pred"]
        return criterion(pred, target)

    def train_op(self, model, image, target, index_=None):
        """
        Takes care of the training step.
        :param model: The model used.
        :param image: The input image.
        :param target: The target class.
        :param index_: Required parameter. # TODO: Check if it's needed. If yes, then check why it's needed.
        :return:
        """
        start_idx = 0
        num_iters = 1000
        losses = dict()
        losses["loss"] = []
        while start_idx < num_iters:
            model.train()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = self._collect_output(model, image)
                loss = self._collect_loss({"target": target}, output, self.criterion)
                loss.backward()
                self.optimizer.step()
            start_idx += self.bs
            losses["loss"] += [loss]
        return {"losses": losses}

    def step_ops(self):
        """
        This is the function called by EDflow.
        :return: Our designed training step.
        """
        return self.train_op

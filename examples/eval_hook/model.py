import torch
import torch.nn as nn
import torch.nn.functional as F

from edflow.iterators.torch_iterator import TorchHookedModelIterator
from edflow.eval.pipeline import EvalHook
from edflow.main import get_obj_from_str
from edflow.hooks.pytorch_hooks import ToNumpyHook


def empty_callback(root, data_in, data_out, config):

    for i in range(min(10, len(data_out))):
        ex_in = data_in[i]
        ex_out = data_out[i]

    print("EMPTY CB DONE")


class CNN(nn.Module):
    def __init__(self, config):
        """
        The Network class you are using for training.
        :param config: This parameter is mandatory for EDflow. The config file contains different
                       parameters for the model.
        """
        super(CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 2)
        self.conv3 = nn.Conv2d(50, 100, 4, 2)

        self.upconv1 = nn.ConvTranspose2d(100, 50, 4, 2)
        self.upconv2 = nn.ConvTranspose2d(50, 20, 5, 2)
        self.upconv3 = nn.ConvTranspose2d(20, 3, 5, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))

        return F.sigmoid(x)


class Iterator(TorchHookedModelIterator):
    def __init__(self, config, root, model, **kwargs):
        """
        The iterator class is the backbone of your edflow training. It will handle your training
        loop and the parameters. In here, you specify the input and output of
        :param config: This parameter is mandatory for edflow. The config file
            contains different parameters for the model.
        :param root: The path where to the source code of your project. The model, iterator and dataset parameters in
                     the config are provided with respect to this path.
        :param model: The model you want to use.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(config, root, model, **kwargs)

        self.model = model
        self.config = config

        dset = get_obj_from_str(config["dataset"])
        dset = dset(config)

        self.hooks += [ToNumpyHook()]
        self.hooks += [
            EvalHook(
                dset,
                ["target"],
                callbacks=[empty_callback],
                meta=config,
                step_getter=self.get_global_step,
            )
        ]

    def initialize(self, checkpoint=None, **kwargs):
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
        if torch.cuda.is_available():
            self.model.cuda()

    def eval_op(self, model, image, target, index_=None):
        """
        Takes care of the training step.
        :param model: The model used.
        :param image: The input image.
        :param target: The target class.
        :param index_: Required parameter. # TODO: Check if it's needed. If yes, then check why it's needed.
        :return:
        """

        output = model(image)

        return {"generated": output, "index_": index_, "target": target}

    def step_ops(self):
        """
        This is the function called by EDflow.
        :return: Our designed training step.
        """
        return self.eval_op

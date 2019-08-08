# edflow

A framework independent engine for training and evaluating in batches.

## Table of Contents
1. [Installation](#Installation)
2. [Getting Started](#Getting-Started)
    1. [TensorFlow Eager](#TensorFlow-Eager)
    2. [PyTorch](#PyTorch)
    3. [TensorFlow Graph-Building](#TensorFlow-Graph-Building)
3. [Documentation](#Documentation)
4. [Command-Line Parameters](#Command-Line-Parameters)
5. [Contributions](#Contributions)
6. [LICENSE](#LICENSE)
7. [Authors](#Authors)

## Installation

    git clone https://github.com/pesser/edflow.git
    cd edflow
    pip install .


## Getting Started


    cd examples


### TensorFlow Eager

You provide an implementation of a model and an iterator and use `edflow` to
train and evaluate your model. An example can be found in
`template_tfe/edflow.py`:

```python
import functools
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow.keras as tfk
import numpy as np
from edflow import TemplateIterator, get_logger


class Model(tfk.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = tfk.layers.Conv2D(filters=6, kernel_size=5)
        self.pool = tfk.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tfk.layers.Conv2D(filters=16, kernel_size=5)
        self.fc1 = tfk.layers.Dense(units=120)
        self.fc2 = tfk.layers.Dense(units=84)
        self.fc3 = tfk.layers.Dense(units=config["n_classes"])

        input_shape = (config["batch_size"], 28, 28, 1)
        self.build(input_shape)

    def call(self, x):
        x = self.pool(tf.nn.relu(self.conv1(x)))
        x = self.pool(tf.nn.relu(self.conv2(x)))
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.criterion = functools.partial(
            tfk.losses.sparse_categorical_crossentropy, from_logits=True
        )
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        # to save and restore
        self.tfcheckpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer
        )

    def save(self, checkpoint_path):
        self.tfcheckpoint.write(checkpoint_path)

    def restore(self, checkpoint_path):
        self.tfcheckpoint.restore(checkpoint_path)

    def step_op(self, model, **kwargs):
        # get inputs
        inputs, labels = kwargs["image"], kwargs["class"]

        # compute loss
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = self.criterion(y_true=labels, y_pred=outputs)
            mean_loss = tf.reduce_mean(loss)

        def train_op():
            grads = tape.gradient(mean_loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        def log_op():
            acc = np.mean(np.argmax(outputs, axis=1) == labels)
            min_loss = np.min(loss)
            max_loss = np.max(loss)
            return {
                "images": {"inputs": inputs},
                "scalars": {
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "mean_loss": mean_loss,
                    "acc": acc,
                },
            }

        def eval_op():
            return {"outputs": np.array(outputs), "loss": np.array(loss)[:, None]}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
```

Specify your parameters in a `yaml` config file, e.g.
`template_tfe/config.yaml`:

```yaml
dataset: edflow.datasets.fashionmnist.FashionMNIST
model: template_tfe.edflow.Model
iterator: template_tfe.edflow.Iterator
batch_size: 4
num_epochs: 2

n_classes: 10
```

#### Train
To start training, use the `-t/--train <config>` command-line option and,
optionally, the `-n/--name <name>` option to more easily find your experiments
later on:


```
$ edflow -t template_tfe/config.yaml -n hello_tfe
[INFO] [train]: Starting Training.
[INFO] [train]: Instantiating dataset.
[INFO] [FashionMNIST]: Using split: train
[INFO] [train]: Number of training samples: 60000
[INFO] [train]: Warm up batches.
[INFO] [train]: Reset batches.
[INFO] [train]: Instantiating model.
[INFO] [train]: Instantiating iterator.
[INFO] [train]: Initializing model.
[INFO] [train]: Starting Training with config:
batch_size: 4
dataset: edflow.datasets.fashionmnist.FashionMNIST
hook_freq: 1
iterator: template_tfe.edflow.Iterator
model: template_tfe.edflow.Model
n_classes: 10
num_epochs: 2
num_steps: 30000

[INFO] [train]: Saved config at logs/2019-08-05T18:55:20_hello_tfe/configs/train_2019-08-05T18:55:26.yaml
[INFO] [train]: Iterating.
[INFO] [LoggingHook]: global_step: 0
[INFO] [LoggingHook]: acc: 0.25
[INFO] [LoggingHook]: max_loss: 2.3287339210510254
[INFO] [LoggingHook]: mean_loss: 2.256807565689087
[INFO] [LoggingHook]: min_loss: 2.2113394737243652
[INFO] [LoggingHook]: project root: logs/2019-08-05T18:55:20_hello_tfe/train
...
```

edflow shows the progress of your training and scalar logging values. The log
file, log outputs and checkpoints can be found in the `train` folder of the
project root at `logs/2019-08-05T18:55:20_hello_tfe/`. By default, checkpoints
are written after each epoch, or when an exception is encountered, including
a `KeyboardInterrupt`. The checkpoint frequency can be adjusted with a
`ckpt_freq: <frequency>` entry in the config file. All config file entries can
also be specified on the command line as, e.g., `--ckpt_freq <frequency>`.

#### Interrupt and Resume
Use `CTRL-C` to interrupt the training:


    [INFO] [LambdaCheckpointHook]: Saved model to logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt

To resume training, run


    edflow -t template_tfe/config.yaml -p logs/2019-08-05T18:55:20_hello_tfe/


It will load the last checkpoint in the project folder and continue training
and logging into the same folder.
This lets you easily adjust parameters without having to start training from
scratch, e.g.


    edflow -t template_tfe/config.yaml -p logs/2019-08-05T18:55:20_hello_tfe/ --batch_size 32


will continue with an increased batch size. Instead of loading the latest
checkpoint, you can load a specific checkpoint by adding `-c <path to
checkpoint>`:


    edflow -t template_tfe/config.yaml -p logs/2019-08-05T18:55:20_hello_tfe/ -c logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt


#### Evaluate
Evaluation mode will write all outputs of `eval_op` to disk and prepare them
for consumption by your evaluation functions. Just replace `-t` by `-e`:


    edflow -e template_tfe/config.yaml -p logs/2019-08-05T18:55:20_hello_tfe/ -c logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt


If `-c` is not specified, it will evaluate the latest checkpoint. The
evaluation mode will finish with

```
[INFO] [EvalHook]: All data has been produced. You can now also run all callbacks using the following command:
edeval -c logs/2019-08-05T18:55:20_hello_tfe/eval/2019-08-05T19:22:23/1207/model_output.csv -cb <your callback>
```

Your callbacks will get the path to the evaluation folder, the input dataset as
seen by your model, an output dataset which contains the corresponding outputs
of your model and the config used for evaluation. `template_tfe/edflow.py`
contains an example callback computing the average loss and accuracy:

```python
def acc_callback(root, data_in, data_out, config):
    from tqdm import trange

    logger = get_logger("acc_callback")
    correct = 0
    seen = 0
    loss = 0.0
    for i in trange(len(data_in)):
        labels = data_in[i]["class"]
        outputs = data_out[i]["outputs"]
        loss = data_out[i]["loss"].squeeze()

        prediction = np.argmax(outputs, axis=0)
        correct += labels == prediction
        loss += loss
    logger.info("Loss: {}".format(loss / len(data_in)))
    logger.info("Accuracy: {}".format(correct / len(data_in)))
```

which can be executed with:


```
$ edeval -c logs/2019-08-05T18:55:20_hello_tfe/eval/2019-08-05T19:22:23/1207/model_output.csv -cb template_tfe.edflow.acc_callback
...
INFO:acc_callback:Loss: 0.00013115551471710204
INFO:acc_callback:Accuracy: 0.7431
```

### PyTorch

The same example as implemented by [TensorFlow Eager](#TensorFlow-Eager), can
be found for PyTorch in `template_pytorch/edflow.py` and requires only slightly
different syntax:

```python
import functools
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
        inputs = inputs.transpose(2, 3).transpose(1, 2)
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
                "images": {"inputs": inputs.detach().numpy()},
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
                "loss": np.array(loss.detach().numpy())[:, None],
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
```

You can experiment with it in the exact same way as [above](#TensorFlow-Eager).
For example, to [start training](#Train) run:


    edflow -t template_tfe/config.yaml -n hello_pytorch


See also [interrupt and resume](#interrupt-and-resume) and
[evaluation](#Evaluate).


### TensorFlow Graph-Building

edflow also supports graph-based execution, e.g.

    cd examples
    edflow -t mnist_tf/train.yaml -n hello_tensorflow

With TensorFlow 2.x going eager by default and TensorFlow 1.x supporting eager
execution, support for TensorFlow's 1.x graph
building will fade away.




## Documentation

For more information, look into our [documentation](https://edflow.readthedocs.io/en/latest/).


## Command-Line Parameters
    
```
$ edflow --help
usage: edflow [-h] [-n description]
              [-b [base_config.yaml [base_config.yaml ...]]] [-t config.yaml]
              [-e [config.yaml [config.yaml ...]]] [-p PROJECT]
              [-c CHECKPOINT] [-r] [-log LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -n description, --name description
                        postfix of log directory.
  -b [base_config.yaml [base_config.yaml ...]], --base [base_config.yaml [base_config.yaml ...]]
                        Path to base config. Any parameter in here is
                        overwritten by the train of eval config. Useful e.g.
                        for model parameters, which stay constant between
                        trainings and evaluations.
  -t config.yaml, --train config.yaml
                        path to training config
  -e [config.yaml [config.yaml ...]], --eval [config.yaml [config.yaml ...]]
                        path to evaluation configs
  -p PROJECT, --project PROJECT
                        path to existing project
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        path to existing checkpoint
  -r, --retrain         reset global step
  -log LEVEL, --log-level LEVEL
                        Set the std-out logging level.
```


## Contributions
[![GitHub-Commits][GitHub-Commits]](https://github.com/pesser/edflow/graphs/commit-activity)
[![GitHub-Issues][GitHub-Issues]](https://github.com/pesser/edflow/issues)
[![GitHub-PRs][GitHub-PRs]](https://github.com/pesser/edflow/pulls)
[![GitHub-Status][GitHub-Status]](https://github.com/pesser/edflow/releases)
[![GitHub-Stars][GitHub-Stars]](https://github.com/pesser/edflow/stargazers)
[![GitHub-Forks][GitHub-Forks]](https://github.com/pesser/edflow/network)
[![GitHub-Updated][GitHub-Updated]](https://github.com/pesser/edflow/pulse)

## LICENSE
 
[![LICENSE][LICENSE]](https://raw.githubusercontent.com/pesser/edflow/master/LICENSE)

## Authors

Mimo Tilbich [![GitHub-Contributions][GitHub-Contributions]](https://github.com/pesser/edflow/graphs/contributors)


[GitHub-Status]: https://img.shields.io/github/tag/pesser/edflow.svg?maxAge=86400&logo=github&logoColor=white
[GitHub-Forks]: https://img.shields.io/github/forks/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Stars]: https://img.shields.io/github/stars/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Commits]: https://img.shields.io/github/commit-activity/y/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Issues]: https://img.shields.io/github/issues-closed/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-PRs]: https://img.shields.io/github/issues-pr-closed/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Contributions]: https://img.shields.io/github/contributors/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Updated]: https://img.shields.io/github/last-commit/pesser/edflow/master.svg?logo=github&logoColor=white&label=pushed

[LICENSE]: https://img.shields.io/github/license/pesser/edflow.svg

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/pesser"><img src="https://avatars3.githubusercontent.com/u/2175508?v=4" width="100px;" alt="Patrick Esser"/><br /><sub><b>Patrick Esser</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=pesser" title="Code">💻</a> <a href="#ideas-pesser" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-pesser" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/jhaux"><img src="https://avatars0.githubusercontent.com/u/9572598?v=4" width="100px;" alt="Johannes Haux"/><br /><sub><b>Johannes Haux</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=jhaux" title="Code">💻</a> <a href="https://github.com/pesser/edflow/commits?author=jhaux" title="Documentation">📖</a> <a href="#ideas-jhaux" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/rromb"><img src="https://avatars1.githubusercontent.com/u/38811725?v=4" width="100px;" alt="rromb"/><br /><sub><b>rromb</b></sub></a><br /><a href="#tutorial-rromb" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/ArWeHei"><img src="https://avatars2.githubusercontent.com/u/46443020?v=4" width="100px;" alt="arwehei"/><br /><sub><b>arwehei</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=ArWeHei" title="Documentation">📖</a> <a href="#infra-ArWeHei" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="http://sandrobraun.de"><img src="https://avatars0.githubusercontent.com/u/6517465?v=4" width="100px;" alt="Sandro Braun"/><br /><sub><b>Sandro Braun</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=theRealSuperMario" title="Code">💻</a> <a href="#example-theRealSuperMario" title="Examples">💡</a> <a href="https://github.com/pesser/edflow/commits?author=theRealSuperMario" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://conrad-sachweh.de"><img src="https://avatars0.githubusercontent.com/u/6422533?v=4" width="100px;" alt="Conrad Sachweh"/><br /><sub><b>Conrad Sachweh</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=conrad784" title="Documentation">📖</a> <a href="https://github.com/pesser/edflow/commits?author=conrad784" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/mritv"><img src="https://avatars1.githubusercontent.com/u/39053439?v=4" width="100px;" alt="Ritvik Marwaha"/><br /><sub><b>Ritvik Marwaha</b></sub></a><br /><a href="#example-mritv" title="Examples">💡</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->
Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
   source/source_files/edflow

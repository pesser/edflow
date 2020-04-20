edflow
======

A framework independent engine for training and evaluating in batches.

Table of Contents
-----------------

1. `Installation <#Installation>`__
2. `Getting Started <#Getting-Started>`__

   1. `TensorFlow Eager <#TensorFlow-Eager>`__
   2. `PyTorch <#PyTorch>`__

Installation
------------

::

   git clone https://github.com/pesser/edflow.git
   cd edflow
   pip install .

Getting Started
---------------

::

   cd examples

TensorFlow Eager
~~~~~~~~~~~~~~~~

You provide an implementation of a model and an iterator and use
``edflow`` to train and evaluate your model. An example can be found in
``template_tfe/edflow.py``:

.. code:: python

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
               # the default logger understands the keys "images" (written as png
               # in the log directory) and "scalars" (written to stdout and the
               # log file).
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
               # values under "labels" are written into a single file,
               # remaining values are written into one file per example.
               # Here, "outputs" would be small enough to write into labels, but
               # for illustration we do not write it as labels.
               return {"outputs": np.array(outputs), "labels": {"loss": np.array(loss)}}

           return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

Specify your parameters in a ``yaml`` config file, e.g.
``template_tfe/config.yaml``:

.. code:: yaml

   dataset: edflow.datasets.fashionmnist.FashionMNIST
   model: template_tfe.edflow.Model
   iterator: template_tfe.edflow.Iterator
   batch_size: 4
   num_epochs: 2

   n_classes: 10

Train
^^^^^

To start training, specify configuration files with
``-b/--base <config>`` command-line option, use the ``-t/--train`` flag
to enable training mode and, optionally, the ``-n/--name <name>`` option
to more easily find your experiments later on:

::

   $ edflow -b template_tfe/config.yaml -t -n hello_tfe
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

edflow shows the progress of your training and scalar logging values.
The log file, log outputs and checkpoints can be found in the ``train``
folder of the project root at ``logs/2019-08-05T18:55:20_hello_tfe/``.
By default, checkpoints are written after each epoch, or when an
exception is encountered, including a ``KeyboardInterrupt``. The
checkpoint frequency can be adjusted with a ``ckpt_freq: <frequency>``
entry in the config file. All config file entries can also be specified
on the command line as, e.g., ``--ckpt_freq <frequency>``.

Interrupt and Resume
^^^^^^^^^^^^^^^^^^^^

Use ``CTRL-C`` to interrupt the training:

::

   [INFO] [LambdaCheckpointHook]: Saved model to logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt

To resume training, run

::

   edflow -b template_tfe/config.yaml -t -p logs/2019-08-05T18:55:20_hello_tfe/

It will load the last checkpoint in the project folder and continue
training and logging into the same folder. This lets you easily adjust
parameters without having to start training from scratch, e.g.

::

   edflow -b template_tfe/config.yaml -t -p logs/2019-08-05T18:55:20_hello_tfe/ --batch_size 32

will continue with an increased batch size. Instead of loading the
latest checkpoint, you can load a specific checkpoint by adding
``-c <path to checkpoint>``:

::

   edflow -b template_tfe/config.yaml -t -p logs/2019-08-05T18:55:20_hello_tfe/ -c logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt

Evaluate
^^^^^^^^

Evaluation mode will write all outputs of ``eval_op`` to disk and
prepare them for consumption by your evaluation functions. Just remove
the training flag ``-t``:

::

   edflow -b template_tfe/config.yaml -p logs/2019-08-05T18:55:20_hello_tfe/ -c logs/2019-08-05T18:55:20_hello_tfe/train/checkpoints/model-1207.ckpt

If ``-c`` is not specified, it will evaluate the latest checkpoint. The
evaluation mode will finish with

::

   [INFO] [EvalHook]: All data has been produced. You can now also run all callbacks using the following command:
   edeval -c logs/2019-08-05T18:55:20_hello_tfe/eval/2019-08-05T19:22:23/1207/model_output.csv -cb <name>:<your callback>

Your callbacks will get the path to the evaluation folder, the input
dataset as seen by your model, an output dataset which contains the
corresponding outputs of your model and the config used for evaluation.
``template_tfe/edflow.py`` contains an example callback computing the
average loss and accuracy:

.. code:: python

   def acc_callback(root, data_in, data_out, config):
       from tqdm import trange

       logger = get_logger("acc_callback")
       correct = 0
       seen = 0
       # labels are loaded directly into memory
       loss1 = np.mean(data_out.labels['loss'])
       loss2 = 0.0
       for i in trange(len(data_in)):
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

which can be executed with:

::

   $ edeval -c logs/2019-08-05T18:55:20_hello_tfe/eval/2019-08-05T19:22:23/1207/model_output.csv -cb tfe_cb:template_tfe.edflow.acc_callback
   ...
   INFO:acc_callback:Loss1: 0.4174468219280243
   INFO:acc_callback:Loss2: 0.4174468546746697
   INFO:acc_callback:Accuracy: 0.8484

PyTorch
~~~~~~~

The same example as implemented by `TensorFlow
Eager <#TensorFlow-Eager>`__, can be found for PyTorch in
``template_pytorch/edflow.py`` and requires only slightly different
syntax:

.. code:: python

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
                   "images": {
                       "inputs": inputs.detach().permute(0, 2, 3, 1).numpy()
                   },
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

You can experiment with it in the exact same way as
`above <#TensorFlow-Eager>`__. For example, to `start
training <#Train>`__ run:

::

   edflow -b template_pytorch/config.yaml -t -n hello_pytorch

See also `interrupt and resume <#interrupt-and-resume>`__ and
`evaluation <#Evaluate>`__.

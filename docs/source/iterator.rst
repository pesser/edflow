
Iterators
=========

Iterators are the 'main hub' in EDFlow.
They combine all other elements and manage the actual workflow.

You may have noticed that iterators are sometimes also called 'Trainers'.
That's because the Iterator actually trains the model during the training phase.
Afterwards the evaluator also qualifies as a more or less altered iterator.

The iterators ``init`` **must** include:

- initialisation of the model
- initialisation of the hooks and extension to the list of hooks
- ``super().__init__(*args, **kwargs)`` to invoke the ``__init__`` of its parent
-  a ``step_ops()`` method that return a list of operations to execute on the feeds.

For machine learning purposes the ``step_ops`` method should always return a
``train_ops``` operation which calculates losses for the optimizer and returns the
loss score.

Training logic should be implemented in the ``run(fetches, feed_dict)`` method.
For instance, alternating training steps for GANs can be achieved by adding/removing
the respective training operations from ``fetches``.
Many more possibilities, like exchanging the optimizer etc. are imaginable.

EDFlow provides a number of iterators out of the box, that feature most tools
usually needed.

- PyHookedModelIterator
- TFHookedModelIterator
- TFBaseTrainer
- TFBaseEvaluator
- TFFrequencyTrainer
- TFListTrainer
- TorchHookedModelIterator


Epochs and Global Step
----------------------
In one epoch you iterate through your whole dataset.
If your specify a number of training steps then ``EDflow`` will run as many
epochs as possible with the given dataset but not finish an epoch if the desired
training step is reached.
``num_steps`` trumps ``num_epochs``

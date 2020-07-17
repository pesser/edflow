
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


Accessing a Set of Fixed Examples
---------------------------------

Sometimes it is useful to log the behavior of the model on a fixed batch of examples.
To do this, one can access a set of fixed examples through ``iterators.template_iterator.TemplateIterator.get_fixed_examples``.
The indices of the fixed examples are chosen randomly on the length of the dataset by default.
One can specify a list of fixed indices through the config in the following way.

.. code-block:: yaml

    fixed_example_indices: 
        train: [0, 1, 2, 3]
        validation: [0, 1, 2, 3]

From within the iterator, the fixed examples can be access as follows

.. code-block:: python

    def step_op(self, model, **kwargs):
        def log_op():
            fixed_examples = self.get_fixed_examples("validation")
            # Do fancy stuff
            logs["fixed_metric"] = foo_bar(fixed_examples)

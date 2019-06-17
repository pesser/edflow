
Models
======

Models can, but may not be your way to set up your machine learning model.
For simple feed-forward networks it is a good idea to implement them as a model
class (inherited from ``object``) with simple ``input`` and ``output`` methods.
Usually the whole actual model in between is defined in the ``__init__`` method.

The iterator then takes the model as one of its arguments and adds the optimizer
logic to the respective model.
This allows for easy exchange between models that only requires changing one line
of code in the ``config.yaml``.

More advanced models that may require to reuse parts of the model should only
define the architecture but leave the inputs and outputs to the iterator.

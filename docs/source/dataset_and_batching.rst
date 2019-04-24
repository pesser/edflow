
Datasets and Batching
=====================

The simplest dataset to think of is a ``class`` that gives a ``dict`` as examples.
The ``dict`` then contains a pair consisting of an example(i.e. an image) and a label (see MNIST examples).

It is also possible to specify other keys and more complex examples as long as your model can handle them correctly.

Batches are automatically created in any size you specify in the config.
It combines multiple examples in such a way that they work with the model.
If your model runs with batch size 1 it also runs with any other batch size.

A few examples are given here:

.. note::
    put some examples here


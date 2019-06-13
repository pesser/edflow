
Iterators
=========

PyIterator
----------
The PyIterator makes no assumption about the  framwork whatsoever.

TFIterator
----------
The TFIterator needs to respect some tensorflow concepts such as
.. note::
   some concepts


Epochs and Global Step
----------------------
In one epoch you iterate through your whole dataset.
If your specify a number of training steps then ``EDflow`` will run as many
epochs as possible with the given dataset but not finish and epoch if the desired
training step is reached.

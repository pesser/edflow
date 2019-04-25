
What Happens When I Run ``EDflow``
==================================

When you successfully build your model start ``EDflow`` with::
    edflow -t your_model/train.yaml

``EDflow`` then loads your config with all parameters you specified.
Then all objects are constructed with the config and other needed objects.
``EDflow`` then runs the iterator for each iteration in training.


 
.. note::
    here goes a nice gif of edflow in action

What Happens When I Run EDflow
==================================


``config``
---------
At the heart of every training or evaluation is the **config** file.
It is a ``dict`` that contains the keywords and values you specify in train.yaml.
Some keys are mandatory, like:
- ``dataset`` package link to the data set class
- ``model`` package link to the model class
- ``iterator`` package link to the iterator class
- ``batch_size`` how large a batch should be
- ``num_steps`` or ``num_epochs`` how long should the training be

EDFlow is able to handle multiple config files but typically it is recommended to have a base config file, which is included with the ``-b`` option and separate training and evaluation configs can be included on top of that, if needed.

- Test_mode is set to true (e.g. for dropout)

Workflow
--------

When you have successfully built your model your model with::

    edflow -t your_model/train.yaml

This triggers EDFlow's signature workflow:

1. The ``ProjectManager`` is initialized

  - It creates the folder structure, takes a snapshot of the code and keeps track directory addresses through attributes
  - It is still to decide on the best way to take the snapshot, feel free to participate and contribute_

2. All processes are initialized

  - if ``-t`` option is given, a training process is started
  - for each ``-e`` option an evaluation process is called

3. The training process

  - ``Logger`` is initialized
  - ``Dataset`` is initialized
  - The batches are built
  - ``model`` is initialized

    - #TODO initialize a dummy if no model is given

  - ``Trainer``/``Iterator`` is initialized
  - if ``--checkoint`` is given, load checkpoint
  - If ``--retrain`` is given, reset global step (begin training with pre-trained model)
  - ``Iterator.iterate`` is called

    - This is the data loop, only argument is the batched data
    - tqdm_ is called: for epoch in epochs, for batch in batches
    - initialize ``fetches``

      - nested ``dict``
      - leaves must be functions i.e. ``{global_step:get_global_step()}``

    - ``feeds`` are initialized as a copy of batch (this allows to manipulate the feed)
    - all ``hook`` s' ``before_step(global_step, fetches, feeds, batch)`` is called

      - ``hook`` s can add data, manipulate feeds(i.e. make numpy arrays tf objects), log batch data...

    - ``self.run(fetches, feeds)`` is called

      - every function in fetches is called with feeds as argument

    - ``global_step`` is incremented
    - all ``hook`` s' ``after_step(global_step, fetches, feeds, batch)`` is called

 
.. note::
    here goes a nice gif of edflow in action

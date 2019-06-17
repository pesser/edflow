
Hooks
=====

Hooks are a distinct EDFlow feature.
You can think of them as plugins for your trainer.

Each Hook is inherited from ``edflow.hooks.hook.Hook`` or one of it's inherited classes.
It contains methods for different parts of a training loop:

- ``before_epoch(epoch)``
- ``before_step(step, fetches, feeds, batch)``
- ``after_step(step, last_results)``
- ``after_epoch(epoch)``
- ``at_exception(exception)``

Coming soon:

- ``before_training``
- ``after_training``

EDFlow already comes with a number of hooks that allow for conversion of arrays to
tensors, save checkpoints, call other hooks on intervals, log your data...
All of this functionality can be expanded and transferred easily between projects
which is one of the main assets of EDFlow.

In order to add a hook to your iterator simply expand the list of current hooks
(some come 'pre-installed' with an iterator) like that::

    self.hooks += [hook, another_hook, so_many_hooks]

after you initialized each hook with its respective parameters.

Once you seized the concept of hooks, they really are one of EDFlows greatest
tools and come with all the advantages of modularity.

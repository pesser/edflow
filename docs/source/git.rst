Integrations
============

git
---

Git integration can be enabled with the config parameter ``--integrations/git
True``.  This assumes that you are starting ``edflow`` in a directory which is
part of a git repository. For every run of ``edflow``, git integration amounts
to creating a tagged commit that contains a snapshot of all ``.py`` and ``.yaml``
files found under ``code_root``, and all git tracked files at the time of the
run. The name of the tag can be found as ``git_tag: <tag>`` in the ``log.txt`` of
the run directory. You can get an overview of all tags with ``git tag``. This
allows you to easily compare your working directory to the code used in a
previous experiment with ``git diff <tag>`` (git ignores untracked files in the
working directory for its diff, so you might want to add them first), or two
experiments you ran with ``git diff <tag1> <tag2>``.  Furthermore, it allows
you to reproduce or continue training of an old experiment with ``git checkout
<tag>``.

wandb
-----

Weights and biases integration can be enabled with the config parameter
``--integrations/wandb True``. This will log the config and scalar logging
values to weights and biases.

tensorboardX
------------

TensorboardX integration can be enabled with the config parameter
``--integrations/tensorboardX True``. This will log the config and scalar logging
values to tensorboardX.

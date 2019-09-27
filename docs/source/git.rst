Git integration
==================================

Git integration can be enabled by setting the environment variable
``EDFLOW_GIT``, e.g.

    export EDFLOW_GIT=1

This assumes that you are starting ``edflow`` in a directory which is part of a
git repository. For every run of ``edflow``, git integration amounts to
creating a tagged commit that contains a snapshot of all tracked files at the
time of the run. You can get an overview of all tags with ``git tag``. This
allows you to easily compare your working directory to the code used in a
previous experiment with ``git diff <timestamp>_<name>`` or two experiments you
ran with ``git diff <timestamp1>_<name1> <timestamp2>_<name2>``. Furthermore,
it allows you to reproduce or continue training of an old experiment with ``git
checkout <timestamp>_<name>``.

Note however that only tracked files are included in the commit (we run
``git add -u`` in the background) and no new files are added. It is your
responsibility to make sure required files are tracked.

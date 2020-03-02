
FAQ
=====

How do I set a random seed?
   Iterators or models or datasets can use a random seed from
   the config. How and where to set such seeds is application
   specific. It is recommended to create local pseudo-random-number-generators
   whenever possible, e.g. using `RandomState
   <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html>`_
   for `numpy`.

   Note that loading examples from a dataset happens in multiple processes, and
   the same random seed is copied to all child processes. If your
   :meth:`edflow.data.dataset.DatasetMixin.get_example` method relies on random
   numbers, you should use :class:`edflow.util.PRNGMixin` to make sure examples
   in your batches are independent. This will add a :attr:`prng` attribute (a
   `RandomState` instance) to your class, which will be seeded differently in
   each process.

How do I run tests locally?
   We use `pytest <https://docs.pytest.org/en/latest/>`_ for our tests and you
   can run ``pytest --ignore="examples"`` to run the general tests. To run
   framework dependent tests and see the precise testing protocol executed by
   `travis <https://travis-ci.org/>`_, see `.travis.yml`.

Why can't my implementations be imported?
   In general, it is your responsibility to make sure `python` can import your
   implementations (e.g. install your implementations or add their location to
   your `PYTHONPATH`). To support the common practice of executing `edflow` one
   directory above your implementations, we add the current working directory
   to `python`'s import path.

   For example, if `/a/b/myimplementations/c/d.py` contains your `MyModel`
   class, you can specify `myimplementations.c.d.MyModel` for your `model`
   config parameter if you run edflow in `a/b/`.

Why is my code not copied to the log folder?
   You can always specify the path to your code to copy with the `code_root`
   config option. Similar to how implementations are found (see previous
   question), we support the common practice of executing `edflow` one
   directory above your implementations.

   For example, if `/a/b/myimplementations/c/d.py` contains your `MyModel`
   class and you specify `myimplementations.c.d.MyModel` for your `model`
   config parameter, `edflow` will use `$(pwd)/myimplementations` as the code
   root which assumes you are executing `edflow` in `/a/b`.

How can I kill edflow zombie processes?
   You can use `edlist` to show all edflow processes. All sub-processes share
   the same process group id (`pgid`), so you can easily send all of them a
   signal with `kill -- -<pgid>`.

How do I set breakpoints? `import pdb; pdb.set_trace()` is not working.
   Use `import edflow.fpdb as pdb; pdb.set_trace()` instead. `edflow` runs
   trainings and evaluations in their own processes. Hence, `sys.stdin` must be
   set properly to be able to interact with the debugger.

Error when using TFE: `NotImplementedError: object proxy must define reduce_ex()`.
   This was addressed in this issue : https://github.com/pesser/edflow/issues/240
   When adding the config to a model that inherits from `tf.keras.Model`, the config cannot be dumped.
   It looks like keras changes lists within the config to a `ListWrapper` object, which are not reducable by `yaml.dump`

   Workaround 
      is to simply not do self.config = config and save everything you need in a field in the model.
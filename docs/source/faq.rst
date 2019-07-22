
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

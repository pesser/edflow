Contributions
*************

If you have any new applications that require custom hooks or iterators feel
free to contribute at any time.

``EDflow`` is continuously expanded and gains new capabilities with every use.
Examples of models are always welcome and we are happy if want to contribute in
any way.

We are working on github and celebrate every pull request.

black
-----

Before requesting a pull please run black_ for better code style or simply add
black_ to your pre-commit hook:

0. Install black_ with ::

   $ pip install black

1. Paste the following into at the top <project-root>/.git/hooks/pre-commit.sample::

   # run black on all staged files
   staged=$(git diff --name-only --cached)
   black $staged
   # add them again after formatting
   git add $staged

2. Rename ``pre-commit.sample`` to ``pre-commit``
3. Make it executable using::

   $ chmod +x pre-commit

4. Done!

Or run black by hand and use this command before every commit:::

    black ./


Continuous Integration
----------------------

We use travisCI_ for continuous integration.
You do not need to worry about as long as your code passes all tests (this includes
a formatting test with black).

.. note::

    this should include an example to run the tests locally as well

Documenation
------------

   This is a short summary how the documentation works and how it can be built

The documentation uses sphinx_ and is available under readthedocs.org_.
It also uses all-contributors_ for honoring contributors.

sphinx
======

To build the documentation locally, install `sphinx` and run:::

    $ cd docs
    $ make html

The html files are available under the then existing directory ``docs/_build/html/``

The docsting format which is preferred is `numpy`.

We use `sphinx-apidoc` to track all files automatically:::

    $ cd docs
    $ sphinx-apidoc -o ./source/source_files ../edflow

all-contirbutors
================

We use all-contributors locally and manage the contributors by hand.

To do so, install `all-contributors` as described here (we advise you to install it inside the repo but unstage the added files).
Then run the following command to add a contributor or contribution:::

    all-contributors add <username> <contribution>

If this does not work for you (sometimes with npm the case) use:::

    ./node_modules/.bin/all-contributors add <username> <contribution>

Known Issues
------------

We noticed that mocking `numpy` in ``config.py`` will not work due to some requirements when importing numpy in EDFlow.
Thus we need to require numpy when building the documentation.

Locally, this means that you need to have numpy installed in your environment.

Concerning ``readthedocs.org``, this means that we require a ``readthedocs.yml`` in the source directory which points to ``extra_requirements`` in ``setup.py``, where numpy is a dependency.
Other dependencies are `sphinx` and `sphinx_rtd_theme`.


.. image:: https://img.shields.io/github/commit-activity/y/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/commit-activity/y/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-Commits
 <https://github.com/pesser/edflow/graphs/commit-activity>

.. image:: https://img.shields.io/github/issues-closed/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/issues-closed/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-Issues
 <https://github.com/pesser/edflow/issues>

.. image:: https://img.shields.io/github/issues-pr-closed/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/issues-pr-closed/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-PRs
 <https://github.com/pesser/edflow/pulls>

.. image:: https://img.shields.io/github/tag/pesser/edflow.svg?maxAge=86400&logo=github&logoColor=white
   :target: https://img.shields.io/github/tag/pesser/edflow.svg?maxAge=86400&logo=github&logoColor=white
   :alt: GitHub-Status
 <https://github.com/pesser/edflow/releases>

.. image:: https://img.shields.io/github/stars/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/stars/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-Stars
 <https://github.com/pesser/edflow/stargazers>

.. image:: https://img.shields.io/github/forks/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/forks/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-Forks
 <https://github.com/pesser/edflow/network>

.. _black: https://github.com/ambv/black

.. _readthedocs.org: https://edflow.readthedocs.io/en/latest/

.. _all-contributors: https://allcontributors.org

.. _travisCI: https://travis-ci.org/pesser/edflow/

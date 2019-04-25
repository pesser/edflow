
Contributions
=============

If you have any new applications that require custom hooks or iterators feel
free to contribute at any time.

``EDflow`` is continuously expanded and gains new capabilities with every use.
Examples of models are always welcome and we are happy if want to contribute in
any way.

We are working on github and celebrate every pull request.

Before requesting a pull please run black_ for better code style or simply add
black_ to your pre-commit hook:

0. Install black_ with
::
   $ pip install black
1. Paste the following into at the top <project-root>/.git/hooks/pre-commit.sample
::
   # Run black on all staged files
   staged=$(git diff --name-only --cached)
   black $staged
   # Add them again after formatting
   git add $staged
2. Rename ``pre-commit.sample`` to ``pre-commit``
3. Make it excutable using
::
   $ chmod +x pre-commit
4. Done!


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

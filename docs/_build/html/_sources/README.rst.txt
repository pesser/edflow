Setup
-----

There are two ways for setting up **EDFlow** on your system:


#. 
   Use PyPI:

    Recommended: Install edflow into a conda environment.

   .. code-block::

       conda create --name myenv python=3.6
       source activate myenv

    Pull and install the current in the current directory with PyPi
    ``pip install -e git+https://github.com/pesser/edflow.git``

#. 
   Use ``setup.py``\ :

    Pull repository
    ``git clone https://github.com/pesser/edflow.git``
    ``cd edflow``
    In edflow directory
    ``python3 setup.py``

Workflow
--------

For more information, look into our `starter guide <link>`_.

Example
-------

Tensorflow
^^^^^^^^^^

Pytorch
^^^^^^^

Other
-----

Parameters
^^^^^^^^^^


* 
  ``--config path/to/config``

    yaml file with all information see [Workflow][#Workflow]

* 
  ``--checkpoint path/to/checkpoint to restore``

* 
  ``--noeval``
    only run training

* 
  ``--retrain``
    reset global step to zero

Known Issues
^^^^^^^^^^^^

Compatibility
^^^^^^^^^^^^^

Contributions
-------------

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

LICENSE
-------

coming soon…

Authors
-------

Mimo Tilbich

.. image:: https://img.shields.io/github/contributors/pesser/edflow.svg?logo=github&logoColor=white
   :target: https://img.shields.io/github/contributors/pesser/edflow.svg?logo=github&logoColor=white
   :alt: GitHub-Contributions
 <https://github.com/pesser/edflow/graphs/contributors>

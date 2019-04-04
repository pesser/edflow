# EDFlow - Evaluation Driven workFlow

A small framework for training and evaluating tensorflow models by Mimo Tilbich.

## Table of Contents
1. [Setup](#Setup)
2. [Workflow](#Workflow)
3. [Example](#Example)
4. [Other](#Other)
    1. [Parameters](#Parameters)
    2. [Known Issues](#Known Issues)
    3. [Compatibility](#Compatibility)
5. [Contributions](#Contributions)
6. [LICENSE](#LICENSE)
7. [Authors](#Authors)

## Setup

There are two ways for setting up **EDFlow** on your system:

1. Use PyPI:

    Recommended: Install edflow into a conda environment.
    ```
    conda create --name myenv python=3.6
    source activate myenv
    ```
    
    Pull and install the current in the current directory with PyPi
    `pip install -e git+https://github.com/pesser/edflow.git`

2. Use `setup.py`:

    Pull repository
    `git clone https://github.com/pesser/edflow.git`
    `cd edflow`
    In edflow directory
    `python3 setup.py`

## Workflow

For more information, look into our [starter guide](link).


## Example

### Tensorflow

### Pytorch


## Other

### Parameters

- `--config path/to/config`

    yaml file with all information see [Workflow][#Workflow]

- `--checkpoint path/to/checkpoint to restore`

- `--noeval`
    only run training

- `--retrain`
    reset global step to zero

### Known Issues

### Compatibility

## Contributions
[![GitHub-Commits][GitHub-Commits]](https://github.com/pesser/edflow/graphs/commit-activity)
[![GitHub-Issues][GitHub-Issues]](https://github.com/pesser/edflow/issues)
[![GitHub-PRs][GitHub-PRs]](https://github.com/pesser/edflow/pulls)
[![GitHub-Status][GitHub-Status]](https://github.com/pesser/edflow/releases)
[![GitHub-Stars][GitHub-Stars]](https://github.com/pesser/edflow/stargazers)
[![GitHub-Forks][GitHub-Forks]](https://github.com/pesser/edflow/network)
[![GitHub-Updated][GitHub-Updated]](https://github.com/pesser/edflow/pulse)
  
## LICENSE
 
[![LICENSE][LICENSE]](https://raw.githubusercontent.com/pesser/edflow/master/LICENSE)

## Authors

Mimo Tilbich [![GitHub-Contributions][GitHub-Contributions]](https://github.com/pesser/edflow/graphs/contributors)


[GitHub-Status]: https://img.shields.io/github/tag/pesser/edflow.svg?maxAge=86400&logo=github&logoColor=white
[GitHub-Forks]: https://img.shields.io/github/forks/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Stars]: https://img.shields.io/github/stars/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Commits]: https://img.shields.io/github/commit-activity/y/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Issues]: https://img.shields.io/github/issues-closed/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-PRs]: https://img.shields.io/github/issues-pr-closed/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Contributions]: https://img.shields.io/github/contributors/pesser/edflow.svg?logo=github&logoColor=white
[GitHub-Updated]: https://img.shields.io/github/last-commit/pesser/edflow/master.svg?logo=github&logoColor=white&label=pushed

[LICENSE]: https://img.shields.io/github/license/pesser/edflow.svg

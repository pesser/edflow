# EDFlow - Evaluation Driven workFlow

A small framework for training and evaluating tensorflow models by Mimo Tilbich.

## Table of Contents
1. [Setup](#Setup)
2. [Workflow](#Workflow)
3. [Example](#Example)
4. [Other](#Other)
    1. [Parameters](#Parameters)
    2. [Known Issues](#Known-Issues)
    3. [Compatibility](#Compatibility)
5. [Contributions](#Contributions)
6. [LICENSE](#LICENSE)
7. [Authors](#Authors)

## Setup
Clone this repository:

    git clone https://github.com/pesser/edflow.git
    cd edflow

We provide different [conda](https://conda.io) environments in the folder
`environments`:

- `edflow_tf_cu9.yaml`: Use if you have `CUDA>=9` available and
  want to use tensorflow.
- `edflow_pt_cu9.yaml`: Use if you have `CUDA>=9` available and
  want to use pytorch.
- `edflow_cpu`: Use if you don't have a `CUDA>=9` GPU available.

Choose an appropriate environment and execute

    conda env create -f environments/<env>.yaml
    conda activate <env>
    pip install -e .

where `<env>` is one of the `yaml` files described above.


## Workflow

For more information, look into our [starter guide](link).


## Example

### Tensorflow

    cd examples
    edflow -t mnist_tf/train.yaml -n hello_tensorflow


### Pytorch

    cd examples
    edflow -t mnist_pytorch/mnist_config.yaml -n hello_pytorch


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

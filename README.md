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

For more information, look into our [documentation](https://edflow.readthedocs.io/en/latest/).


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

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/pesser"><img src="https://avatars3.githubusercontent.com/u/2175508?v=4" width="100px;" alt="Patrick Esser"/><br /><sub><b>Patrick Esser</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=pesser" title="Code">💻</a> <a href="#ideas-pesser" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tutorial-pesser" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/jhaux"><img src="https://avatars0.githubusercontent.com/u/9572598?v=4" width="100px;" alt="Johannes Haux"/><br /><sub><b>Johannes Haux</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=jhaux" title="Code">💻</a> <a href="https://github.com/pesser/edflow/commits?author=jhaux" title="Documentation">📖</a> <a href="#ideas-jhaux" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/rromb"><img src="https://avatars1.githubusercontent.com/u/38811725?v=4" width="100px;" alt="rromb"/><br /><sub><b>rromb</b></sub></a><br /><a href="#tutorial-rromb" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/ArWeHei"><img src="https://avatars2.githubusercontent.com/u/46443020?v=4" width="100px;" alt="arwehei"/><br /><sub><b>arwehei</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=ArWeHei" title="Documentation">📖</a> <a href="#infra-ArWeHei" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="http://sandrobraun.de"><img src="https://avatars0.githubusercontent.com/u/6517465?v=4" width="100px;" alt="Sandro Braun"/><br /><sub><b>Sandro Braun</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=theRealSuperMario" title="Code">💻</a> <a href="#example-theRealSuperMario" title="Examples">💡</a> <a href="https://github.com/pesser/edflow/commits?author=theRealSuperMario" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://conrad-sachweh.de"><img src="https://avatars0.githubusercontent.com/u/6422533?v=4" width="100px;" alt="Conrad Sachweh"/><br /><sub><b>Conrad Sachweh</b></sub></a><br /><a href="https://github.com/pesser/edflow/commits?author=conrad784" title="Documentation">📖</a> <a href="https://github.com/pesser/edflow/commits?author=conrad784" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/mritv"><img src="https://avatars1.githubusercontent.com/u/39053439?v=4" width="100px;" alt="Ritvik Marwaha"/><br /><sub><b>Ritvik Marwaha</b></sub></a><br /><a href="#example-mritv" title="Examples">💡</a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->
Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
   source/source_files/edflow

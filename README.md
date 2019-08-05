# edflow

A framework independent engine for training and evaluating in batches.

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

## Installation

    git clone https://github.com/pesser/edflow.git
    cd edflow
    pip install .


## Getting started


    cd examples


### TensorFlow eager


```python
```



### TensorFlow graph-building

edflow supports graph-based execution, e.g.

    cd examples
    edflow -t mnist_tf/train.yaml -n hello_tensorflow

With TensorFlow 2.x going eager by default and TensorFlow 1.x supporting eager
execution, support for TensorFlow's 1.x graph
building will fade away.


### Pytorch

    cd examples
    edflow -t mnist_pytorch/mnist_config.yaml -n hello_pytorch



## Documentation

For more information, look into our [documentation](https://edflow.readthedocs.io/en/latest/).


## `edflow` command-line parameters
    
```bash
$ edflow --help
usage: edflow [-h] [-n description]
              [-b [base_config.yaml [base_config.yaml ...]]] [-t config.yaml]
              [-e [config.yaml [config.yaml ...]]] [-p PROJECT]
              [-c CHECKPOINT] [-r] [--nogpu] [-log LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -n description, --name description
                        postfix of log directory.
  -b [base_config.yaml [base_config.yaml ...]], --base [base_config.yaml [base_config.yaml ...]]
                        Path to base config. Any parameter in here is
                        overwritten by the train of eval config. Useful e.g.
                        for model parameters, which stay constant between
                        trainings and evaluations.
  -t config.yaml, --train config.yaml
                        path to training config
  -e [config.yaml [config.yaml ...]], --eval [config.yaml [config.yaml ...]]
                        path to evaluation configs
  -p PROJECT, --project PROJECT
                        path to existing project
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        path to existing checkpoint
  -r, --retrain         reset global step
  --nogpu               disable gpu for tensorflow
  -log LEVEL, --log-level LEVEL
                        Set the std-out logging level.
```


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

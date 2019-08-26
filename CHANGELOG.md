# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- TFMultiStageTrainer and TFMultiStageModel
    - adds interface for model and trainer to derive from to support multiple stages. For example pretraining and training.
    - example can be found in `examples/multistage_trainer`
- `edlist` shows process group id (`pgid`), see FAQ.
- CHANGELOG.md to document notable changes.

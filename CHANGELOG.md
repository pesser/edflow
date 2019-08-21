# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `edlist` shows process group id (`pgid`), see FAQ.
- CHANGELOG.md to document notable changes.

### Changed
- `LambdaCheckpointHook` uses global step and doesn't save on first step.

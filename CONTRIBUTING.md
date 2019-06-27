# Contributing to edflow

The following is a set of guidelines for contributing to edflow. These are mostly guidelines, not rules. Use your best judgment on when to apply them and always be nice to each other :heart:.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[I just have a question!!!](#i-just-have-a-question)

[The Basics](#the-basics)

[How Can I Contribute?](#how-can-i-contribute)

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Code Contribution](#first-code-contribution)
- [Pull Requests](#pull-requests)

[Styleguides](#styleguides)

- [Git Commit Messages](#git-commit-messages)
- [Documentation Styleguide](#documentation-styleguide)
- [Code Styleguide](#code-styleguide)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [pesser@github.com](mailto:pesser@github.com).

## I just have a question!

> **Note:** [Please don't file an issue to ask a question.]

If you have any questions on how to use edflow, feel free to contact Mimo Tillbich.

## The Basics

Before you do anything, we advise you to read the [documentation](https://edflow.readthedocs.io/en/latest/), it's really not that long and helps a lot...

## How Can I Contribute?

First of all we are very happy if you are even considering to contribute. As the developer base and the user base are not that big (some say they are the same people) we welcome everyone who sees the benefits of edflow for their work.

### Reporting Bugs

When you are creating a bug report, please include as many details as possible. Fill out [the required template](/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.
All bugs are tracked as github issues.

Always remember to be decisive when issuing bugs:

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as many details as possible. 
- **Provide specific examples to demonstrate the steps**. This can be code snippets or a link to the whole project folder.
- **Describe the behaviour you observed**
- **Explain which behavior you expected to see instead and why.**
- **Include details about your configuration and environment**. Sometimes the devil is not in the detail but in the environment

### Suggesting Enhancements

As edflow is still work in progress, there is much to add in features, documentation etc.
Feel free to propose enhancements that seem interesting/functional to you. But be patient! It might take a while to implement them.
All enhancement suggestions are tracked as github issues.

As for bugs please be precise in your suggestions:

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
- **Provide specific examples to demonstrate the steps**. This always supports your case as this makes it easier to understand.
- If you want to change existing features, **describe the current behavior** and **explain which behavior you expected to see instead** and why.
- **Explain why this enhancement would be useful** if this is not already clear from the example.

### First Code Contribution

The first step is always the hardest. None of us is Linus Torvalds. This is the first open project for most of us so don't feel intimidated, we are learning, too.
At least try to follow the guidelines you find [here](#contributing-to-edflow)

### Pull Requests

Please follow these steps when contributing:

1. Follow all instructions in [the template](PULL_REQUEST_TEMPLATE.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all [tests](https://travis-ci.org/pesser/edflow/) are passing.

This makes working together easier for all of us.

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- When only changing documentation, include `[ci skip]` in the commit title
- Consider starting the commit message with an applicable emoji:
  - :art: `:art:` when improving the format/structure of the code
  - :racehorse: `:racehorse:` when improving performance
  - :memo: `:memo:` when writing docs
  - :bug: `:bug:` when fixing a bug
  - :fire: `:fire:` when removing code or files
  - :green_heart: `:green_heart:` when fixing tests
  - :white_check_mark: `:white_check_mark:` when adding tests

### Documentation Styleguide

- Use [sphinx](https://www.sphinx-doc.org/en/master/).
- Use [reStructuredText](http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html).

### Code Styleguide

Just remember to use black (This is necessary to pass the tests)



Yours truly,
Mimo tillbich :heart:

# Contributing

Thank you for your interest in contributing to `matchminer-ai`.

Contributions are welcome through GitHub pull requests. If you do not have
write access to the repository, please fork the repository and open a pull
request from your fork.

## Development Setup

Clone the repository and move into the repository root:

```shell
git clone https://github.com/dfci/matchminer-ai-inference.git
cd matchminer-ai-inference
```

Create and activate a virtual environment:

```shell
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode with development dependencies:

```shell
pip install -e ".[dev]"
```

The `dev` dependencies include `pytest`, `pre-commit`, and other local
development tools.

Install the pre-commit hooks:

```shell
pre-commit install
```

The installed pre-commit hooks run automatically when you commit and check the
files included in that commit. The first hook run may download hook
environments managed by pre-commit.

> [!WARNING]
> The hooks include sensitive data checks and are intentionally cautious. Please
> review all flagged items carefully. To bypass the hook after review, use
> `SKIP=gitleaks git commit`.

## Before Opening a Pull Request

Before opening a pull request, please confirm that:

* the installed pre-commit hooks run successfully on your commit or commits;
* the relevant tests pass from the repository root, for example:

```shell
pytest
```

* your changes do not include PHI, PII, credentials, secrets, generated cache
  files, or model outputs containing sensitive data.

## Opening a Pull Request

Open a pull request from your branch or fork to the main `matchminer-ai`
repository.

In the pull request description, please include:

* a brief summary of the changes;
* any relevant context or linked issue;
* the tests or checks you ran; and
* confirmation that the pull request does not include PHI, PII, credentials,
  secrets, generated cache files, or model outputs containing sensitive data.

GitHub Actions runs automated checks on pull requests. These checks include
building the package and running the default test suite. The automated checks
must pass before a pull request can be merged.

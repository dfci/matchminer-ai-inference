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

## Before Opening a Pull Request

Make sure your commits pass the installed pre-commit hooks, then run the
relevant tests from the repository root:

```shell
pytest
```

Do not include PHI, PII, credentials, secrets, generated cache files, or
model outputs containing sensitive data in commits.

GitHub Actions runs automated checks on pull requests. These checks include the
default test suite and must pass before merge.

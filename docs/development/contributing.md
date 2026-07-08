# Contributing

Thank you for your interest in contributing to `matchminer-ai`.

Contributions are welcome through GitHub pull requests. Please fork the
repository and open a pull request from your fork.

## Local Setup

Clone your fork and move into the repository root:

```shell
git clone https://github.com/<your-username>/matchminer-ai-inference.git
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

Install the pre-commit hooks:

```shell
pre-commit install
```

The pre-commit hooks run formatting, linting, type-checking, and sensitive data
checks before commits.

!!! warning
    The hooks include sensitive data checks and are intentionally cautious.
    Review all flagged items carefully. To bypass the hook after review, use
    `SKIP=gitleaks git commit`.

## Before You Open a Pull Request

Run the relevant tests from the repository root:

```shell
pytest
```

Do not include PHI, PII, credentials, secrets, generated cache files, or model
outputs containing sensitive data.

## Pull Request Checks

GitHub Actions runs automated checks on pushes and pull requests. These checks
install the package, run the pre-commit hooks, scan for sensitive data, and run
the test suite. These checks **must** pass before merge.

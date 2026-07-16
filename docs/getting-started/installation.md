# Installation

This package requires Python 3.13+.

We recommend using [`uv`](https://docs.astral.sh/uv/) to create the Python
environment and install the package:

```shell
uv venv --python 3.13
source .venv/bin/activate
uv pip install matchminer-ai
```

For local development or documentation work from a repository checkout, install
the package in editable mode with the relevant optional dependencies:

```shell
uv pip install -e ".[dev]"
uv pip install -e ".[docs]"
```


See [contributing](../development/contributing.md) for the full local
development setup.

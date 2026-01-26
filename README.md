# mmai

MatchMiner-AI Python package.

This repository contains the Python package that implements the MatchMiner-AI
pipeline.

Requires Python 3.12+.

## Development

Clone the repository and work from the repo root:

```sh
git clone https://gitlab.dfci.harvard.edu/ksg/trial-matching/hf-stage/mmai-package.git
cd mmai-package
```

We recommend working in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the package with development dependencies:

```sh
python -m pip install -e ".[dev]"
```

Install optional local LLM dependencies:

```sh
python -m pip install -e ".[local]"
```

Note: the first local LLM run will download large model weights and can be slow.
If you already have the weights cached, point Hugging Face to that location, e.g.:

```sh
export HF_HOME=/path/to/hf_cache
```

## Code quality checks
This repository uses `pre-commit` for local checks at commit time, including:

- Python formatting and linting
- Static type checking
- Basic repository hygiene (whitespace, file size)
- Detection of accidentally committed secrets

To enable the hooks (recommended):
```sh
pre-commit install
```

Hooks run automatically on `git commit`.

## Tests
Run the test suite with:

```sh
pytest
```

GPU integration tests (requires CUDA + model downloads):

```sh
pytest -m resource_heavy
```

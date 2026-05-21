# matchminer-ai

## Overview

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

This package is currently pre-v1 and under active development. APIs, configuration options, and outputs may change.

## Compute requirements

The most compute-intensive step is summarizing patient notes with the language model. Full pipeline runs can use either a local high-memory GPU environment, such as an NVIDIA H100 80GB, or a compatible remote inference server. See the \[example notebook\](examples/run\_examples.ipynb) for details on these two options.

Other entry points, such as running from precomputed summaries, may require less compute.

## Installation

This package requires Python 3.13+. Clone this repository and install the package in editable mode:

```shell
git clone https://github.com/dfci/matchminer-ai-inference.git
cd matchminer-ai-inference
pip install -e .
```

## Quickstart

See the example notebook for a full walkthrough using sample input data:
\[example notebook\](examples/run\_examples.ipynb)

## Citation

If you use `matchminer-ai`, please cite:
Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: 10.48550/arXiv.2412.17228

## Contributing

We recommend working in a virtual or conda environment.
Using `venv`:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This repository uses pre-commit for local code quality checks. To enable the hooks:

```shell
pre-commit install
```

Run the test suite with:

```shell
# lightweight tests
pytest
# tests requiring GPU
pytest -m resource_heavy
```

# matchminer-ai

## Overview

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

This package is currently pre-v1 and under active development. APIs, configuration options, and outputs may change.

## Compute requirements

The most compute-intensive step is summarizing patient notes with the default Gemma 4 language model. Full pipeline runs can use either a local high-memory GPU environment, such as an NVIDIA H100 80GB, or a remote OpenAI-compatible chat completions endpoint. See the [example notebook](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb) for details on these two options.

Other entry points, such as running from precomputed summaries, may require less compute.

## Installation

This package requires Python 3.13+.

The package has been tested in Linux environments.

We recommend using [`uv`](https://docs.astral.sh/uv/) to create the Python
environment and install the package:

```shell
uv venv --python 3.13
source .venv/bin/activate
uv pip install matchminer-ai
```

## Quickstart

See the example notebook for a full walkthrough using sample input data:
[example notebook](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

## Documentation

This repository includes a MkDocs documentation site with tutorials, reference
pages, and generated API docs.

```shell
uv pip install -e ".[docs]"
mkdocs serve
```

The docs source lives in [`docs/`](docs/).

## Citation

If you use `matchminer-ai`, please cite:
>Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

## Contributing

Contributions are welcome! Please follow our
[contribution instructions][contributing] if you are interested in contributing to this project.

[contributing]: https://github.com/dfci/matchminer-ai-inference/blob/main/CONTRIBUTING.md

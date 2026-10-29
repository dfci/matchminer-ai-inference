# matchminer-ai

## Overview

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

For detailed instructions, please see the
[documentation website](https://dfci.github.io/matchminer-ai-inference/).

> [!WARNING]
> This package is currently pre-v1 and under active development.

## Compute requirements

A GPU is recommended for the clinical trial matching inference workflow. Please see the
[requirements documentation](https://dfci.github.io/matchminer-ai-inference/getting-started/requirements/)
for more information on compute expectations and GPU recommendations.

## Installation

This package requires Python 3.12+.

The package has been tested in Linux environments.

We recommend using [`uv`](https://docs.astral.sh/uv/) to create the Python
environment and install the package:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install matchminer-ai
```

## Model Compatibility
This package uses model collections available on HuggingFace at [huggingface.co/ksg-dfci/](https://huggingface.co/ksg-dfci/) starting with collection [0526](https://huggingface.co/collections/ksg-dfci/matchminer-ai-0526). As collections are updated, they will be tied to a specific version of this package. Please see the [user guide](/docs/user-guide/patient-centric-matching.md/#model-collections) for more information.

| Collection | Compatible package versions | Notes |
|------------|-----------------------------|-------|
| 1225 | NA | initial model and synthetic data release |
| 0426 | NA | MatchMiner-AI updates for April 2026 ('v21'), incorporating more quantitative scoring of degree of match between patient and trial space. |
| 0526 | v0.2.0+ | May 2026 MatchMiner-AI updates ('v22'), incorporating switch to Gemma-4-31B-IT , and repetition penalty adjustment to improve patient summarization. |

## Quickstart
See the example notebook for a full walkthrough using sample input data:
[example notebook](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

## Citation

If you use `matchminer-ai`, please cite:
>Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

## Contributing

Contributions are welcome! Please follow our
[contribution instructions][contributing] if you are interested in contributing
to this project.

[contributing]: https://dfci.github.io/matchminer-ai-inference/development/contributing/

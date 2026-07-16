# matchminer-ai

## Overview

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

This package is currently pre-v1 and under active development.

For detailed instructions on getting started using this package, please see the
[documentation website](https://dfci.github.io/matchminer-ai-inference/).

## Compute requirements

To run the full inference pipeline, you will require access to GPU-backed model inference. Please see the
[requirements documentation](https://dfci.github.io/matchminer-ai-inference/getting-started/requirements/)
for more information on compute expectations and GPU recommendations.

Note that the GPU does not need to be on the same computer running your
workflow. Please see the documentation on
[choosing an inference setup](https://dfci.github.io/matchminer-ai-inference/user-guide/local-vs-remote/)
for more information.

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

## Citation

If you use `matchminer-ai`, please cite:
>Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

## Contributing

Contributions are welcome! Please follow our
[contribution instructions][contributing] if you are interested in contributing
to this project.

[contributing]: https://dfci.github.io/matchminer-ai-inference/development/contributing/

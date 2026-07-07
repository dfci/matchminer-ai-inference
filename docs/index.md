# MatchMiner-AI

`matchminer-ai` is a Python package for running the clinical trial matching
inference workflow described in
[Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228).

The package provides modular functions for the core MatchMiner-AI workflow:
summarizing trials and patient histories, generating embeddings of each,
retrieving candidate matches, scoring match quality, and assessing exclusion
criteria.

!!! warning "Pre-v1 package"
    This package is currently pre-v1 and under active development. APIs,
    configuration options, and outputs may change.

## Compute Requirements

The most compute-intensive step is summarizing patient notes with the default
Gemma 4 language model. Full pipeline runs can use either a local high-memory
GPU environment, such as an NVIDIA H100 80GB, or a compatible remote vLLM
inference server configured with the Gemma 4 reasoning parser.

Other entry points, such as running from precomputed summaries, may require
less compute.

## Installation

This package requires Python 3.13+.

```shell
pip install matchminer-ai
```

## Where To Start

- Read the [patient-centric matching tutorial](tutorials/patient-centric-matching.md)
  for workflow background and server-mode guidance.
- Open the [example notebook](tutorials/example-notebook.md) for a runnable
  walkthrough using sample input data.
- Browse the [API reference](api/index.md) for importable functions and
  parameters.
- Review the [configuration reference](reference/configuration.md) when editing
  preset YAML files.
- See [contributing](development/contributing.md) for local development, checks,
  and pull request guidance.

## Citation

If you use `matchminer-ai`, please cite:

> Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA,
> Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M,
> Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz
> T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI:
> An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026.
> doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

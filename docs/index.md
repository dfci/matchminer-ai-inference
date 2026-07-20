# MatchMiner-AI

Welcome to the `matchminer-ai` package documentation.

`matchminer-ai` is a Python package for running clinical trial matching
inference workflows. The package provides modular functions for summarizing
trials and patient histories, generating embeddings of each, retrieving
candidate matches, scoring match quality, and assessing exclusion criteria.

This package is based on
[Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228).
If you use `matchminer-ai`, please cite the study below.

Contributions to the package and documentation are welcome through GitHub. See our
[contribution instructions](development/contributing.md) for more information.

!!! warning "Pre-v1 package"
    This package is currently pre-v1 and under active development. APIs,
    configuration options, and outputs may change.

## Where To Start

- Read [requirements](getting-started/requirements.md) to understand compute
  and data expectations before installing.
- Follow [installation](getting-started/installation.md) once you have the
  required environment.
- Use the [quickstart](getting-started/quickstart.md) for the public function
  sequence and runnable example notebook.
- Read the [patient-centric matching workflow](user-guide/patient-centric-matching.md)
  for background and workflow concepts.
- Review the [configuration reference](reference/configuration.md) when editing
  preset YAML files.

## Citation

If you use `matchminer-ai`, please cite:

> Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA,
> Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M,
> Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz
> T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI:
> An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026.
> doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

# mmai package — skeleton reference

This document defines the **initial package skeleton only** for the `mmai` Python package.
It is derived directly from the MVP design document and is intended to guide creation of
the directory tree and empty modules (stubs only).

No implementation details are specified here.

---

## Purpose of the package

An open-source Python package that provides an API for running the MatchMiner-AI pipeline,
exposing the major pipeline phases (trial summarization, patient summarization, embedding,
matching, and checking) as first-class functions, along with end-to-end wrappers.

---

## High-level pipeline workflow

The pipeline consists of the following conceptual steps, in order:

1. summarize_trials
2. summarize_patients
3. embed_for_matching (applied to both trials and patients)
4. generate_candidate_matches
5. reasonable_match_check
6. exclusion_criteria_check

An end-to-end wrapper orchestrates these steps.

---

## Package structure (MVP)

The following directory and file structure is specified in the design document.

mmai/
├── pyproject.toml
├── README.md
├── src/
│ └── mmai/
│ ├── init.py
│ ├── pipeline.py # MMAIPipeline + end-to-end wrapper
│ ├── config.py # Config schema and preset loading
│ ├── presets/
│ │ └── default.yaml # model names and other parameters
│ ├── prompts/
│ │ ├── trial_summary.txt
│ │ ├── patient_summary.txt
│ │ ├── embedding.txt
│ │ └── checker_templates.txt
│ ├── trials/
│ │ ├── init.py
│ │ ├── prompt_builder.py
│ │ ├── summarize.py
│ │ └── postprocess.py
│ ├── patients/
│ │ ├── init.py
│ │ ├── tagging.py
│ │ ├── prompt_builder.py
│ │ ├── summarize.py
│ │ └── postprocess.py
│ ├── embedding/
│ │ ├── init.py
│ │ └── embed.py
│ ├── matching/
│ │ ├── init.py
│ │ ├── match.py
│ │ ├── reasonable_check.py
│ │ └── exclusion_check.py
│ ├── backends.py
│ └── utils/
│ ├── init.py
│ └── logging.py
└── tests/
├── test_pipeline.py
├── test_trials.py
├── test_patients.py
├── test_embedding.py
└── test_matching.py

yaml
Copy code

---

## Public functions (names only)

The following functions are part of the MVP public API.
At the skeleton stage, these should exist only as stubs.

### Trial summarization
- `summarize_trials(...)`

### Patient summarization
- `summarize_patients(...)`

### Embedding
- `embed_for_matching(...)`

### Candidate generation
- `generate_candidate_matches(...)`

### Match evaluation
- `reasonable_match_check(...)`
- `exclusion_criteria_check(...)`

### High-level pipeline
- `run_patient_centric_matching_pipeline(...)`

---

## Configuration (skeleton-level)

- Initializing `MMAIPipeline()` with no arguments loads a default preset.
- The default preset is versioned and represents the supported configuration.
- Custom configurations may be used experimentally.

No configuration schema is defined at the skeleton stage.

---

## Scope of this document

This document is intentionally limited to:
- directory structure
- module names
- public function names

No behavior, logic, validation, or implementation details are specified here.
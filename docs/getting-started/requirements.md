# Requirements

Before installing `matchminer-ai`, check that you have the compute environment
and data needed for the workflow you want to run.

## Compute

This package requires Python 3.13+.

Steps that use an LLM, such as summarizing patient notes, can run using a model hosted locally or through a remote OpenAI-compatible endpoint. You therefore do not necessarily need a GPU on the computer running `matchminer-ai`. See our [documentation on choosing an inference setup](../user-guide/local-vs-remote.md) for more information.

!!! warning
    Before sending clinical text to any remote endpoint, make sure the endpoint
    is approved for your data and institution.

Other parts of the workflow generate embeddings or run classification models locally. These models may be able to run on CPU, but CPU performance has not been formally evaluated and may be impractically slow. We recommend using a GPU for these steps.

See our [GPU recommendations](gpu-recommendations.md) documentation for examples of hardware we have used to run the full workflow.

## Data

If you are starting from the beginning of the patient-centric workflow, you
need two input tables:

- trial-level data for `summarize_trials`
- note-level patient data for `summarize_patients`

### Trial input

Trial summarization expects one row per trial with trial identifiers, title,
brief summary, and eligibility criteria.

If you are preparing trial input from ClinicalTrials.gov, first obtain the study
records as JSON and then build one row per study by selecting these fields:

| `summarize_trials` column | ClinicalTrials.gov JSON field |
| --- | --- |
| `trial_id` | `protocolSection.identificationModule.nctId` |
| `trial_title` | `protocolSection.identificationModule.briefTitle` or `officialTitle` |
| `brief_summary` | `protocolSection.descriptionModule.briefSummary` |
| `eligibility_criteria` | `protocolSection.eligibilityModule.eligibilityCriteria` |

### Patient input

Patient summarization expects one row per clinical note. Each row should include
a patient identifier, note text, and note date.

Users are responsible for preparing note-level input data approved for their
environment.

### Detailed input requirements

See the
[`summarize_trials`](../api/trials.md) and
[`summarize_patients`](../api/patients.md) API docs for current DataFrame
column requirements.

The example notebook uses sample input data from
[`examples/data/`](https://github.com/dfci/matchminer-ai-inference/tree/main/examples/data),
which is the best place to inspect the expected input shape when preparing
your own data.

You do not always have to start at the beginning. There are entry points later
in the workflow if you already have summaries, embeddings, candidate matches, or
other intermediate results. The input requirements for those entry points are
documented with the corresponding [Package API](../api/trials.md) pages.

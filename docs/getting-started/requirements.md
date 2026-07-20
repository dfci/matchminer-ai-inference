# Requirements

Before installing `matchminer-ai`, check that you have the compute environment
and data needed for the workflow you want to run.

## Compute

This package requires Python 3.13+.

Steps that use an LLM, such as summarizing patient notes, can run using a model hosted locally or through a remote OpenAI-compatible endpoint. You therefore do not necessarily need a GPU on the computer running `matchminer-ai`. See our [documentation on choosing an inference setup](../user-guide/local-vs-remote.md) for more information.

Other parts of the workflow generate embeddings or run classification models locally. These models may be able to run on CPU, but CPU performance has not been formally evaluated and may be impractically slow. We recommend using a GPU for these steps.

See our [GPU recommendations](gpu-recommendations.md) documentation for examples of hardware we have used to run the full workflow.

## Data

If you are starting from the beginning of the patient-centric workflow, you
need trial-level input data and note-level patient input data.

Trial summarization expects one row per trial with trial identifiers, title,
brief summary, and eligibility criteria. The package has been tested with trial
text prepared from ClinicalTrials.gov records, but it does not currently include
a helper for pulling ClinicalTrials.gov records or transforming them into
package input format.

Patient summarization expects one row per note with patient identifiers, note
text, and note dates. See the
[`summarize_trials`](../api/trials.md) and
[`summarize_patients`](../api/patients.md) API docs for current DataFrame
column requirements.

You do not always have to start at the beginning. There are entry points later
in the workflow if you already have summaries, embeddings, candidate matches, or
other intermediate results. The input requirements for those entry points are
documented with the corresponding function APIs.

The example notebook uses sample input data from
[`examples/data/`](https://github.com/dfci/matchminer-ai-inference/tree/main/examples/data),
which is the best place to inspect the expected input shape before preparing
your own data.

# Requirements

Before installing `matchminer-ai`, check that you have the compute environment
and data needed for the workflow you want to run.

## Compute

This package requires Python 3.13+.

Some parts of `matchminer-ai` are computationally intensive because they use
large language models. For example, summarizing patient notes usually requires
GPU-backed inference.

That GPU does not necessarily need to be on your own computer. For LLM-based
steps, the package can send requests to an OpenAI-compatible endpoint. That
endpoint could be:

- a vLLM server running on your own computer;
- a vLLM server running on another computer or cluster; or
- another approved OpenAI-compatible endpoint.

!!! warning
    Before sending clinical text to any remote endpoint, make sure the endpoint
    is approved for your data and institution.

If you do not have access to GPU-backed inference, you may still be able to use
parts of the package that start from existing summaries, embeddings, or match
results. A full end-to-end workflow will usually require access to GPU-backed
model inference somewhere.

For more detail, see [local vs remote inference](../user-guide/local-vs-remote.md)
and the [configuration reference](../reference/configuration.md).

## Data

If you are starting from the beginning of the patient-centric workflow, you
need trial-level input data and note-level patient input data.

Trial summarization expects one row per trial with trial identifiers, title,
brief summary, and eligibility criteria. Patient summarization expects one row
per note with patient identifiers, note text, and note dates. See the
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

# Requirements

Before installing `matchminer-ai`, check that you have the compute environment
and data needed for the workflow you want to run.

## Compute

This package requires Python 3.13+.

Some parts of `matchminer-ai` are computationally intensive because they use
large language models. For example, summarizing patient notes usually requires
access a GPU for model inference. Please see our [GPU recommendations](gpu-recommendations.md) documentation for more information on which GPUs are compatible with this package workflow.

The GPU **does not need to be on the same computer running your workflow**.
LLM-based steps can run with an in-process local vLLM backend, a local vLLM
server, or another approved OpenAI-compatible endpoint. Please see our [documentation on choosing an inference set up](../user-guide/local-vs-remote.md).




!!! warning
    Before sending clinical text to any remote endpoint, make sure the endpoint
    is approved for your data and institution.

If you do not have access to GPU-backed inference, you may still be able to use
parts of the package that start from existing summaries, embeddings, or match
results. TrialSpace, TrialChecker, and BoilerplateChecker models may run on CPU, albeit slowly.


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

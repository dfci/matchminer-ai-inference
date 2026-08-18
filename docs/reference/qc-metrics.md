# QC metrics

Patient and trial summarization can produce QC reports. These reports are meant
to help you review how a summarization run behaved before using the outputs
downstream.

To request a QC report, pass `return_qc=True` to `summarize_patients` or
`summarize_trials`.

QC reports summarize how the current pipeline behaves, given the current
prompts, filters, and postprocessing rules. Metrics are returned as rows with
`metric`, `value`, `denominator`, `percent`, and `ids`.

## Patient summarization QC

Returned by `summarize_patients(..., return_qc=True)`.

- `patients_exclusion_criteria_not_extracted`: exclusion criteria not successfully extracted.
- `patients_missing_keyword:<keyword>`: summaries missing an expected keyword.
- `patients_exceed_embedding_token_limit`: summaries whose embedding-tokenized
  length exceeds `embedding.max_seq_length`, the configured truncation cutoff
  used during embedding generation.

## Trial summarization QC
Returned by `summarize_trials(..., return_qc=True)`.

- `trials_missing_in_output`: trials present in the input but not represented
  in the output after summarization/postprocessing.
- `trials_failed_inference`: trials whose LLM inference finished with an error
  after retries were exhausted. These trials are removed before
  postprocessing.
- `trials_truncated_llm_response`: trials where the LLM stopped due to max
  token length.
- `spaces_per_trial_min|median|max`: min/median/max number of spaces per trial.
- `trials_with_non_distinct_spaces`: trials with duplicate space numbers or
  duplicate space text.
- `spaces_dropped_missing_keyword:<keyword>`: trial spaces dropped due to missing a required keyword.
- `trials_exclusion_criteria_not_extracted`: trials whose exclusion criteria was not extracted.
- `spaces_exceed_embedding_token_limit`: trial spaces whose embedding-tokenized
  length exceeds `embedding.max_seq_length`, the configured truncation cutoff
  used during embedding generation.

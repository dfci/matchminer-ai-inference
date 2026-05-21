# QC metrics

QC reports summarize how the current pipeline behaves, given the current
prompts, filters, and postprocessing rules. Metrics are returned as rows with
`metric`, `value`, `denominator`, `percent`, and `ids`.

## Patient summarization QC

Returned by `summarize_patients(..., return_qc=True)` and
`summarize_patient_notes(..., return_qc=True)`.

- `patients_dropped_noninformative_summary`: summaries dropped because they match non-informative patterns (e.g.,
  "no information", "no malignancy").
- `patients_exclusion_criteria_not_extracted`: exclusion criteria not successfully extracted.
- `patients_missing_keyword:<keyword>`: summaries missing an expected keyword.
- `patients_exceed_embedding_token_limit`: summaries whose embedding-tokenized
  length exceeds the allowable amount by the embedding model.

## Trial summarization QC
Returned by `summarize_trials(..., return_qc=True)`.

- `trials_missing_in_output`: trials present in the input but not represented
  in the output after summarization/postprocessing.
- `trials_truncated_llm_response`: trials where the LLM stopped due to max
  token length.
- `spaces_per_trial_min|median|max`: min/median/max number of spaces per trial.
- `trials_with_non_distinct_spaces`: trials with duplicate space numbers or
  duplicate space text.
- `spaces_dropped_missing_keyword:<keyword>`: trial spaces dropped due to missing a required keyword.
- `trials_exclusion_criteria_not_extracted`: trials whose exclusion criteria was not extracted.
- `spaces_exceed_embedding_token_limit`: trial spaces whose embedding-tokenized
  length exceeds the allowable amount by the embedding model.

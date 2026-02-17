# QC metrics

QC reports summarize how the current pipeline behaves, given the current
prompts, filters, and postprocessing rules. Metrics are returned as rows with
`metric`, `value`, `percent`, and `ids`.

## Patient summarization QC

### Tagging QC
Returned by `extract_relevant_sentences(..., return_qc=True)`.

- `patients_with_no_tagged_notes`: patients whose tagged long text is empty
  after the tagging step.

### Summary-only QC
Returned by `summarize_from_relevant_sentences(..., return_qc=True)`.

- `patients_dropped_noninformative_summary`: summaries dropped because they match non-informative patterns (e.g.,
  "no information", "no malignancy").
- `patients_truncated_llm_response`: summaries where the LLM stopped due to
  max token length.
- `patients_exclusion_criteria_not_extracted`: exclusion criteria not successfully extracted.
- `patients_missing_keyword:<keyword>`: summaries missing an expected keyword.
- `patients_exceed_embedding_token_limit`: summaries whose embedding-tokenized
  length exceeds the allowable amount by the embedding model.

### Full QC
Returned by `summarize_patients(..., return_qc=True)`.

- Includes tagging QC and summary-only QC, plus:
- `patients_missing_summaries`: patients in the input notes who are missing
  from the final summaries output.

### Patient percent denominators
- `patients_with_no_tagged_notes` and `patients_missing_summaries` are
  calculated as a percent of patients in the original input notes.
- `patients_dropped_noninformative_summary` and
  `patients_truncated_llm_response` are calculated as a percent of patients
  that reached the summarization step.
- `patients_exclusion_criteria_not_extracted`,
  `patients_missing_keyword:<keyword>`, and
  `patients_exceed_embedding_token_limit` are calculated as a percent of
  patients in the summary output table.

## Trial summarization QC
Returned by `summarize_trials(..., return_qc=True)`.

- `trials_missing_in_output`: trials present in the input but not represented
  in the output after summarization/postprocessing.
- `trials_truncated_llm_response`: trials where the LLM stopped due to max
  token length (from the separate trial-level `finish_reasons` series indexed
  by `trial_id`).
- `spaces_per_trial_min|median|max`: min/median/max number of spaces per trial.
- `trials_with_non_distinct_spaces`: trials with duplicate space numbers or
  duplicate space text.
- `spaces_dropped_missing_keyword:<keyword>`: spaces in the pre-filtered output
  that lack the keyword (these would be dropped by keyword filtering).
- `trials_exclusion_criteria_not_extracted`: trials with missing/empty
  general exclusion criteria in the final output.
- `spaces_exceed_embedding_token_limit`: spaces whose embedding-tokenized
  length exceeds `max_embedding_input_tokens` (default `2500`).

### Trial percent denominators
- Trial-level metrics use unique trials in `trial_source` as the denominator
  (for example, `trials_missing_in_output`,
  `trials_with_non_distinct_spaces`,
  `trials_exclusion_criteria_not_extracted`).
- `trials_truncated_llm_response` uses the number of generated trial summaries
  (`len(finish_reasons)`) as the denominator.
- Space-level metrics use the number of rows in the final `trial_spaces` table
  as the denominator (for example,
  `spaces_dropped_missing_keyword:<keyword>`,
  `spaces_exceed_embedding_token_limit`).

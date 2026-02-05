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

- `patients_dropped_noninformative_summary`: summaries dropped by
  `clean_bad_data` because they match non-informative patterns (e.g.,
  "no information", "no malignancy").
- `patients_exclusion_criteria_not_extracted`: summary text equals the
  exclusion criteria text, implying exclusions were not separated.
- `patients_missing_keyword:<keyword>`: summaries missing an expected keyword.
- `patient_summaries_excessive_length`: summaries above the length threshold.

### Full QC
Returned by `summarize_patients(..., return_qc=True)`.

- Includes tagging QC and summary-only QC, plus:
- `patients_missing_summaries`: patients in the input notes who are missing
  from the final summaries output.

## Trial summarization QC
Returned by `summarize_trials(..., return_qc=True)`.

- `trials_missing_in_output`: trials present in the input but not represented
  in the output after summarization/postprocessing.
- `spaces_per_trial_min|median|max`: min/median/max number of spaces per trial.
- `trials_with_non_distinct_spaces`: trials with duplicate space numbers or
  duplicate space text.
- `spaces_dropped_missing_keyword:<keyword>`: spaces in the pre-filtered output
  that lack the keyword (these would be dropped by keyword filtering).
- `trials_exclusion_criteria_not_extracted`: trials with missing/empty
  general exclusion criteria in the final output.
- `spaces_excessive_length`: spaces whose text exceeds the length threshold.

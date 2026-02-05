# QC metrics

QC reports summarize how the current pipeline behaves, given the current
prompts, filters, and postprocessing rules. Metrics are returned as rows with
`metric`, `value`, `percent`, and `ids`.

## Patient summarization QC

### Tagging QC
Returned by `extract_relevant_sentences(..., return_qc=True)`.

- `patients_with_no_tagged_notes`: patients whose tagged long text is empty.

### Summary-only QC
Returned by `summarize_from_relevant_sentences(..., return_qc=True)`.

- `patients_dropped_noninformative_summary`: dropped because summaries match
  non-informative patterns (e.g., no information/no malignancy).
- `patients_summary_equals_boilerplate`: summary text equals boilerplate text.
- `patients_missing_keyword:<keyword>`: missing expected keyword in summary.
- `patient_summaries_excessive_length`: summary length exceeds the threshold.

### Full QC
Returned by `summarize_patients(..., return_qc=True)`.

- Includes tagging QC and summary-only QC, plus:
- `patients_missing_summaries`: patients present in input notes but missing
  from final summaries.

## Trial summarization QC
Returned by `summarize_trials(..., return_qc=True)`.

- `trials_missing_summaries`: trials missing any summary rows.
- `spaces_per_trial_min|median|max`: distribution of spaces per trial.
- `trials_with_non_distinct_spaces`: duplicate space numbers or duplicate text.
- `spaces_missing_keyword:<keyword>`: spaces missing expected sections.
- `trials_missing_boilerplate_exclusions`: missing/empty boilerplate text.
- `spaces_excessive_length`: space text exceeds the length threshold.

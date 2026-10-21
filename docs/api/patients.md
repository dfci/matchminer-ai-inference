# Patients

::: matchminer_ai.patients
    options:
      members:
        - summarize_patients

## Resuming patient summarization

Pass a run-specific filesystem directory as `checkpoint_dir` to save progress
during patient summarization. If processing is interrupted, call
`summarize_patients` again with the same directory to continue from the last
saved point.

Omit `checkpoint_dir` to run without writing checkpoint files.

### Checkpoint files

When `checkpoint_dir` is supplied, patient summarization writes these
Parquet files to that directory:

- `prepared_chunks.parquet` — the patient note chunks prepared for
  summarization, saved once before the first round runs.
- `round_0000.parquet`, `round_0001.parquet`, ... — one file per
  summarization round, each holding every patient's result for that round.

Each round checkpoint has the following columns:

| Column | Description |
| --- | --- |
| `patient_id` | Patient identifier. |
| `prior_summary` | The summary carried into this round, before this round's chunk was incorporated. |
| `patient_reasoning_text` | The model's reasoning trace for this round. |
| `patient_answer_text` | The model's summary output for this round. |
| `finish_reason` | Generation finish reason, e.g. `"stop"` or `"error"`. |

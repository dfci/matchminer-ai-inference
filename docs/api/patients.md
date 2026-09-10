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

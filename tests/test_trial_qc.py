import pandas as pd

from mmai._qc.trials import trial_qc_report
from mmai.config import MMAIConfig


def test_trial_qc_report_metrics(monkeypatch):
    """Validate trial QC metrics for missing summaries, duplicates, and boilerplate gaps."""
    trial_source = pd.DataFrame(
        [
            {"trial_id": "T1"},
            {"trial_id": "T2"},
            {"trial_id": "T3"},
        ]
    )
    trial_spaces = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "space_trial_id": "T1-1",
                "clinical_space_number": 1,
                "clinical_space_summary": (
                    "Age 18+. Sex male. Cancer type allowed: Lung. "
                    "Histology allowed: Adenocarcinoma. Cancer burden allowed: "
                    "Metastatic. Prior treatment required: None. "
                    "Prior treatment excluded: None. Biomarkers required: EGFR. "
                    "Biomarkers excluded: None."
                ),
                "general_exclusion_criteria": "None",
            },
            {
                "trial_id": "T1",
                "space_trial_id": "T1-1-dup",
                "clinical_space_number": 1,
                "clinical_space_summary": (
                    "Age 18+. Sex male. Cancer type allowed: Lung."
                ),
                "general_exclusion_criteria": "Boilerplate exclusions:\nNone.",
            },
            {
                "trial_id": "T2",
                "space_trial_id": "T2-1",
                "clinical_space_number": 1,
                "clinical_space_summary": "Cancer type allowed: Breast.",
                "general_exclusion_criteria": "",
            },
        ]
    )
    finish_reasons = pd.Series(
        ["stop", "length"],
        index=["T1", "T2"],
    )
    monkeypatch.setattr(
        "mmai._qc.trials.get_backend",
        lambda _: type(
            "MockBackend",
            (),
            {
                "count_embedding_tokens": lambda self, texts, embedding_config: [
                    120,
                    10,
                    3400,
                ]
            },
        )(),
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        embedding={"model_path": "m", "device": "cpu", "prompt_file": "embedding.txt"},
        model_metadata_cache_dir=None,
        raw={},
    )

    report = trial_qc_report(
        trial_spaces,
        trial_source=trial_source,
        unfiltered_spaces=trial_spaces,
        finish_reasons=finish_reasons,
        config=config,
        max_embedding_input_tokens=2500,
    ).set_index("metric")

    assert report.loc["trials_missing_in_output", "value"] == 1
    assert report.loc["trials_missing_in_output", "percent"] > 0
    assert report.loc["trials_missing_in_output", "ids"] == ["T3"]
    assert report.loc["spaces_per_trial_max", "value"] == 2
    assert report.loc["trials_with_non_distinct_spaces", "value"] == 1
    assert report.loc["trials_with_non_distinct_spaces", "ids"] == ["T1"]
    assert report.loc["spaces_exceed_embedding_token_limit", "value"] == 1
    assert report.loc["spaces_exceed_embedding_token_limit", "ids"] == ["T2-1"]
    assert report.loc["trials_exclusion_criteria_not_extracted", "value"] == 2
    assert report.loc["spaces_dropped_missing_keyword:Age", "value"] >= 1

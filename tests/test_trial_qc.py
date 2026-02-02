import pandas as pd

from mmai.qc.trials import trial_qc_report


def test_trial_qc_report_metrics():
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

    report = trial_qc_report(
        trial_spaces,
        trial_source=trial_source,
        max_space_length=50,
    ).set_index("metric")

    assert report.loc["trials_missing_summaries", "value"] == 1
    assert report.loc["trials_missing_summaries", "percent"] > 0
    assert report.loc["trials_missing_summaries", "ids"] == ["T3"]
    assert report.loc["spaces_per_trial_max", "value"] == 2
    assert report.loc["trials_with_non_distinct_spaces", "value"] == 1
    assert report.loc["trials_with_non_distinct_spaces", "ids"] == ["T1"]
    assert report.loc["trials_missing_boilerplate_exclusions", "value"] == 2
    assert report.loc["spaces_excessive_length", "value"] >= 1
    assert report.loc["spaces_missing_keyword:Age", "value"] >= 1

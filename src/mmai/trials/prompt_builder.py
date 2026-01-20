"""Trial prompt builders."""

from __future__ import annotations

import re
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def load_prompt_text(filename: str) -> str:
    """Load a prompt text file from the package."""
    prompt_path = resources.files("mmai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def format_trial_text(
    trial_text: str, primer_filename: str, question_filename: str
) -> str:
    """Wrap a trial document in the configured prompt template."""
    return (
        load_prompt_text(primer_filename)
        + "Here is a clinical trial document: \n"
        + trial_text
        + "\n"
        + load_prompt_text(question_filename)
    )


def get_filled_trial_prompt(
    trial_text: str, primer_filename: str, question_filename: str
) -> list[dict[str, str]]:
    """Prepare the chat-style prompt for a trial document."""
    return [
        {
            "role": "system",
            "content": """
Reasoning: high.
""",
        },
        {
            "role": "user",
            "content": format_trial_text(
                trial_text, primer_filename, question_filename
            ),
        },
    ]


def build_trial_text(trials: "pd.DataFrame") -> "pd.Series":
    """Create the raw trial text used for LLM summarization."""
    text = (
        trials["trial_title"].fillna("")
        + "\n"
        + trials["brief_summary"].fillna("")
        + "\n"
        + trials["eligibility_criteria"].fillna("")
    )
    return text.apply(lambda value: re.sub(r"\s+", " ", value).strip())

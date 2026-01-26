"""Patient prompt builders."""

from __future__ import annotations

from importlib import resources


def load_prompt_text(filename: str) -> str:
    """Load a prompt text file from the package."""
    prompt_path = resources.files("mmai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def format_patient_text(
    patient_text: str, primer_filename: str, question_filename: str
) -> str:
    """Wrap patient text in the configured prompt template."""
    return (
        load_prompt_text(primer_filename)
        + "The excerpt for you to summarize is:\n"
        + patient_text
        + "\n"
        + load_prompt_text(question_filename)
    )


def get_filled_patient_prompt(
    patient_text: str, primer_filename: str, question_filename: str
) -> list[dict[str, str]]:
    """Prepare the chat-style prompt for a patient summary."""
    return [
        {
            "role": "system",
            "content": """
Reasoning: high.
""",
        },
        {
            "role": "user",
            "content": format_patient_text(
                patient_text, primer_filename, question_filename
            ),
        },
    ]

"""vLLM reasoning parser helpers."""

from __future__ import annotations

from typing import Any


def parse_reasoning_output(
    text: str,
    *,
    parser_name: str | None,
    tokenizer: Any,
) -> tuple[str, str]:
    """Split raw vLLM output into ``(reasoning, final_text)``."""
    if parser_name is None:
        return "", text.strip()

    if parser_name == "gemma4":
        from vllm.reasoning.gemma4_utils import parse_thinking_output

        parsed = parse_thinking_output(text)
        reasoning = parsed.get("thinking") or ""
        content = parsed.get("answer") or ""
    elif parser_name == "openai_gptoss":
        from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output

        reasoning, content = parse_chat_output(text)
    else:
        from vllm.reasoning import ReasoningParserManager

        parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
        parser = parser_cls(tokenizer)
        reasoning, content = parser.extract_reasoning(text, request=None)

    return (reasoning or "").strip(), (content or "").strip()


def compose_raw_text(reasoning: str, content: str) -> str:
    """Reconstruct a readable debug string from structured reasoning output."""
    reasoning = reasoning.strip()
    content = content.strip()
    if reasoning and content:
        return f"{reasoning}\n{content}"
    return reasoning or content


__all__ = [
    "compose_raw_text",
    "parse_reasoning_output",
]

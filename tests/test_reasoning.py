import sys
from types import ModuleType

from matchminer_ai.llm.reasoning import parse_reasoning_output


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_parse_reasoning_output_uses_gemma4_utility(monkeypatch):
    """Gemma 4 local parsing uses vLLM's offline thinking parser."""

    def parse_thinking_output(text):
        assert text == "raw output"
        return {"thinking": " model reasoning ", "answer": " final text "}

    monkeypatch.setitem(sys.modules, "vllm", _module("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.reasoning", _module("vllm.reasoning"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning.gemma4_utils",
        _module(
            "vllm.reasoning.gemma4_utils",
            parse_thinking_output=parse_thinking_output,
        ),
    )

    reasoning, content = parse_reasoning_output(
        "raw output",
        parser_name="gemma4",
        tokenizer=object(),
    )

    assert reasoning == "model reasoning"
    assert content == "final text"


def test_parse_reasoning_output_uses_reasoning_parser_manager(monkeypatch):
    """Other parsers go through vLLM's class-based parser registry."""

    class FakeParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def extract_reasoning(self, text, request):
            assert text == "raw output"
            assert request is None
            return " parsed reasoning ", " parsed final "

    class FakeReasoningParserManager:
        @classmethod
        def get_reasoning_parser(cls, parser_name):
            assert parser_name == "qwen3"
            return FakeParser

    monkeypatch.setitem(sys.modules, "vllm", _module("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.reasoning",
        _module(
            "vllm.reasoning",
            ReasoningParserManager=FakeReasoningParserManager,
        ),
    )

    reasoning, content = parse_reasoning_output(
        "raw output",
        parser_name="qwen3",
        tokenizer=object(),
    )

    assert reasoning == "parsed reasoning"
    assert content == "parsed final"

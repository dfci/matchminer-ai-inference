from matchminer_ai.llm.prompt_rendering import build_prompt_list


class MockTokenizer:
    """Minimal tokenizer stub that records chat-template calls."""

    def __init__(self):
        """Initialize call storage."""
        self.calls = []

    def apply_chat_template(self, **kwargs):
        """Record template kwargs and return deterministic rendered text."""
        self.calls.append(kwargs)
        return "rendered prompt"


def test_build_prompt_list_passes_chat_template_kwargs(monkeypatch):
    """Local prompt building applies configured chat-template kwargs."""
    tokenizer = MockTokenizer()
    monkeypatch.setattr(
        "matchminer_ai.llm.prompt_rendering._get_chat_template_tokenizer",
        lambda model_name: tokenizer,
    )

    messages = [[{"role": "user", "content": "hello"}]]
    prompts = build_prompt_list(
        messages,
        llm_config={
            "model_name": "google/gemma-4-31B-it",
            "sampling_params": {"max_tokens": 5},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert tokenizer.calls[0]["enable_thinking"] is True
    assert prompts[0].prompt_text == "rendered prompt"
    assert prompts[0].messages == messages[0]


def test_build_prompt_list_remote_mode_skips_chat_template(monkeypatch):
    """Remote prompt building preserves messages without loading a HF tokenizer."""

    def fail_if_called(model_name):
        """Fail if remote prompt building tries to load a tokenizer."""
        raise AssertionError("remote prompt building should not load tokenizer")

    monkeypatch.setattr(
        "matchminer_ai.llm.prompt_rendering._get_chat_template_tokenizer",
        fail_if_called,
    )

    messages = [
        [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "hello"},
        ]
    ]
    prompts = build_prompt_list(
        messages,
        llm_config={
            "backend_mode": "remote",
            "model_name": "non-hf-endpoint-model",
            "sampling_params": {"max_tokens": 5},
        },
    )

    assert prompts[0].messages == messages[0]
    assert prompts[0].max_tokens == 5
    assert prompts[0].prompt_text == "system: system text\n\nuser: hello"

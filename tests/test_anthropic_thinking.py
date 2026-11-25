"""Thinking, effort and structured output on AnthropicBackend: each is sent as given and
recorded, a setting the Messages API does not document is refused before any request, thinking
tokens are counted when the API reports them, and no thinking text reaches a record."""

import json
from pathlib import Path

import httpx
import pytest

from helpers import EXAMPLES, FakeServer, anthropic_message
from reliacode import AnthropicBackend, BackendError, Ledger, Rule, code, load_codebook

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
MESSAGES = [
    {"role": "system", "content": "Code the tone."},
    {"role": "user", "content": "The market moved to the old school yard."},
]
SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "enum": ["positive", "negative"]}},
    "required": ["code"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "test-key-for-thinking")


def backend_for(server: FakeServer, **settings: object) -> AnthropicBackend:
    return AnthropicBackend(
        "claude-test-model", api_key_env=KEY_ENV, transport=server.transport, **settings
    )


def ok(**fields: object) -> httpx.Response:
    return httpx.Response(200, json=anthropic_message(**fields))


# Thinking -------------------------------------------------------------------------------------


def test_thinking_is_sent_as_given_and_recorded() -> None:
    server = FakeServer(ok())
    setting = {"type": "adaptive", "display": "summarized"}
    response = backend_for(server, thinking=setting).complete(MESSAGES)
    assert server.body()["thinking"] == setting
    assert response.parameters.thinking == setting


def test_without_a_thinking_setting_no_field_is_sent_and_the_record_says_so() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert "thinking" not in server.body()
    assert response.parameters.thinking is None


def test_the_thinking_setting_cannot_be_changed_through_a_copy() -> None:
    server = FakeServer(ok())
    backend = backend_for(server, thinking={"type": "adaptive"})
    backend.thinking["type"] = "disabled"  # type: ignore[index]
    backend.complete(MESSAGES)
    assert server.body()["thinking"] == {"type": "adaptive"}


def test_an_older_models_budget_is_sent_when_it_fits_the_token_limit() -> None:
    server = FakeServer(ok())
    setting = {"type": "enabled", "budget_tokens": 2000}
    backend_for(server, thinking=setting, max_tokens=4000).complete(MESSAGES)
    assert server.body()["thinking"] == setting


@pytest.mark.parametrize(
    ("setting", "problem"),
    [
        ("adaptive", "should be a mapping"),
        ({}, "its type is missing"),
        ({"type": "auto"}, "its type is 'auto'"),
        ({"type": "enabled"}, 'needs "budget_tokens"'),
        ({"type": "enabled", "budget_tokens": 512}, "at least"),
        ({"type": "enabled", "budget_tokens": 4096}, "less than max_tokens"),
        ({"type": "adaptive", "display": "full"}, '"display" should be one of'),
        ({"type": "adaptive", "extra": float("nan")}, "only JSON values"),
        ({"type": "adaptive", "api_key": "x"}, "looks like a credential"),
    ],
)
def test_a_thinking_setting_the_api_does_not_document_is_refused(setting, problem) -> None:
    server = FakeServer(ok())
    with pytest.raises((TypeError, ValueError), match=problem):
        backend_for(server, thinking=setting, max_tokens=4096)
    assert server.requests == []


def test_thinking_tokens_are_counted_and_a_missing_count_stays_missing(tmp_path: Path) -> None:
    usage = {"input_tokens": 80, "output_tokens": 300, "output_tokens_details": {}}
    usage["output_tokens_details"]["thinking_tokens"] = 288
    response = backend_for(FakeServer(ok(usage=usage)), thinking={"type": "adaptive"}).complete(
        MESSAGES
    )
    assert response.reasoning_tokens == 288
    assert response.completion_tokens == 300  # thinking is part of the output, counted once
    usage = {"input_tokens": 80, "output_tokens": 300}
    response = backend_for(FakeServer(ok(usage=usage))).complete(MESSAGES)
    assert response.reasoning_tokens is None


def test_no_thinking_text_reaches_a_result_or_a_ledger_line(tmp_path: Path) -> None:
    secret = "A summary of the model's reasoning that must not be kept."
    content = [
        {"type": "thinking", "thinking": secret, "signature": "signature-abc"},
        {"type": "text", "text": '{"code": "economic"}'},
    ]
    usage = {"input_tokens": 90, "output_tokens": 40, "output_tokens_details": {}}
    usage["output_tokens_details"]["thinking_tokens"] = 30
    server = FakeServer(ok(content=content, usage=usage))
    backend = backend_for(server, thinking={"type": "adaptive", "display": "summarized"})
    ledger = Ledger(tmp_path / "ledger.jsonl")
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    run = code(codebook, ["A text."], backend=backend, variables="dominant_frame", ledger=ledger)
    assert run.results[0].code == "economic"
    assert run.results[0].answers == ('{"code": "economic"}',)
    for saved in (run.model_dump_json(), (tmp_path / "ledger.jsonl").read_text("utf-8")):
        assert secret not in saved
        assert "signature-abc" not in saved
    assert ledger.entries()[0].reasoning_tokens == 30


def test_a_refused_thinking_setting_gets_a_hint() -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": ""}}
    body["error"]["message"] = "thinking.type.enabled is not supported for this model"
    server = FakeServer(httpx.Response(400, json=body))
    backend = backend_for(server, thinking={"type": "enabled", "budget_tokens": 2000})
    with pytest.raises(BackendError, match="may not take this thinking setting"):
        backend.complete(MESSAGES)


# Effort ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_effort_is_sent_in_the_output_config_and_recorded(level: str) -> None:
    server = FakeServer(ok())
    response = backend_for(server, effort=level).complete(MESSAGES)
    assert server.body()["output_config"] == {"effort": level}
    assert response.parameters.effort == level


@pytest.mark.parametrize("level", ["Low", "minimal", "", 3])
def test_an_effort_the_api_does_not_document_is_refused(level: object) -> None:
    with pytest.raises(ValueError, match="effort should be one of"):
        AnthropicBackend("claude-test-model", effort=level)


def test_a_refused_effort_gets_a_hint() -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": ""}}
    body["error"]["message"] = "output_config.effort: xhigh is not supported on this model"
    server = FakeServer(httpx.Response(400, json=body))
    with pytest.raises(BackendError, match="may not take this effort level"):
        backend_for(server, effort="xhigh").complete(MESSAGES)


# Structured output ----------------------------------------------------------------------------


def test_the_answers_schema_goes_in_the_output_config_and_is_recorded() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES, json_schema=SCHEMA)
    assert server.body()["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert response.parameters.structured_output is True


def test_without_structured_output_the_schema_is_not_sent_and_the_record_says_so() -> None:
    server = FakeServer(ok())
    response = backend_for(server, structured_output=False).complete(MESSAGES, json_schema=SCHEMA)
    assert "output_config" not in server.body()
    assert response.parameters.structured_output is False
    with pytest.raises(TypeError, match="True or False"):
        AnthropicBackend("claude-test-model", structured_output="yes")


def test_effort_and_the_schema_go_together() -> None:
    server = FakeServer(ok())
    backend_for(server, effort="low").complete(MESSAGES, json_schema=SCHEMA)
    assert server.body()["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }


# One call per unit, cached, with adaptive thinking: the whole request ---------------------------


def test_the_request_for_one_call_per_unit_with_rules_caching_and_thinking() -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    rule = Rule(
        text="Severity is none if and only if incivility_form is none.",
        variables=["incivility_form", "severity"],
        check=lambda codes: (codes["severity"] == "none") == (codes["incivility_form"] == "none"),
    )
    server = FakeServer(ok(text='{"incivility_form": "none", "severity": "none"}'))
    backend = backend_for(server, max_tokens=4000, cache="1h", thinking={"type": "adaptive"})
    text = "Thanks for posting the new library hours."
    run = code(codebook, [text], backend=backend, recipe="per_unit", rules=[rule])
    (prompt,) = run.prompts
    body = server.body()
    assert list(body) == ["model", "max_tokens", "system", "messages", "thinking", "output_config"]
    assert body == {
        "model": "claude-test-model",
        "max_tokens": 4000,
        "system": [
            {
                "type": "text",
                "text": prompt.system,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [{"role": "user", "content": prompt.user_template.format(text=text)}],
        "thinking": {"type": "adaptive"},
        "output_config": {"format": {"type": "json_schema", "schema": prompt.json_schema}},
    }
    assert "Rules across variables" in prompt.system
    parameters = run.results[0].parameters
    assert (parameters.cache, parameters.thinking, parameters.effort) == (
        "1h",
        {"type": "adaptive"},
        None,
    )
    assert parameters.structured_output is True
    assert json.loads(json.dumps(body))  # every value is JSON

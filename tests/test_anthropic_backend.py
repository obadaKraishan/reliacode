"""AnthropicBackend sends each request to the Messages API in its own form, records what it
asked, reads the answer's text and counts, and explains every refusal in plain words. Every
answer comes from a mock server: no request leaves the test."""

import json
from pathlib import Path

import httpx
import pytest

from helpers import MINIMAL, FakeClock, FakeServer, anthropic_message
from reliacode import (
    AnthropicBackend,
    Backend,
    BackendError,
    Ledger,
    code,
    parse_codebook,
)
from reliacode.anthropic_backend import API_VERSION

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
KEY = "test-anthropic-key-123"
MESSAGES = [
    {"role": "system", "content": "You code the tone of short texts."},
    {"role": "user", "content": "Text to code:\n\nThe park reopened after the storm."},
]


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)


def backend_for(server: FakeServer, clock: FakeClock | None = None, **settings: object):
    clock = clock or FakeClock()
    return AnthropicBackend(
        "claude-test-model",
        api_key_env=KEY_ENV,
        transport=server.transport,
        sleep=clock.sleep,
        clock=clock,
        **settings,
    )


def ok(**fields: object) -> httpx.Response:
    return httpx.Response(200, json=anthropic_message(**fields), headers={"request-id": "req_1"})


def error(status: int, message: str, kind: str = "invalid_request_error", **headers: str):
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return httpx.Response(status, json=body, headers=headers)


# The request ----------------------------------------------------------------------------------


def test_it_is_a_backend() -> None:
    assert isinstance(backend_for(FakeServer(ok())), Backend)


def test_the_request_goes_to_the_messages_api_in_its_own_form() -> None:
    server = FakeServer(ok())
    backend_for(server).complete(MESSAGES)
    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["anthropic-version"] == API_VERSION == "2026-10-21"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in request.headers
    assert server.body() == {
        "model": "claude-test-model",
        "max_tokens": 4096,
        "system": [{"type": "text", "text": "You code the tone of short texts."}],
        "messages": [MESSAGES[1]],
    }
    assert list(server.body()) == ["model", "max_tokens", "system", "messages"]


def test_no_temperature_is_sent_unless_given_and_the_record_says_so() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert "temperature" not in server.body()
    assert response.parameters.temperature is None
    response = backend_for(server, temperature=0.5, api_version="2023-06-01").complete(MESSAGES)
    assert server.body()["temperature"] == 0.5
    assert response.parameters.temperature == 0.5
    assert server.requests[-1].headers["anthropic-version"] == "2023-06-01"


def test_the_record_holds_what_was_asked() -> None:
    response = backend_for(FakeServer(ok()), max_tokens=4000).complete(MESSAGES)
    parameters = response.parameters
    assert parameters.endpoint == "https://api.anthropic.com/v1/messages"
    assert parameters.model == "claude-test-model"
    assert parameters.max_tokens == 4000
    assert parameters.api_version == "2026-10-21"
    assert parameters.seed is None
    assert parameters.structured_output is False
    assert parameters.thinking is None and parameters.effort is None and parameters.cache is None
    assert parameters.extra_body == {}


def test_second_asks_keep_their_turns_and_several_system_messages_become_blocks() -> None:
    server = FakeServer(ok())
    messages = [
        {"role": "system", "content": "First part."},
        {"role": "system", "content": "Second part."},
        {"role": "user", "content": "Text to code: A."},
        {"role": "assistant", "content": "not json"},
        {"role": "user", "content": "That answer cannot be used."},
    ]
    backend_for(server).complete(messages)
    body = server.body()
    assert body["system"] == [
        {"type": "text", "text": "First part."},
        {"type": "text", "text": "Second part."},
    ]
    assert body["messages"] == messages[2:]


def test_a_conversation_without_a_system_message_has_no_system_field() -> None:
    server = FakeServer(ok())
    backend_for(server).complete([MESSAGES[1]])
    assert "system" not in server.body()


def test_a_system_message_after_another_message_is_refused_before_sending() -> None:
    server = FakeServer(ok())
    with pytest.raises(ValueError, match="Message 2 is a system message after another message"):
        backend_for(server).complete([MESSAGES[1], MESSAGES[0]])
    assert server.requests == []


def test_extra_fields_are_sent_after_the_backends_own_and_recorded() -> None:
    server = FakeServer(ok())
    response = backend_for(server, extra_body={"metadata": {"user_id": "study-a"}}).complete(
        MESSAGES
    )
    assert list(server.body())[-1] == "metadata"
    assert response.parameters.extra_body == {"metadata": {"user_id": "study-a"}}


@pytest.mark.parametrize(
    "name",
    [
        "model",
        "messages",
        "system",
        "max_tokens",
        "stream",
        "temperature",
        "thinking",
        "output_config",
        "cache_control",
    ],
)
def test_a_field_the_backend_sets_itself_is_refused_in_extra_body(name: str) -> None:
    with pytest.raises(ValueError, match=f'extra_body names "{name}"'):
        AnthropicBackend("claude-test-model", extra_body={name: 1})


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://api.anthropic.com", "https://api.anthropic.com"),
        ("https://api.anthropic.com/", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com"),
        ("http://localhost:8080", "http://localhost:8080"),
    ],
)
def test_the_base_url_is_read_without_its_path(given: str, expected: str) -> None:
    backend = AnthropicBackend("m", base_url=given)
    assert backend.base_url == expected
    assert backend.endpoint == expected + "/v1/messages"


@pytest.mark.parametrize(
    ("base_url", "problem"),
    [
        ("https://user:pass@api.anthropic.com", "user name, a password or a query"),
        ("https://api.anthropic.com?key=1", "user name, a password or a query"),
        ("ftp://api.anthropic.com", "should start with http"),
        ("https://api.anthropic.com:port", "can be read"),
        ("", "the API's address"),
    ],
)
def test_a_base_url_that_cannot_be_used_is_refused(base_url: str, problem: str) -> None:
    with pytest.raises((TypeError, ValueError), match=problem):
        AnthropicBackend("m", base_url=base_url)


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"api_version": "latest"}, "YYYY-MM-DD"),
        ({"api_key_env": None}, "needs an API key"),
        ({"max_tokens": 0}, "at least 1"),
        ({"max_tokens": None}, "needs a limit"),
        ({"temperature": -1}, "at least 0"),
        ({"timeout": 0}, "more than 0"),
        ({"max_retries": -1}, "at least 0"),
    ],
)
def test_settings_that_cannot_be_used_are_refused(settings: dict, problem: str) -> None:
    with pytest.raises((TypeError, ValueError), match=problem):
        AnthropicBackend("m", **settings)


# The answer -----------------------------------------------------------------------------------


def test_the_answer_and_its_counts_are_read() -> None:
    usage = {
        "input_tokens": 30,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 12,
        "output_tokens_details": {"thinking_tokens": 5},
    }
    response = backend_for(FakeServer(ok(usage=usage))).complete(MESSAGES)
    assert response.text == '{"code": "positive"}'
    assert response.model == "claude-test-model"
    assert response.prompt_tokens == 30
    assert response.completion_tokens == 12
    assert response.total_tokens == 42
    assert response.reasoning_tokens == 5
    assert response.finish_reason == "end_turn"
    assert response.response_id == "msg_test_1"
    assert response.request_id == "req_1"
    assert response.system_fingerprint is None
    assert response.attempts == 1
    assert response.batch_id is None and response.custom_id is None


def test_counts_the_api_leaves_out_stay_missing() -> None:
    response = backend_for(FakeServer(ok(usage={}))).complete(MESSAGES)
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None
    assert response.reasoning_tokens is None
    assert response.cache_read_tokens is None and response.cache_write_tokens is None


def test_only_text_blocks_are_read_and_no_thinking_text_is_kept() -> None:
    content = [
        {"type": "thinking", "thinking": "A private chain of reasoning.", "signature": "sig-1"},
        {"type": "text", "text": '{"code": '},
        {"type": "text", "text": '"negative"}'},
    ]
    response = backend_for(FakeServer(ok(content=content))).complete(MESSAGES)
    assert response.text == '{"code": "negative"}'
    record = response.model_dump_json()
    assert "private chain" not in record
    assert "sig-1" not in record


@pytest.mark.parametrize(
    ("answer", "what"),
    [
        (httpx.Response(200, text="<html>hello</html>"), "not JSON"),
        (httpx.Response(200, json=["a"]), "not a JSON object"),
        (httpx.Response(200, json={"id": "msg_1"}), 'no "content" list'),
        (httpx.Response(200, json=anthropic_message(content=[{"type": "text"}])), "no text"),
    ],
)
def test_an_answer_that_is_not_a_message_is_explained(answer: httpx.Response, what: str) -> None:
    with pytest.raises(BackendError, match=what) as caught:
        backend_for(FakeServer(answer)).complete(MESSAGES)
    assert caught.value.reason == "unexpected_answer"


def test_an_error_in_a_successful_status_is_reported() -> None:
    body = {"type": "error", "error": {"type": "api_error", "message": "Something failed."}}
    with pytest.raises(BackendError, match=r'The API said: "Something failed\."') as caught:
        backend_for(FakeServer(httpx.Response(200, json=body))).complete(MESSAGES)
    assert caught.value.reason == "server_error"


# Refusals -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "reason", "words"),
    [
        (error(401, "invalid x-api-key", "authentication_error"), "key_refused", KEY_ENV),
        (error(403, "not allowed", "permission_error"), "key_refused", "may use the model"),
        (error(402, "billing", "billing_error"), "billing", "Claude Console"),
        (error(404, "model: claude-nope not found", "not_found_error"), "model_not_found", "list"),
        (error(404, "Not found", "not_found_error"), "no_endpoint", "Check base_url"),
        (error(413, "too large", "request_too_large"), "bad_request", "larger than the API"),
        (error(400, "output_config.format: schema is invalid"), "bad_request", "structured_output"),
        (httpx.Response(302, headers={"location": "https://elsewhere.test"}), "redirected", "else"),
    ],
)
def test_refusals_are_explained_in_plain_words(answer, reason: str, words: str) -> None:
    with pytest.raises(BackendError, match=words) as caught:
        backend_for(FakeServer(answer)).complete(MESSAGES)
    assert caught.value.reason == reason
    assert caught.value.status == answer.status_code
    assert caught.value.endpoint == "https://api.anthropic.com/v1/messages"


def test_a_refused_temperature_gets_a_hint() -> None:
    answer = error(400, "temperature: is not supported for this model")
    with pytest.raises(BackendError, match="temperature=None") as caught:
        backend_for(FakeServer(answer), temperature=0.0).complete(MESSAGES)
    assert caught.value.reason == "bad_request"


def test_a_refusal_quotes_the_apis_message() -> None:
    with pytest.raises(BackendError, match='The API said: "messages: at least one message"'):
        backend_for(FakeServer(error(400, "messages: at least one message"))).complete(MESSAGES)


# Retries --------------------------------------------------------------------------------------


def test_an_overloaded_api_is_tried_again() -> None:
    clock = FakeClock()
    server = FakeServer(error(529, "Overloaded", "overloaded_error"), ok())
    response = backend_for(server, clock).complete(MESSAGES)
    assert response.attempts == 2
    assert clock.sleeps == [1.0]


def test_retry_after_is_honored() -> None:
    clock = FakeClock()
    server = FakeServer(
        error(429, "rate limited", "rate_limit_error", **{"retry-after": "7"}), ok()
    )
    backend_for(server, clock).complete(MESSAGES)
    assert clock.sleeps == [7.0]


def test_a_wait_longer_than_a_minute_stops_at_once() -> None:
    clock = FakeClock()
    server = FakeServer(error(429, "rate limited", "rate_limit_error", **{"retry-after": "600"}))
    with pytest.raises(BackendError, match="asked to wait 600 seconds") as caught:
        backend_for(server, clock).complete(MESSAGES)
    assert caught.value.reason == "rate_limited"
    assert len(server.requests) == 1
    assert clock.sleeps == []


def test_the_apis_should_retry_header_decides() -> None:
    no = error(500, "do not retry", "api_error", **{"x-should-retry": "false"})
    server = FakeServer(no)
    with pytest.raises(BackendError) as caught:
        backend_for(server).complete(MESSAGES)
    assert caught.value.reason == "server_error"
    assert len(server.requests) == 1
    yes = error(400, "try again", **{"x-should-retry": "true"})
    server = FakeServer(yes, ok())
    assert backend_for(server).complete(MESSAGES).attempts == 2


@pytest.mark.parametrize(
    ("answer", "reason", "words"),
    [
        (error(529, "Overloaded", "overloaded_error"), "server_error", "was overloaded"),
        (error(500, "Internal", "api_error"), "server_error", "a problem of its own"),
        (error(429, "Slow down", "rate_limit_error"), "rate_limited", "too many requests"),
    ],
)
def test_a_problem_that_lasts_is_reported_after_the_retries(answer, reason, words) -> None:
    clock = FakeClock()
    server = FakeServer(answer)
    with pytest.raises(BackendError, match=words) as caught:
        backend_for(server, clock, max_retries=2).complete(MESSAGES)
    assert caught.value.reason == reason
    assert len(server.requests) == 3
    assert clock.sleeps == [1.0, 2.0]


def test_timeouts_and_broken_connections_are_tried_again_then_reported() -> None:
    server = FakeServer(httpx.ReadTimeout("slow"), httpx.ReadError("cut"), ok())
    assert backend_for(server).complete(MESSAGES).attempts == 3
    server = FakeServer(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendError, match="did not answer within 600 seconds") as caught:
        backend_for(server, max_retries=1).complete(MESSAGES)
    assert caught.value.reason == "timeout"
    server = FakeServer(httpx.RemoteProtocolError("cut"))
    with pytest.raises(BackendError, match="broke off") as caught:
        backend_for(server, max_retries=0).complete(MESSAGES)
    assert caught.value.reason == "connection_lost"


def test_no_server_is_reported_at_once() -> None:
    server = FakeServer(httpx.ConnectError("refused"))
    with pytest.raises(BackendError, match="no server answering") as caught:
        backend_for(server).complete(MESSAGES)
    assert caught.value.reason == "no_server"
    assert len(server.requests) == 1
    local = AnthropicBackend(
        "m", api_key_env=KEY_ENV, base_url="http://localhost:9", transport=server.transport
    )
    with pytest.raises(BackendError, match="listens on this port"):
        local.complete(MESSAGES)


# With code() ----------------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", ["per_variable", "per_unit"])
def test_code_works_unchanged_with_both_recipes(recipe: str, tmp_path: Path) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["system"][0]["text"].startswith("You are a careful coder")
        text = '{"tone": "negative"}' if recipe == "per_unit" else '{"code": "negative"}'
        return httpx.Response(200, json=anthropic_message(text))

    codebook = parse_codebook(MINIMAL)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    backend = backend_for(FakeServer(answer))
    run = code(codebook, ["The bus was late again."], backend=backend, ledger=ledger, recipe=recipe)
    assert run.table() == [{"unit": 0, "tone": "negative"}]
    (entry,) = ledger.entries()
    assert entry.outcome == "answered"
    assert entry.prompt_tokens == 50 and entry.completion_tokens == 7
    assert entry.parameters is not None
    assert entry.parameters.api_version == "2026-10-21"


def test_the_backend_closes_and_its_repr_shows_no_key() -> None:
    with backend_for(FakeServer(ok())) as backend:
        backend.complete(MESSAGES)
    assert repr(backend) == f"AnthropicBackend(model='claude-test-model', api_key_env='{KEY_ENV}')"
    with pytest.raises(RuntimeError):
        backend.complete(MESSAGES)

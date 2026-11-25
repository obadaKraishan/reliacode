"""OpenAICompatibleBackend: what it sends and how it reads the answer, against a mock server."""

from datetime import timezone

import httpx
import pytest

from helpers import FakeClock, FakeServer, NetworkBlockedError, completion
from reliacode.backends import Backend, BackendError, OpenAICompatibleBackend, Response

LOCAL = "http://localhost:11434/v1"
MESSAGES = [
    {"role": "system", "content": "Code the tone of the text."},
    {"role": "user", "content": "The new library opens on Monday, and everyone is invited."},
]
SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "enum": ["positive", "negative"]}},
    "required": ["code"],
    "additionalProperties": False,
}


def backend_for(server: FakeServer, **settings: object) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(LOCAL, "tiny-model", transport=server.transport, **settings)


def ok(**fields: object) -> httpx.Response:
    return httpx.Response(200, json=completion(**fields))


# What is sent ------------------------------------------------------------------------------------


def test_the_request_goes_to_the_chat_completions_endpoint() -> None:
    server = FakeServer(ok())
    backend_for(server).complete(MESSAGES)
    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.headers["content-type"] == "application/json"


def test_the_body_holds_the_model_the_messages_and_temperature_zero() -> None:
    server = FakeServer(ok())
    backend_for(server).complete(MESSAGES)
    assert server.body() == {
        "model": "tiny-model",
        "messages": MESSAGES,
        "temperature": 0.0,
        "stream": False,
    }


def test_seed_and_token_limit_are_sent_when_set() -> None:
    server = FakeServer(ok())
    backend_for(server, seed=7, max_tokens=50, temperature=0.2).complete(MESSAGES)
    body = server.body()
    assert (body["seed"], body["max_tokens"], body["temperature"]) == (7, 50, 0.2)


def test_temperature_none_leaves_the_field_out_and_the_record_says_so() -> None:
    server = FakeServer(ok())
    response = backend_for(server, temperature=None).complete(MESSAGES)
    assert server.body() == {"model": "tiny-model", "messages": MESSAGES, "stream": False}
    assert response.parameters.temperature is None


def test_the_default_temperature_is_still_zero_and_still_sent() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert list(server.body()) == ["model", "messages", "temperature", "stream"]
    assert response.parameters.temperature == 0.0


def test_a_schema_is_sent_as_structured_output() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES, json_schema=SCHEMA)
    assert server.body()["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA},
    }
    assert response.parameters.structured_output is True


def test_structured_output_can_be_turned_off() -> None:
    server = FakeServer(ok())
    response = backend_for(server, structured_output=False).complete(MESSAGES, json_schema=SCHEMA)
    assert "response_format" not in server.body()
    assert response.parameters.structured_output is False


def test_no_schema_means_no_response_format() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert "response_format" not in server.body()
    assert response.parameters.structured_output is False


# How the answer is read --------------------------------------------------------------------------


def test_the_answer_becomes_a_full_record() -> None:
    server = FakeServer(httpx.Response(200, json=completion(), headers={"x-request-id": "req-9"}))
    clock = FakeClock(step=0.5)
    response = backend_for(server, clock=clock, seed=3).complete(MESSAGES)
    assert isinstance(response, Response)
    assert response.text == '{"code": "positive"}'
    assert response.model == "tiny-model-served"
    assert response.parameters.model == "tiny-model"
    assert response.parameters.endpoint == "http://localhost:11434/v1/chat/completions"
    assert response.parameters.seed == 3
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        42,
        6,
        48,
    )
    assert response.finish_reason == "stop"
    assert response.response_id == "chatcmpl-test-1"
    assert response.request_id == "req-9"
    assert response.system_fingerprint == "fp_test"
    assert response.latency_seconds == 0.5
    assert response.attempts == 1
    assert response.created_at.tzinfo == timezone.utc


def test_counts_the_server_does_not_give_stay_missing() -> None:
    body = completion()
    del body["usage"], body["system_fingerprint"], body["id"], body["model"]
    response = backend_for(FakeServer(httpx.Response(200, json=body))).complete(MESSAGES)
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        None,
        None,
        None,
    )
    assert response.system_fingerprint is None
    assert response.response_id is None
    assert response.request_id is None
    assert response.model is None


def test_a_partial_or_odd_usage_block_keeps_only_real_counts() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": "6", "total_tokens": True}
    response = backend_for(FakeServer(ok(usage=usage))).complete(MESSAGES)
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        10,
        None,
        None,
    )
    response = backend_for(FakeServer(ok(usage=None))).complete(MESSAGES)
    assert response.prompt_tokens is None
    response = backend_for(FakeServer(ok(usage={"prompt_tokens": -3}))).complete(MESSAGES)
    assert response.prompt_tokens is None


def test_an_empty_content_is_an_empty_text() -> None:
    body = completion()
    body["choices"][0]["message"]["content"] = None
    body["choices"][0]["finish_reason"] = "length"
    response = backend_for(FakeServer(httpx.Response(200, json=body))).complete(MESSAGES)
    assert response.text == ""
    assert response.finish_reason == "length"


def test_content_given_as_text_parts_is_joined() -> None:
    body = completion()
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": '{"code": '},
        {"type": "text", "text": '"negative"}'},
    ]
    response = backend_for(FakeServer(httpx.Response(200, json=body))).complete(MESSAGES)
    assert response.text == '{"code": "negative"}'


# Unexpected answers ------------------------------------------------------------------------------


def _with_content(content: object) -> dict[str, object]:
    body = completion()
    body["choices"][0]["message"]["content"] = content
    return body


@pytest.mark.parametrize(
    ("answer", "what"),
    [
        (httpx.Response(200, text="<html>Welcome to nginx</html>"), "is not JSON"),
        (httpx.Response(200, json=["a", "list"]), "not a JSON object"),
        (httpx.Response(200, json={"object": "list", "data": []}), 'no "choices" list'),
        (httpx.Response(200, json=completion(choices=[])), 'no "choices" list'),
        (httpx.Response(200, json=completion(choices=[{"text": "hi"}])), 'no "message"'),
        (httpx.Response(200, json=_with_content(5)), '"content" is not text'),
        (httpx.Response(200, json=_with_content([{"type": "text", "text": 5}])), "not text"),
    ],
)
def test_an_answer_that_is_not_a_chat_completion_is_explained(answer, what) -> None:
    with pytest.raises(BackendError) as caught:
        backend_for(FakeServer(answer)).complete(MESSAGES)
    error = caught.value
    assert error.reason == "unexpected_answer"
    assert error.endpoint == "http://localhost:11434/v1/chat/completions"
    message = str(error)
    assert "not in the shape of a chat completion" in message
    assert what in message
    assert "/v1" in message


def test_the_unexpected_answer_is_quoted_briefly() -> None:
    long_page = "<html>" + "x" * 1000 + "</html>"
    with pytest.raises(BackendError) as caught:
        backend_for(FakeServer(httpx.Response(200, text=long_page))).complete(MESSAGES)
    assert 'The answer began: "<html>xxx' in str(caught.value)
    assert len(str(caught.value)) < 600


def test_an_error_status_raises_a_backend_error() -> None:
    with pytest.raises(BackendError) as caught:
        backend_for(FakeServer(httpx.Response(400, json={"error": "bad"}))).complete(MESSAGES)
    assert caught.value.status == 400


# Settings ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://localhost:11434/v1/",
        "  http://localhost:11434/v1  ",
        "http://localhost:11434/v1/chat/completions",
    ],
)
def test_the_base_address_is_tidied(base_url: str) -> None:
    backend = OpenAICompatibleBackend(base_url, "tiny-model")
    assert backend.base_url == "http://localhost:11434/v1"
    assert backend.endpoint == "http://localhost:11434/v1/chat/completions"


@pytest.mark.parametrize(
    ("base_url", "error", "match"),
    [
        (None, TypeError, "server's address"),
        ("", TypeError, "server's address"),
        ("localhost:11434/v1", ValueError, "should start with http"),
        ("ftp://files.example.org/v1", ValueError, "should start with http"),
        ("http:///v1", ValueError, "should start with http"),
        ("http://localhost:port/v1", ValueError, "that can be read"),
    ],
)
def test_a_wrong_base_address_is_explained(base_url, error, match) -> None:
    with pytest.raises(error, match=match):
        OpenAICompatibleBackend(base_url, "tiny-model")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:hunter2-not-a-key@api.example.org/v1",
        "https://api.example.org/v1?key=placeholder-value",
        "https://api.example.org/v1#placeholder-value",
    ],
)
def test_an_address_with_a_login_or_query_is_refused_without_quoting_it(base_url: str) -> None:
    with pytest.raises(ValueError, match="does not belong in the address") as caught:
        OpenAICompatibleBackend(base_url, "tiny-model")
    assert "hunter2" not in str(caught.value)
    assert "placeholder-value" not in str(caught.value)


@pytest.mark.parametrize(
    ("settings", "error", "match"),
    [
        ({"model": ""}, TypeError, "name of a model"),
        ({"model": None}, TypeError, "name of a model"),
        ({"timeout": 0}, ValueError, "timeout should be more than 0"),
        ({"timeout": "60"}, TypeError, "timeout should be a number"),
        ({"temperature": -0.5}, ValueError, "temperature should be at least 0"),
        ({"temperature": True}, TypeError, "temperature should be a number"),
        ({"seed": 1.5}, TypeError, "seed should be a whole number"),
        ({"max_tokens": 0}, ValueError, "max_tokens should be at least 1"),
        ({"structured_output": "yes"}, TypeError, "True or False"),
    ],
)
def test_wrong_settings_are_explained(settings, error, match) -> None:
    arguments = {"base_url": LOCAL, "model": "tiny-model", **settings}
    with pytest.raises(error, match=match):
        OpenAICompatibleBackend(**arguments)


def test_the_model_name_is_trimmed() -> None:
    assert OpenAICompatibleBackend(LOCAL, " tiny-model ").model == "tiny-model"


def test_repr_names_the_server_and_the_model() -> None:
    backend = OpenAICompatibleBackend(LOCAL, "tiny-model")
    assert repr(backend) == (
        "OpenAICompatibleBackend(base_url='http://localhost:11434/v1', model='tiny-model', "
        "api_key_env=None)"
    )


def test_it_is_a_backend_and_closes_its_connections() -> None:
    server = FakeServer(ok())
    with backend_for(server) as backend:
        assert isinstance(backend, Backend)
        backend.complete(MESSAGES)
    with pytest.raises(RuntimeError):
        backend.complete(MESSAGES)  # the client is closed


def test_without_a_mock_transport_the_request_is_blocked() -> None:
    # The proof that these tests cannot reach a server: a real transport hits the socket block.
    backend = OpenAICompatibleBackend("http://127.0.0.1:9/v1", "tiny-model")
    with pytest.raises(NetworkBlockedError):
        backend.complete(MESSAGES)

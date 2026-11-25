"""The key for Anthropic's API is a secret: read by variable name at call time, sent only in the
Authorization header, and never stored, logged, shown in an error or kept in a record.

Every test uses a random key made when the test runs, so no key is ever written in a file.
"""

import logging
import secrets
import traceback

import httpx
import pytest

from helpers import FakeServer, anthropic_message
from reliacode import AnthropicBackend, BackendError

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
MESSAGES = [
    {"role": "system", "content": "Code the tone."},
    {"role": "user", "content": "The library added a reading room."},
]


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    value = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, value)
    return value


def backend(server: FakeServer, **settings: object) -> AnthropicBackend:
    return AnthropicBackend(
        "claude-test-model",
        api_key_env=KEY_ENV,
        transport=server.transport,
        sleep=lambda seconds: None,
        **settings,
    )


def everything_about(error: BaseException) -> str:
    shown = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return "\n".join([str(error), repr(error), shown, repr(vars(error))])


def test_the_key_goes_only_into_the_authorization_header(key: str) -> None:
    server = FakeServer(httpx.Response(200, json=anthropic_message()))
    backend(server).complete(MESSAGES)
    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {key}"
    assert key not in str(request.url)
    assert key.encode() not in request.content
    others = [value for name, value in request.headers.items() if name != "authorization"]
    assert all(key not in value for value in others)


def test_the_key_is_read_when_each_request_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    server = FakeServer(httpx.Response(200, json=anthropic_message()))
    made = backend(server)  # made before the variable exists
    first, second = "test-" + secrets.token_urlsafe(24), "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, first)
    made.complete(MESSAGES)
    monkeypatch.setenv(KEY_ENV, f" {second}\n")
    made.complete(MESSAGES)
    assert [r.headers["authorization"] for r in server.requests] == [
        f"Bearer {first}",
        f"Bearer {second}",
    ]


def test_the_key_is_not_kept_on_the_backend_or_in_the_record(key: str) -> None:
    server = FakeServer(httpx.Response(200, json=anthropic_message()))
    made = backend(server)
    response = made.complete(MESSAGES)
    assert key not in repr(made)
    assert all(key not in repr(value) for value in vars(made).values())
    assert "authorization" not in made._client.headers
    assert key not in response.model_dump_json()
    assert key not in repr(response)


def test_the_key_is_not_in_any_log_line(key: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    body = {"type": "error", "error": {"type": "authentication_error", "message": f"bad {key}"}}
    server = FakeServer(
        httpx.Response(529, json={"type": "error", "error": {"message": f"busy {key}"}}),
        httpx.Response(200, json=anthropic_message()),
        httpx.Response(401, json=body),
    )
    made = backend(server)
    made.complete(MESSAGES)
    with pytest.raises(BackendError):
        made.complete(MESSAGES)
    assert caplog.records
    assert key not in caplog.text


@pytest.mark.parametrize(
    ("status", "content"),
    [
        (401, b'{"type": "error", "error": {"message": "invalid x-api-key {key}"}}'),
        (403, b"forbidden for {key}"),
        (400, b'{"type": "error", "error": {"message": "bad request from {key}"}}'),
        (429, b'{"type": "error", "error": {"message": "slow down {key}"}}'),
        (200, b"<html>you sent {key}</html>"),
        (200, b'{"echo": "{key}"}'),
        (200, b'{"type": "error", "error": {"message": "failed for {key}"}}'),
        (302, b""),
    ],
)
def test_an_api_that_echoes_the_key_does_not_put_it_in_the_error(
    key: str, status: int, content: bytes
) -> None:
    headers = {"location": f"https://elsewhere.test/?k={key}"} if status == 302 else {}
    answer = httpx.Response(
        status, content=content.replace(b"{key}", key.encode()), headers=headers
    )
    with pytest.raises(BackendError) as caught:
        backend(FakeServer(answer), max_retries=0).complete(MESSAGES)
    assert key not in everything_about(caught.value)


def test_an_unset_variable_is_named_and_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    server = FakeServer()
    with pytest.raises(BackendError, match=f"The environment variable {KEY_ENV} is not set") as e:
        backend(server).complete(MESSAGES)
    assert e.value.reason == "missing_key"
    assert server.requests == []


def test_a_key_that_cannot_be_sent_is_refused_without_showing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, f"{value} {value}")
    server = FakeServer()
    with pytest.raises(BackendError) as caught:
        backend(server).complete(MESSAGES)
    assert caught.value.reason == "bad_key"
    assert value not in everything_about(caught.value)
    assert server.requests == []


@pytest.mark.parametrize("not_a_name", ["sk-" + "ant-" + "x" * 30, "anthropic_api_key", "", 7])
def test_api_key_env_must_be_a_variable_name_and_is_never_echoed(not_a_name: object) -> None:
    with pytest.raises(ValueError, match="name of the environment variable") as caught:
        AnthropicBackend("claude-test-model", api_key_env=not_a_name)
    if isinstance(not_a_name, str) and not_a_name:
        assert not_a_name not in everything_about(caught.value)

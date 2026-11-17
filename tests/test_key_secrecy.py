"""The API key is a secret: read by variable name at call time, sent only in the Authorization
header, and never stored, logged, shown in an error, kept in a result or written in a fixture.

Every test uses a random key made when the test runs, so no key is ever written in a file.
"""

import logging
import re
import secrets
import traceback
from pathlib import Path

import httpx
import pytest

from helpers import FakeServer, completion
from reliacode.backends import BackendError, OpenAICompatibleBackend

KEY_ENV = "RELIACODE_TEST_API_KEY"
HOSTED = "https://api.example.org/v1"
MESSAGES = [{"role": "user", "content": "The bakery on Elm Street sold out of bread by noon."}]
ROOT = Path(__file__).parent.parent


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    value = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, value)
    return value


def hosted(server: FakeServer, **settings: object) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        HOSTED, "hosted-model", api_key_env=KEY_ENV, transport=server.transport, **settings
    )


def everything_about(error: BaseException) -> str:
    """The error as a user can see it: its message, its repr and the full traceback."""
    shown = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return "\n".join([str(error), repr(error), shown, repr(vars(error))])


# Sent only in the Authorization header -----------------------------------------------------------


def test_the_key_is_sent_only_in_the_authorization_header(key: str) -> None:
    server = FakeServer(httpx.Response(200, json=completion()))
    hosted(server).complete(MESSAGES)
    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {key}"
    assert key not in str(request.url)
    assert key.encode() not in request.content
    others = [value for name, value in request.headers.items() if name != "authorization"]
    assert all(key not in value for value in others)


def test_a_local_server_gets_no_authorization_header() -> None:
    server = FakeServer(httpx.Response(200, json=completion()))
    OpenAICompatibleBackend(HOSTED, "m", transport=server.transport).complete(MESSAGES)
    assert "authorization" not in server.requests[0].headers


def test_the_key_is_read_when_each_request_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    server = FakeServer(httpx.Response(200, json=completion()))
    backend = hosted(server)  # made before the variable exists
    first, second = "test-" + secrets.token_urlsafe(24), "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, first)
    backend.complete(MESSAGES)
    monkeypatch.setenv(KEY_ENV, f"  {second}\n")  # surrounding spaces and a line break are trimmed
    backend.complete(MESSAGES)
    assert [r.headers["authorization"] for r in server.requests] == [
        f"Bearer {first}",
        f"Bearer {second}",
    ]


# Never stored, logged or kept in a result --------------------------------------------------------


def test_the_key_is_not_kept_on_the_backend(key: str) -> None:
    server = FakeServer(httpx.Response(200, json=completion()))
    backend = hosted(server)
    backend.complete(MESSAGES)
    assert key not in repr(backend)
    assert "RELIACODE_TEST_API_KEY" in repr(backend)  # the name is shown, the value is not
    assert all(key not in repr(value) for value in vars(backend).values())
    assert "authorization" not in backend._client.headers


def test_the_key_is_not_in_the_response_record(key: str) -> None:
    server = FakeServer(httpx.Response(200, json=completion()))
    response = hosted(server).complete(MESSAGES)
    assert key not in response.model_dump_json()
    assert key not in repr(response)


def test_the_key_is_not_in_any_log_line(key: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, httpx and httpcore included
    server = FakeServer(
        httpx.Response(200, json=completion()),
        httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {key}"}}),
    )
    backend = hosted(server)
    backend.complete(MESSAGES)
    with pytest.raises(BackendError):
        backend.complete(MESSAGES)
    assert caplog.records, "the backend should log its requests at DEBUG level"
    assert key not in caplog.text
    assert all(key not in record.getMessage() for record in caplog.records)


# Never shown in an error -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(401, json={"error": {"message": "Incorrect API key provided: {key}"}}),
        httpx.Response(403, text="forbidden for {key}"),
        httpx.Response(400, json={"error": {"message": "bad request from {key}"}}),
        httpx.Response(200, text="<html>you sent {key}</html>"),
        httpx.Response(200, json={"echo": "{key}"}),
    ],
    ids=["401-echo", "403-echo", "400-echo", "html-echo", "json-echo"],
)
def test_a_server_that_echoes_the_key_does_not_put_it_in_the_error(
    key: str, answer: httpx.Response
) -> None:
    echoed = httpx.Response(
        answer.status_code,
        headers={"content-type": answer.headers.get("content-type", "text/plain")},
        content=answer.content.replace(b"{key}", key.encode()),
    )
    with pytest.raises(BackendError) as caught:
        hosted(FakeServer(echoed)).complete(MESSAGES)
    assert key not in everything_about(caught.value)


def test_an_unset_variable_is_named_but_no_value_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(BackendError) as caught:
        hosted(FakeServer()).complete(MESSAGES)
    error = caught.value
    assert error.reason == "missing_key"
    assert f"The environment variable {KEY_ENV} is not set" in str(error)
    assert "export RELIACODE_TEST_API_KEY=..." in str(error)


def test_an_empty_variable_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "   ")
    server = FakeServer()
    with pytest.raises(BackendError, match="is set but empty"):
        hosted(server).complete(MESSAGES)
    assert server.requests == []


@pytest.mark.parametrize("damage", ["{key} {key}", "{key}\nsecond-line", "{key}é"])
def test_a_key_that_cannot_be_sent_is_refused_without_showing_it(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    value = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, damage.format(key=value))
    server = FakeServer()
    with pytest.raises(BackendError) as caught:
        hosted(server).complete(MESSAGES)
    assert caught.value.reason == "bad_key"
    assert value not in everything_about(caught.value)
    assert server.requests == []


@pytest.mark.parametrize(
    "not_a_name",
    [
        "sk-" + "or-v1-" + "0" * 32,  # a key passed by mistake (built here, never written whole)
        "gsk_" + "a1" * 20,
        "openrouter_api_key",
        "MY-KEY",
        "",
        42,
    ],
)
def test_api_key_env_must_be_a_variable_name_and_is_never_echoed(not_a_name: object) -> None:
    with pytest.raises(ValueError, match="name of the environment variable") as caught:
        OpenAICompatibleBackend(HOSTED, "m", api_key_env=not_a_name)
    if isinstance(not_a_name, str) and not_a_name:
        assert not_a_name not in everything_about(caught.value)


# Never written in a fixture or a file ------------------------------------------------------------

#: Shapes of real keys (the prefixes are split so this file does not match itself).
KEY_SHAPES = [
    re.compile("sk-" + r"(?:ant-|or-v1-|proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile("gsk" + r"_[A-Za-z0-9]{20,}"),
    re.compile("hf" + r"_[A-Za-z0-9]{20,}"),
    re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    re.compile("AKIA" + r"[0-9A-Z]{16}"),
    re.compile("AIza" + r"[0-9A-Za-z_\-]{30,}"),
    re.compile("Bearer" + r" [A-Za-z0-9_\-.]{20,}"),
]


def tracked_text_files() -> list[Path]:
    folders = ["src", "tests", "docs", "examples", ".github"]
    files = [path for folder in folders for path in (ROOT / folder).rglob("*") if path.is_file()]
    files += [ROOT / name for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md")]
    text_types = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg"}
    return [path for path in files if path.suffix in text_types]


def test_no_file_in_the_repository_holds_a_key() -> None:
    files = tracked_text_files()
    assert any(path.name == "test_key_secrecy.py" for path in files)
    for path in files:
        text = path.read_text(encoding="utf-8")
        for shape in KEY_SHAPES:
            assert not shape.search(text), f"{path.relative_to(ROOT)} holds a key-like string"

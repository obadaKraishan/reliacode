"""Extra request fields: sent as given, recorded in every response and ledger line, and refused
when the backend sets the field itself or the name looks like a credential."""

import json
from pathlib import Path

import httpx
import pytest

from helpers import CODEBOOK_HASH, PROMPT_HASH, FakeServer, completion
from reliacode import Ledger, LedgerEntry, OpenAICompatibleBackend, RequestParameters, read_ledger
from reliacode.backends import CredentialFieldError

LOCAL = "http://localhost:8000/v1"
MESSAGES = [{"role": "user", "content": "The river path reopened after the flood."}]
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}


def backend_for(server: FakeServer, **settings: object) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(LOCAL, "gpt-oss-120b", transport=server.transport, **settings)


def ok() -> httpx.Response:
    return httpx.Response(200, json=completion())


# Sent as given, and recorded ---------------------------------------------------------------------


def test_extra_fields_are_sent_as_given() -> None:
    server = FakeServer(ok())
    extra = {
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
    }
    backend_for(server, extra_body=extra).complete(MESSAGES)
    body = server.body()
    assert {name: body[name] for name in extra} == extra
    assert body["model"] == "gpt-oss-120b"
    assert body["temperature"] == 0.0


def test_extra_fields_come_after_the_backends_own_fields() -> None:
    server = FakeServer(ok())
    backend_for(server, extra_body={"reasoning_effort": "high"}, seed=3).complete(MESSAGES)
    assert list(server.body()) == [
        "model",
        "messages",
        "temperature",
        "stream",
        "seed",
        "reasoning_effort",
    ]


def test_extra_fields_are_recorded_in_the_response() -> None:
    extra = {"thinking": {"type": "enabled", "budget_tokens": 2000}}
    response = backend_for(FakeServer(ok()), extra_body=extra, temperature=None).complete(MESSAGES)
    assert response.parameters.extra_body == extra
    assert response.parameters.temperature is None


def test_without_extra_fields_the_record_holds_none_and_the_body_is_as_before() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert response.parameters.extra_body == {}
    assert server.body() == {
        "model": "gpt-oss-120b",
        "messages": MESSAGES,
        "temperature": 0.0,
        "stream": False,
    }


def test_extra_fields_are_recorded_in_the_ledger_line(tmp_path: Path) -> None:
    extra = {"reasoning_effort": "medium"}
    response = backend_for(FakeServer(ok()), extra_body=extra).complete(MESSAGES)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(
        LedgerEntry.from_response(response, outcome="answered", text="An invented text.", **WORK)
    )
    line = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert line["parameters"]["extra_body"] == extra
    assert read_ledger(tmp_path / "ledger.jsonl")[0].parameters.extra_body == extra


def test_the_fields_are_copied_when_the_backend_is_made() -> None:
    server = FakeServer(ok())
    extra: dict[str, object] = {"chat_template_kwargs": {"enable_thinking": False}}
    backend = backend_for(server, extra_body=extra)
    extra["chat_template_kwargs"]["enable_thinking"] = True  # type: ignore[index]
    extra["top_k"] = 5
    response = backend.complete(MESSAGES)
    response.parameters.extra_body["chat_template_kwargs"]["enable_thinking"] = "changed"
    backend.extra_body["top_k"] = 9
    assert server.body()["chat_template_kwargs"] == {"enable_thinking": False}
    assert "top_k" not in server.body()
    assert backend.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert backend.complete(MESSAGES).parameters.extra_body == backend.extra_body


@pytest.mark.parametrize(
    "extra",
    [
        {"max_completion_tokens": 400},
        {"thinking": {"type": "enabled", "budget_tokens": 2000}},
        {"min_tokens": 1, "skip_special_tokens": True},
        {"structured_outputs": {"choice": ["a", "b"]}},
        {"tags": [{"name": "pilot"}, {"tokens_seen": 3}]},
    ],
)
def test_counts_of_tokens_are_not_credentials(extra: dict[str, object]) -> None:
    server = FakeServer(ok())
    backend_for(server, extra_body=extra).complete(MESSAGES)
    assert {name: server.body()[name] for name in extra} == extra


# Refused -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "instead"),
    [
        ("model", "model argument"),
        ("messages", "given to complete()"),
        ("stream", '"stream": false'),
        ("temperature", "temperature=None leaves it out"),
        ("seed", "seed setting"),
        ("max_tokens", "max_tokens setting"),
        ("response_format", "structured_output=True"),
    ],
)
def test_a_field_the_backend_sets_itself_is_refused(name: str, instead: str) -> None:
    with pytest.raises(ValueError, match="a field the backend sets itself") as caught:
        OpenAICompatibleBackend(LOCAL, "m", extra_body={name: 1})
    assert f'"{name}"' in str(caught.value)
    assert instead in str(caught.value)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("reasoning_effort=low", "mapping of field names to values"),
        ([("reasoning_effort", "low")], "mapping of field names to values"),
        ({"when": object()}, "only JSON values"),
        ({"weight": float("nan")}, "only JSON values"),
        ({"nested": {1: "a"}}, "field name in extra_body should be text"),
        ({2: "a"}, "field name in extra_body should be text"),
    ],
)
def test_extra_fields_that_are_not_json_are_explained(extra: object, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        OpenAICompatibleBackend(LOCAL, "m", extra_body=extra)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        ({"api_key": "x"}, '"api_key"'),
        ({"Authorization": "x"}, '"Authorization"'),
        ({"X-Api-Key": "x"}, '"X-Api-Key"'),
        ({"access_token": "x"}, '"access_token"'),
        ({"TOKEN": "x"}, '"TOKEN"'),
        ({"client_secret": "x"}, '"client_secret"'),
        ({"Password": "x"}, '"Password"'),
        ({"metadata": {"session": {"refresh_token": "x"}}}, '"metadata.session.refresh_token"'),
        ({"headers": [{"other": 1}, {"authorization": "x"}]}, '"headers.authorization"'),
        ({"keywords": ["a"]}, '"keywords"'),
        ({"a key with spaces": "x"}, "A field in extra_body"),
    ],
)
def test_a_name_that_looks_like_a_credential_is_refused(extra: dict, named: str) -> None:
    with pytest.raises(ValueError, match="looks like a credential") as caught:
        OpenAICompatibleBackend(LOCAL, "m", extra_body=extra)
    message = str(caught.value)
    assert message.startswith(f"The extra request field {named}") or message.startswith(named)
    assert "nothing was sent" in message
    assert "api_key_env" in message


@pytest.mark.parametrize(
    "extra",
    [
        {"auth": "x"},
        {"x_auth_header": "x"},
        {"oauth": {"state": "x"}},
        {"Credentials": "x"},
        {"service_credential_id": "x"},
        {"bearer": "x"},
        {"BearerValue": "x"},
        {"cookie": "x"},
        {"metadata": {"set_cookie": "x"}},
        {"author": "x"},  # holds "auth": refused too, by the plain rule
    ],
)
def test_names_with_auth_credential_bearer_or_cookie_are_refused(extra: dict) -> None:
    server = FakeServer(ok())
    with pytest.raises(ValueError, match="auth, credential, bearer or cookie"):
        backend_for(server, extra_body=extra)
    assert server.requests == []
    with pytest.raises(CredentialFieldError):
        RequestParameters(endpoint="e", model="m", extra_body=extra)


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "low"},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"metadata": {"user_id": "study-a"}},
        {"top_k": 20, "repetition_penalty": 1.05},
    ],
)
def test_ordinary_names_are_still_sent(extra: dict) -> None:
    server = FakeServer(ok())
    backend_for(server, extra_body=extra).complete(MESSAGES)
    assert {name: server.body()[name] for name in extra} == extra


def test_a_record_refuses_a_credential_name_too() -> None:
    with pytest.raises(CredentialFieldError, match="looks like a credential"):
        RequestParameters(endpoint="e", model="m", extra_body={"api_key": "x"})
    assert issubclass(CredentialFieldError, TypeError)

"""The ledger never holds the text that was coded, the model's answer or the API key.

Coded texts are often private, so a ledger line ties a call to its text only by a hash of the
text and its length in characters. These tests put random secrets in the text, the answer and
the key, and look for them everywhere the ledger writes or shows something: the file, the
reprs, the records, the errors and the log. Every secret is made when the test runs.
"""

import json
import logging
import secrets
from pathlib import Path

import httpx
import pytest

from helpers import CODEBOOK_HASH, PROMPT_HASH, FakeServer, completion
from reliacode.backends import BackendError, OpenAICompatibleBackend
from reliacode.ledger import Ledger, LedgerEntry, LedgerError, hash_text, read_ledger

KEY_ENV = "RELIACODE_TEST_API_KEY"
HOSTED = "https://api.example.org/v1"
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}


class Secrets:
    def __init__(self) -> None:
        self.text_mark = secrets.token_hex(12)
        self.answer_mark = secrets.token_hex(12)
        self.key = "test-" + secrets.token_urlsafe(24)
        self.text = f"A private comment that names {self.text_mark} and should stay private."
        self.answer = json.dumps({"code": "positive", "note": f"echo {self.answer_mark}"})

    def marks(self) -> list[str]:
        return [self.text_mark, self.answer_mark, self.key]

    def assert_absent(self, shown: str, where: str) -> None:
        for mark in self.marks():
            assert mark not in shown, f"a secret reached {where}"


@pytest.fixture
def secret(monkeypatch: pytest.MonkeyPatch) -> Secrets:
    made = Secrets()
    monkeypatch.setenv(KEY_ENV, made.key)
    return made


def hosted(server: FakeServer) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        HOSTED, "hosted-model", api_key_env=KEY_ENV, transport=server.transport, max_retries=0
    )


def messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Text to code:\n\n{text}"}]


def record_a_call_and_a_failure(ledger: Ledger, secret: Secrets) -> None:
    """One call that is answered (the answer echoes a secret), and one that fails with a server
    message that echoes the text and the key."""
    answered = hosted(FakeServer(httpx.Response(200, json=completion(secret.answer))))
    reply = answered.complete(messages(secret.text))
    ledger.record(LedgerEntry.from_response(reply, outcome="answered", text=secret.text, **WORK))

    echo = {"error": {"message": f"Cannot read {secret.text} with key {secret.key}"}}
    failing = hosted(FakeServer(httpx.Response(400, json=echo)))
    with pytest.raises(BackendError) as caught:
        failing.complete(messages(secret.text))
    ledger.record(
        LedgerEntry.from_failure(caught.value, model="hosted-model", text=secret.text, **WORK)
    )


def test_no_secret_reaches_the_file(tmp_path: Path, secret: Secrets) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record_a_call_and_a_failure(ledger, secret)
    raw = (tmp_path / "ledger.jsonl").read_bytes()
    assert raw.count(b"\n") == 2
    for mark in secret.marks():
        assert mark.encode() not in raw
    assert b"Cannot read" not in raw  # the server's message is not kept at all


def test_the_text_is_kept_only_as_its_hash_and_length(tmp_path: Path, secret: Secrets) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record_a_call_and_a_failure(ledger, secret)
    for entry in read_ledger(tmp_path / "ledger.jsonl"):
        assert entry.text_hash == hash_text(secret.text)
        assert entry.text_length == len(secret.text)


def test_no_secret_reaches_a_repr_or_a_record(tmp_path: Path, secret: Secrets) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record_a_call_and_a_failure(ledger, secret)
    entries = ledger.entries()
    shown = [repr(ledger), str(ledger), repr(entries), str(entries)]
    shown += [entry.model_dump_json() for entry in entries]
    shown += [json.dumps(entry.model_dump(mode="json")) for entry in entries]
    shown += [repr(vars(ledger))]
    secret.assert_absent("\n".join(shown), "a repr or a record")


def test_no_secret_reaches_a_log_line(
    tmp_path: Path, secret: Secrets, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, at every level
    ledger = Ledger(tmp_path / "ledger.jsonl")
    record_a_call_and_a_failure(ledger, secret)
    ledger.entries()
    logged = "\n".join(
        f"{record.name} {record.getMessage()} {record.args!r}" for record in caplog.records
    )
    secret.assert_absent(logged, "a log line")


def test_an_unreadable_line_is_reported_without_quoting_it(tmp_path: Path, secret: Secrets) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path)
    line = LedgerEntry.from_response(
        hosted(FakeServer(httpx.Response(200, json=completion()))).complete(messages("x")),
        outcome="answered",
        text="x",
        **WORK,
    ).model_dump(mode="json")
    edited = [
        {**line, "finish_reason": 3, "variable": [secret.text]},  # a wrong value of each kind
        {**line, "outcome": secret.answer},
        {**line, "text_hash": secret.key},
        {**line, "text": secret.text},  # a field no line has
        {**line, f"note {secret.text_mark}": 1},  # a field name that is not a name
    ]
    text = "\n".join(json.dumps(data) for data in edited)
    path.write_text(text + f"\n{{cut off {secret.answer_mark}\n", encoding="utf-8")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    message = str(caught.value)
    assert len(caught.value.problems) == 6
    assert "text: Extra inputs are not permitted" in message  # an ordinary name is named
    assert "a field: Extra inputs are not permitted" in message
    secret.assert_absent(message + repr(caught.value.problems), "an error message")

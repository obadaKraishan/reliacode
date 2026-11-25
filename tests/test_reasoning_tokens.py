"""Reasoning (or thinking) tokens are counted in a field of their own when the server reports
them, and stay missing when it does not: never zero by default."""

import json
import shutil
from pathlib import Path

import httpx
import pytest

from helpers import (
    CODEBOOK_HASH,
    EXAMPLES,
    FIXTURES,
    PROMPT_HASH,
    FakeServer,
    completion,
    response,
)
from reliacode import (
    FakeBackend,
    Ledger,
    LedgerEntry,
    OpenAICompatibleBackend,
    code,
    load_codebook,
    read_ledger,
)
from reliacode.ledger import LEDGER_VERSION

MESSAGES = [{"role": "user", "content": "The orchard opened its gates to school groups."}]
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}


def answered(usage: object) -> object:
    server = FakeServer(httpx.Response(200, json=completion(usage=usage)))
    backend = OpenAICompatibleBackend("http://localhost:8000/v1", "m", transport=server.transport)
    return backend.complete(MESSAGES)


def test_reasoning_tokens_are_read_from_the_completion_details() -> None:
    usage = {
        "prompt_tokens": 300,
        "completion_tokens": 420,
        "total_tokens": 720,
        "completion_tokens_details": {"reasoning_tokens": 400},
    }
    reply = answered(usage)
    assert (reply.prompt_tokens, reply.completion_tokens, reply.reasoning_tokens) == (300, 420, 400)


@pytest.mark.parametrize(
    "details",
    [
        None,
        {},
        {"accepted_prediction_tokens": 0},
        {"reasoning_tokens": None},
        {"reasoning_tokens": "400"},
        {"reasoning_tokens": -1},
        {"reasoning_tokens": True},
        [400],
    ],
)
def test_a_missing_or_odd_reasoning_count_stays_missing(details: object) -> None:
    usage: dict[str, object] = {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}
    if details is not None:
        usage["completion_tokens_details"] = details
    assert answered(usage).reasoning_tokens is None


def test_a_reported_zero_is_kept_as_zero() -> None:
    usage = {"prompt_tokens": 30, "completion_tokens": 4}
    usage["completion_tokens_details"] = {"reasoning_tokens": 0}  # type: ignore[assignment]
    assert answered(usage).reasoning_tokens == 0


def test_no_usage_at_all_means_no_reasoning_count() -> None:
    assert answered(None).reasoning_tokens is None


def test_the_ledger_line_keeps_the_reasoning_count() -> None:
    reply = response(reasoning_tokens=250, completion_tokens=260)
    entry = LedgerEntry.from_response(reply, outcome="answered", text="An invented text.", **WORK)
    assert (entry.completion_tokens, entry.reasoning_tokens) == (260, 250)
    assert entry.token_source == "server"
    missing = LedgerEntry.from_response(
        response(), outcome="answered", text="An invented text.", **WORK
    )
    assert missing.reasoning_tokens is None


def test_a_reasoning_count_alone_still_comes_from_the_server() -> None:
    reply = response(prompt_tokens=None, completion_tokens=None, reasoning_tokens=12)
    entry = LedgerEntry.from_response(reply, outcome="answered", text="Invented.", **WORK)
    assert entry.token_source == "server"
    line = {**entry.model_dump(mode="json"), "token_source": None}
    with pytest.raises(ValueError, match='token_source="server"'):
        LedgerEntry.model_validate(line)


def test_a_failed_call_has_no_reasoning_count() -> None:
    failed = LedgerEntry.from_failure(RuntimeError("x"), text="Invented.", **WORK)
    line = {**failed.model_dump(mode="json"), "reasoning_tokens": 3, "token_source": "server"}
    with pytest.raises(ValueError, match="a failed call got no answer"):
        LedgerEntry.model_validate(line)


# Ledger version 2, and the lines of version 1 ----------------------------------------------------

V1_FILE = FIXTURES / "ledger_v1.jsonl"


def test_new_lines_are_version_2() -> None:
    entry = LedgerEntry.from_response(response(), outcome="answered", text="Invented.", **WORK)
    assert LEDGER_VERSION == 2
    assert entry.model_dump(mode="json")["schema_version"] == 2


def test_a_ledger_written_with_version_1_is_still_read() -> None:
    # Written by reliacode before version 2, with an invented codebook run on FakeBackend.
    raw = [json.loads(line) for line in V1_FILE.read_text(encoding="utf-8").splitlines()]
    assert [line["schema_version"] for line in raw] == [1, 1, 1, 1]
    assert all("reasoning_tokens" not in line for line in raw)
    entries = read_ledger(V1_FILE)
    assert [entry.outcome for entry in entries] == ["answered", "asked_again", "answered", "failed"]
    assert [entry.schema_version for entry in entries] == [1, 1, 1, 1]
    assert all(entry.reasoning_tokens is None for entry in entries)
    assert [entry.parameters.extra_body for entry in entries[:3]] == [{}, {}, {}]
    assert entries[0].prompt_tokens == 120


def test_new_lines_can_follow_version_1_lines_in_one_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    shutil.copy(V1_FILE, path)
    ledger = Ledger(path)
    assert ledger.calls == 4
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    backend = FakeBackend(['{"code": "conflict"}', '{"code": "government"}'], tokens=(50, 5))
    code(codebook, ["An invented third text."], backend=backend, ledger=ledger)
    versions = [entry.schema_version for entry in read_ledger(path)]
    assert versions == [1, 1, 1, 1, 2, 2]
    assert ledger.summary().total.calls == 6

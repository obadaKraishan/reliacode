"""The ledger record: one frozen line per call, with a schema version, that never keeps the text,
the answer or the key."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from helpers import CODEBOOK_HASH, NOON, PROMPT_HASH, response
from reliacode.backends import BackendError
from reliacode.ledger import LEDGER_VERSION, LedgerEntry, hash_text

TEXT = "The bakery on Elm Street sold out of bread by noon."
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}


def answered(**changes: object) -> LedgerEntry:
    return LedgerEntry.from_response(response(), outcome="answered", text=TEXT, **WORK, **changes)


def line(**changes: object) -> dict[str, object]:
    """A ledger line as JSON data, to build entries by hand."""
    data = answered().model_dump(mode="json")
    data.update(changes)
    return data


# The text hash -----------------------------------------------------------------------------------


def test_the_text_hash_is_sha256_of_the_utf8_text() -> None:
    expected = "sha256:" + hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    assert hash_text(TEXT) == expected


def test_the_text_hash_changes_with_any_change_in_the_text() -> None:
    assert hash_text(TEXT) != hash_text(TEXT + " ")
    assert hash_text("Café") != hash_text("Café")  # the text is hashed exactly as given


def test_the_text_hash_needs_a_text() -> None:
    with pytest.raises(TypeError, match="takes a text"):
        hash_text(b"bytes")  # type: ignore[arg-type]


# A call with an answer ---------------------------------------------------------------------------


def test_a_line_from_an_answer_keeps_what_the_call_used() -> None:
    entry = LedgerEntry.from_response(
        response(
            model="example-model-2026",
            response_id="chatcmpl-1",
            request_id="req-1",
            attempts=2,
        ),
        outcome="asked_again",
        text=TEXT,
        ask=1,
        **WORK,
    )
    assert entry.schema_version == LEDGER_VERSION == 1
    assert entry.outcome == "asked_again"
    assert entry.variable == "tone"
    assert entry.model_requested == "example-model-2026"
    assert entry.model_returned == "example-model-2026"
    assert entry.parameters is not None
    assert entry.parameters.endpoint.endswith("/chat/completions")
    assert (entry.prompt_tokens, entry.completion_tokens) == (40, 6)
    assert entry.token_source == "server"
    assert entry.latency_seconds == 0.25
    assert entry.finish_reason == "stop"
    assert (entry.response_id, entry.request_id, entry.attempts) == ("chatcmpl-1", "req-1", 2)
    assert (entry.codebook_hash, entry.prompt_hash) == (CODEBOOK_HASH, PROMPT_HASH)
    assert entry.text_hash == hash_text(TEXT)
    assert entry.text_length == len(TEXT)
    assert (entry.error_reason, entry.http_status) == (None, None)


def test_the_model_asked_for_and_the_model_returned_are_both_kept() -> None:
    reply = response(model="example-model")
    served = reply.model_copy(update={"model": "example-model-0613"})
    entry = LedgerEntry.from_response(served, outcome="answered", text=TEXT, **WORK)
    assert entry.model_requested == "example-model"
    assert entry.model_returned == "example-model-0613"
    assert entry.model == "example-model"


def test_the_time_of_the_answer_is_used_unless_another_is_given() -> None:
    assert answered().at == NOON
    later = datetime(2026, 11, 19, 13, 0, tzinfo=timezone.utc)
    assert answered(at=later).at == later


def test_counts_the_server_did_not_report_stay_missing() -> None:
    reply = response(prompt_tokens=None, completion_tokens=None)
    entry = LedgerEntry.from_response(reply, outcome="answered", text=TEXT, **WORK)
    assert (entry.prompt_tokens, entry.completion_tokens) == (None, None)
    assert entry.token_source is None
    assert entry.counted is False


def test_a_line_with_one_count_is_marked_but_not_counted() -> None:
    reply = response(prompt_tokens=40, completion_tokens=None)
    entry = LedgerEntry.from_response(reply, outcome="answered", text=TEXT, **WORK)
    assert entry.token_source == "server"
    assert entry.counted is False


# A call without an answer ------------------------------------------------------------------------


def test_a_failed_call_keeps_the_kind_of_problem_and_not_the_message() -> None:
    error = BackendError(
        f"The server said: cannot read {TEXT}", reason="bad_request", endpoint="x", status=400
    )
    entry = LedgerEntry.from_failure(error, model="example-model", text=TEXT, ask=2, **WORK)
    assert entry.outcome == "failed"
    assert (entry.error_reason, entry.http_status) == ("bad_request", 400)
    assert entry.model_requested == "example-model"
    assert entry.ask == 2
    assert entry.prompt_tokens is None
    assert entry.token_source is None
    assert entry.latency_seconds is None
    assert entry.attempts is None
    assert TEXT not in entry.model_dump_json()
    assert "cannot read" not in entry.model_dump_json()


def test_another_exception_is_kept_by_its_name() -> None:
    entry = LedgerEntry.from_failure(RuntimeError(TEXT), text=TEXT, **WORK)
    assert (entry.error_reason, entry.http_status) == ("RuntimeError", None)
    assert entry.model_requested is None
    assert entry.model is None
    assert entry.at.tzinfo == timezone.utc


# What a line may hold ----------------------------------------------------------------------------


def test_the_line_has_no_field_for_the_text_the_answer_or_the_key() -> None:
    fields = set(LedgerEntry.model_fields)
    assert not fields & {"text", "texts", "messages", "answer", "answers", "content", "api_key"}
    assert {"text_hash", "text_length"} <= fields


def test_the_line_is_frozen() -> None:
    entry = answered()
    with pytest.raises(ValidationError):
        entry.outcome = "missing"  # type: ignore[misc]


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        LedgerEntry.model_validate(line(text=TEXT))


def test_the_time_needs_a_time_zone_and_is_kept_in_utc() -> None:
    with pytest.raises(ValidationError, match="time zone"):
        LedgerEntry.model_validate(line(at="2026-11-19T12:00:00"))
    entry = LedgerEntry.model_validate(line(at="2026-11-19T14:00:00+02:00"))
    assert entry.at == NOON
    assert entry.at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("field", ["codebook_hash", "prompt_hash", "text_hash"])
def test_hashes_must_be_sha256_hashes(field: str) -> None:
    with pytest.raises(ValidationError, match="should match pattern"):
        LedgerEntry.model_validate(line(**{field: TEXT}))


def test_a_newer_schema_version_is_refused_in_plain_words() -> None:
    with pytest.raises(ValidationError, match="Update reliacode to read it"):
        LedgerEntry.model_validate(line(schema_version=LEDGER_VERSION + 1))
    with pytest.raises(ValidationError, match="version 0"):
        LedgerEntry.model_validate(line(schema_version=0))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"token_source": None}, "token counts need"),
        (
            {"prompt_tokens": None, "completion_tokens": None, "token_source": "server"},
            "no token counts",
        ),
        ({"outcome": "failed"}, "needs an error_reason"),
        ({"outcome": "failed", "error_reason": "timeout"}, "has no token counts"),
        ({"error_reason": "timeout"}, "only a failed call"),
        ({"http_status": 500}, "only a failed call"),
    ],
)
def test_a_line_must_be_consistent(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LedgerEntry.model_validate(line(**changes))


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "latency_seconds"])
def test_counts_and_latency_cannot_be_negative(field: str) -> None:
    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(line(**{field: -1}))


def test_a_line_survives_a_trip_through_json() -> None:
    entry = answered(ask=2)
    assert LedgerEntry.model_validate(json.loads(json.dumps(entry.model_dump(mode="json")))) == (
        entry
    )

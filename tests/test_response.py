"""The response record: immutable, strict, in UTC, and honest about what the server did not say."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from reliacode.backends import Backend, RequestParameters, Response

PARAMETERS = RequestParameters(
    endpoint="http://localhost:11434/v1/chat/completions", model="tiny-model", temperature=0.0
)
WHEN = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def make(**changes: object) -> Response:
    fields: dict[str, object] = {
        "text": '{"code": "positive"}',
        "parameters": PARAMETERS,
        "latency_seconds": 0.25,
        "created_at": WHEN,
    }
    fields.update(changes)
    return Response(**fields)


def test_missing_counts_stay_missing() -> None:
    response = make()
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None
    assert response.model is None
    assert response.finish_reason is None
    assert response.response_id is None
    assert response.request_id is None
    assert response.system_fingerprint is None
    assert response.attempts == 1


def test_counts_are_kept_as_given() -> None:
    response = make(prompt_tokens=0, completion_tokens=7, total_tokens=7)
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (0, 7, 7)


def test_the_record_is_frozen() -> None:
    response = make()
    with pytest.raises(ValidationError):
        response.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        PARAMETERS.model = "other"  # type: ignore[misc]


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError, match="extra"):
        make(api_key="anything")
    with pytest.raises(ValidationError, match="extra"):
        RequestParameters(endpoint="e", model="m", headers={})


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_negative_counts_are_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        make(**{field: -1})


def test_latency_and_attempts_are_checked() -> None:
    with pytest.raises(ValidationError):
        make(latency_seconds=-0.1)
    with pytest.raises(ValidationError):
        make(attempts=0)


def test_timestamps_are_stored_in_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    response = make(created_at=datetime(2026, 11, 17, 7, 0, tzinfo=eastern))
    assert response.created_at == WHEN
    assert response.created_at.utcoffset() == timedelta(0)


def test_a_timestamp_without_a_time_zone_is_refused() -> None:
    with pytest.raises(ValidationError, match="time zone"):
        make(created_at=datetime(2026, 11, 17, 12, 0))


def test_the_record_survives_a_json_round_trip() -> None:
    response = make(prompt_tokens=12, finish_reason="stop", response_id="chatcmpl-1")
    assert Response.model_validate_json(response.model_dump_json()) == response


def test_anything_with_a_complete_method_is_a_backend() -> None:
    class Echo:
        def complete(self, messages, *, json_schema=None):
            return make(text=messages[-1]["content"])

    class NotABackend:
        def answer(self, messages):
            return None

    assert isinstance(Echo(), Backend)
    assert not isinstance(NotABackend(), Backend)

"""Backends: where prompts are sent and where answers come from.

A backend takes a list of chat messages and returns a :class:`Response`, an immutable record of
the answer and of how it was obtained. The rest of reliacode depends only on the :class:`Backend`
interface, so any object with a matching ``complete`` method can be used as a backend.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

#: A chat message: ``{"role": "system" | "user" | "assistant", "content": "..."}``.
Message = Mapping[str, str]


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(timezone.utc)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestParameters(_Record):
    """What was asked of a backend: where the request went, which model, and which settings."""

    endpoint: str
    model: str
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    structured_output: bool = False


class Response(_Record):
    """One answer from a backend, with everything needed to report how it was obtained.

    Token counts, ids and the fingerprint are ``None`` when the server did not report them: a
    missing count is never recorded as zero. ``model`` is the model name as the server returned
    it, which can differ from the name that was asked for (``parameters.model``).
    """

    text: str
    model: str | None = None
    parameters: RequestParameters
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    system_fingerprint: str | None = None
    latency_seconds: float = Field(ge=0)
    attempts: int = Field(default=1, ge=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at needs a time zone; use datetime.now(timezone.utc)")
        return value.astimezone(timezone.utc)


@runtime_checkable
class Backend(Protocol):
    """Anything that can answer chat messages: the interface the rest of reliacode uses."""

    def complete(
        self, messages: Sequence[Message], *, json_schema: Mapping[str, Any] | None = None
    ) -> Response:
        """Send ``messages`` and return the answer.

        Args:
            messages: the conversation so far, oldest first.
            json_schema: if given, the JSON Schema the answer should follow. A backend may pass
                it on to the server (structured output) or ignore it; the caller checks the
                answer either way.
        """
        ...  # pragma: no cover - a protocol has no body

"""Backends: where prompts are sent and where answers come from.

A backend takes a list of chat messages and returns a :class:`Response`, an immutable record of
the answer and of how it was obtained. The rest of reliacode depends only on the :class:`Backend`
interface, so any object with a matching ``complete`` method can be used as a backend.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

#: A chat message: ``{"role": "system" | "user" | "assistant", "content": "..."}``.
Message = Mapping[str, str]

_ROLES = ("system", "user", "assistant")
_EXAMPLE_MESSAGE = '{"role": "user", "content": "..."}'


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


def check_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Check a list of chat messages and return a plain copy of it.

    Raises:
        TypeError, ValueError: a message is not ``{"role": ..., "content": ...}`` with a known
            role and text content. The message says which one and how to fix it.
    """
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError(
            f"messages should be a list of chat messages, each written as {_EXAMPLE_MESSAGE}."
        )
    if not messages:
        raise ValueError(
            f"messages is empty. Give at least one message, such as {_EXAMPLE_MESSAGE}."
        )
    checked = []
    for number, message in enumerate(messages, 1):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(
                f'Message {number} should have exactly two fields, "role" and "content", as in '
                f"{_EXAMPLE_MESSAGE}."
            )
        role, content = message["role"], message["content"]
        if role not in _ROLES:
            raise ValueError(
                f'Message {number} has the role "{role}". Use "system", "user" or "assistant".'
            )
        if not isinstance(content, str):
            raise TypeError(f'The "content" of message {number} should be text.')
        checked.append({"role": role, "content": content})
    return checked


#: What a FakeBackend can be scripted with: answers in order, or a function of the messages.
Script = Sequence[str | BaseException] | Callable[[list[dict[str, str]]], str]


class FakeBackend:
    """A backend with scripted answers, for tests. It never opens a network connection.

    Use it to test code that calls a backend, without a server and without cost::

        backend = FakeBackend(['{"code": "positive"}', '{"code": "negative"}'])

    Args:
        answers: either a list, used in order (a text is returned as the answer; an exception,
            such as a :class:`BackendError`, is raised instead), or a function that takes the
            messages and returns the answer text. Both give the same answers on every run.
        model: the model name the fake reports.

    Every call is kept in ``calls`` (the messages) and ``json_schemas`` (the schema passed, or
    ``None``), so tests can check what was asked. Token counts are not known, so they stay
    ``None`` in every :class:`Response`.
    """

    def __init__(self, answers: Script, *, model: str = "fake-model") -> None:
        if isinstance(answers, (str, bytes)):
            raise TypeError(
                "answers should be a list of answers, such as [answer1, answer2], or a function. "
                "For a single answer, write [answer]."
            )
        self._answers = answers if callable(answers) else list(answers)
        self.model = model
        self.calls: list[list[dict[str, str]]] = []
        self.json_schemas: list[Mapping[str, Any] | None] = []

    def complete(
        self, messages: Sequence[Message], *, json_schema: Mapping[str, Any] | None = None
    ) -> Response:
        """Return the next scripted answer (or raise the next scripted exception)."""
        checked = check_messages(messages)
        self.calls.append(checked)
        self.json_schemas.append(json_schema)
        text = self._next_answer(checked)
        return Response(
            text=text,
            model=self.model,
            parameters=RequestParameters(
                endpoint="fake", model=self.model, structured_output=json_schema is not None
            ),
            finish_reason="stop",
            latency_seconds=0.0,
            created_at=utc_now(),
        )

    def _next_answer(self, messages: list[dict[str, str]]) -> str:
        if callable(self._answers):
            answer = self._answers(messages)
        else:
            call = len(self.calls)
            if call > len(self._answers):
                count = len(self._answers)
                raise RuntimeError(
                    f"FakeBackend has no answer left: it was given {count} "
                    f"{'answer' if count == 1 else 'answers'} and this is call {call}. Add more "
                    "answers, or pass a function that makes an answer from the messages."
                )
            answer = self._answers[call - 1]
            if isinstance(answer, BaseException):
                raise answer
        if not isinstance(answer, str):
            raise TypeError(f"FakeBackend answers should be text, but one is {answer!r}.")
        return answer

    def __repr__(self) -> str:
        return f"FakeBackend(model={self.model!r}, calls={len(self.calls)})"

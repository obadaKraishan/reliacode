"""Backends: where prompts are sent and where answers come from.

A backend takes a list of chat messages and returns a :class:`Response`, an immutable record of
the answer and of how it was obtained. The rest of reliacode depends only on the :class:`Backend`
interface, so any object with a matching ``complete`` method can be used as a backend.
"""

import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

logger = logging.getLogger(__name__)

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


class BackendError(RuntimeError):
    """A backend could not give an answer. The message says what went wrong, where, and what to try.

    Attributes:
        reason: a short label for the kind of problem, such as ``"no_server"`` or
            ``"rate_limited"``, for code that handles some problems differently.
        endpoint: the address the request went to.
        status: the HTTP status code of the server's last answer, or ``None`` if there was none.
    """

    def __init__(
        self, message: str, *, reason: str, endpoint: str | None = None, status: int | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.endpoint = endpoint
        self.status = status


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


#: How much of an unexpected answer an error message quotes.
_SNIPPET = 200
_CHAT_PATH = "/chat/completions"
#: Names of key variables are written in capitals; a key itself almost never is.
_KEY_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
#: A key is printable ASCII with no spaces, so it can go in an HTTP header.
_KEY_CHARACTERS = re.compile(r"[\x21-\x7e]+")
_HIDDEN = "[key hidden]"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
#: Words with which servers say that a model does not exist.
_MODEL_WORD = re.compile(r"\bmodel", re.IGNORECASE)
_MISSING_WORDS = re.compile(
    r"not found|does not exist|not exist|not a valid|invalid|unknown|no such|not available",
    re.IGNORECASE,
)
_SCHEMA_WORDS = re.compile(r"response_format|json_schema|structured|schema", re.IGNORECASE)
#: Waits between tries: 1, 2, 4, ... seconds, never more than 30.
_FIRST_WAIT = 1.0
_LONGEST_WAIT = 30.0
#: A server may ask for a longer wait (Retry-After), but past a minute reliacode stops instead.
_LONGEST_RETRY_AFTER = 60.0


class OpenAICompatibleBackend:
    """A backend for any server that offers the OpenAI-compatible chat completions interface.

    That covers a local Ollama or vLLM server, LM Studio, llama.cpp's server, OpenRouter and other
    hosted services. Requests go to ``{base_url}/chat/completions``; no vendor library is used.

    Args:
        base_url: the server's address, up to and including the version part, such as
            ``"http://localhost:11434/v1"`` for a local Ollama server.
        model: the model's name, as the server knows it, such as ``"qwen3:4b"``.
        api_key_env: the *name* of the environment variable that holds the API key, such as
            ``"OPENROUTER_API_KEY"``, or ``None`` for a local server that needs no key. The key
            is read from that variable each time a request is sent, goes only into the request's
            Authorization header, and is never stored, logged or shown in an error.
        timeout: how many seconds to wait for each answer.
        max_retries: how many times to try again after "too many requests" (status 429), a
            server error (status 500 and up), a timeout or a broken connection. Waits grow from
            1 second to 2, 4 and so on, up to 30 seconds, unless the server says how long to wait
            (Retry-After). Other refusals are not retried, because trying again would not help.
        temperature: the sampling temperature. The default, 0, makes answers as repeatable as
            the server allows.
        seed: a random seed, for servers that support one.
        max_tokens: a limit on the length of each answer, in tokens. The server's own default
            applies when it is ``None``.
        structured_output: when a JSON Schema is given for the answer, send it to the server
            (as ``response_format``), so the server holds the model to it. Set it to ``False``
            for a server that refuses ``response_format``; the answer is still checked.
        transport: an httpx transport. Tests pass ``httpx.MockTransport`` here, so that no
            request leaves the test.
        sleep: the function used to wait between tries. Tests replace it.
        clock: a function that returns seconds, used to measure the latency. Tests replace it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
        structured_output: bool = True,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.base_url = _check_base_url(base_url)
        self.model = _check_model(model)
        self.api_key_env = _check_key_name(api_key_env)
        self.timeout = _check_number("timeout", timeout, minimum=0, inclusive=False)
        self.temperature = _check_number("temperature", temperature, minimum=0)
        self.seed = _check_whole("seed", seed)
        self.max_tokens = _check_whole("max_tokens", max_tokens, minimum=1)
        self.max_retries = _check_whole("max_retries", max_retries, minimum=0)
        if not isinstance(structured_output, bool):
            raise TypeError("structured_output should be True or False.")
        self.structured_output = structured_output
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(transport=transport, timeout=self.timeout)

    @property
    def endpoint(self) -> str:
        """The full address that requests are sent to."""
        return self.base_url + _CHAT_PATH

    def complete(
        self, messages: Sequence[Message], *, json_schema: Mapping[str, Any] | None = None
    ) -> Response:
        """Send ``messages`` to the server and return its answer as a :class:`Response`.

        Raises:
            BackendError: the server could not be reached, refused the request, or answered in
                a shape that is not a chat completion.
        """
        checked = check_messages(messages)
        key = self._read_key()  # at call time, so a changed variable takes effect
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        structured = self.structured_output and json_schema is not None
        body: dict[str, Any] = {
            "model": self.model,
            "messages": checked,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": dict(json_schema)},
            }
        parameters = RequestParameters(
            endpoint=self.endpoint,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            structured_output=structured,
        )
        answer, latency, attempts = self._send(body, headers, key)
        if answer.status_code >= 300:
            raise self._refused(answer, key)
        return self._parse(answer, parameters, latency, attempts=attempts, key=key)

    def _send(
        self, body: dict[str, Any], headers: dict[str, str], key: str | None
    ) -> tuple[httpx.Response, float, int]:
        """Post the request, trying again after a passing problem. Returns the answer, the
        latency of the try that got it, and the number of tries."""
        tries = (self.max_retries or 0) + 1
        first_started = self._clock()
        for attempt in range(1, tries + 1):
            logger.debug(
                "POST %s (model %s, try %d of %d)", self.endpoint, self.model, attempt, tries
            )
            started = self._clock()
            answer: httpx.Response | None = None
            failure: httpx.TransportError | None = None
            try:
                answer = self._client.post(self.endpoint, json=body, headers=headers)
            except httpx.ConnectError as exc:
                raise self._no_server(exc, key) from None
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                failure = exc
            if answer is not None and not _worth_retrying(answer.status_code):
                return answer, max(self._clock() - started, 0.0), attempt
            if attempt == tries:
                elapsed = self._clock() - first_started
                raise self._gave_up(answer, failure, tries, elapsed, key)
            wait = self._wait_before(attempt + 1, answer, key)
            logger.warning(
                "The server at %s %s; trying again in %.1f seconds (try %d of %d).",
                self.endpoint,
                _what_happened(answer, failure),
                wait,
                attempt + 1,
                tries,
            )
            self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises

    def _no_server(self, failure: httpx.ConnectError, key: str | None) -> BackendError:
        detail = _scrub(str(failure), key)
        detail = f" ({detail})" if detail else ""
        if urlsplit(self.base_url).hostname in _LOCAL_HOSTS:
            advice = (
                "Check that the server is running and listens on this port. For Ollama, open "
                "the Ollama app or run ollama serve in a terminal, then try again."
            )
        else:
            advice = "Check the address in base_url and your internet connection, then try again."
        return BackendError(
            f"There is no server answering at {self.base_url}{detail}. {advice}",
            reason="no_server",
            endpoint=self.endpoint,
        )

    def _refused(self, answer: httpx.Response, key: str | None) -> BackendError:
        """The error for an answer that is a redirect or a refusal (status 300 to 499)."""
        status = answer.status_code
        message = _server_message(answer, key)
        said = _said(answer, key)
        if status < 400:
            location = (
                _scrub(answer.headers.get("location", ""), key) or "an address it did not give"
            )
            return BackendError(
                f"The server at {self.endpoint} sent the request on to {location} (status "
                f"{status}). Use that address as base_url: reliacode does not follow redirects, so "
                "a key is never sent to an address you did not choose.",
                reason="redirected",
                endpoint=self.endpoint,
                status=status,
            )
        if status in (401, 403):
            if self.api_key_env:
                advice = (
                    f"Check that {self.api_key_env} holds a current key for this service, and "
                    f'that the key may use the model "{self.model}".'
                )
            else:
                advice = (
                    "This backend sends no key. Put the key in an environment variable and pass "
                    'the variable\'s name, for example api_key_env="OPENROUTER_API_KEY".'
                )
            return BackendError(
                f"The server at {self.endpoint} refused the request (status {status}), which "
                f"usually means a missing or wrong API key. {advice} {said}",
                reason="key_refused",
                endpoint=self.endpoint,
                status=status,
            )
        if (
            status in (400, 404, 422)
            and _MODEL_WORD.search(message)
            and _MISSING_WORDS.search(message)
        ):
            return BackendError(
                f'The server at {self.base_url} has no model called "{self.model}" (status '
                f"{status}). {said} Check the name: for Ollama, ollama list shows the models you "
                f"have and ollama pull {self.model} downloads one; a hosted service lists its "
                "model names on its website.",
                reason="model_not_found",
                endpoint=self.endpoint,
                status=status,
            )
        if status == 404:
            return BackendError(
                f"The server at {self.base_url} has no chat completions endpoint at "
                f"{self.endpoint} (status 404). Check base_url: it usually ends in /v1, such as "
                "http://localhost:11434/v1 for Ollama or https://openrouter.ai/api/v1 for "
                "OpenRouter.",
                reason="no_endpoint",
                endpoint=self.endpoint,
                status=status,
            )
        hint = ""
        if _SCHEMA_WORDS.search(message):
            hint = (
                " This server may not support structured output: create the backend with "
                "structured_output=False, and the expected answer is described in the prompt only."
            )
        return BackendError(
            f"The server at {self.endpoint} refused the request (status {status}). {said}{hint}",
            reason="bad_request",
            endpoint=self.endpoint,
            status=status,
        )

    def _wait_before(self, attempt: int, answer: httpx.Response | None, key: str | None) -> float:
        asked = _retry_after(answer)
        if asked is None:
            return min(_FIRST_WAIT * 2 ** (attempt - 2), _LONGEST_WAIT)
        if asked > _LONGEST_RETRY_AFTER:
            assert answer is not None  # only an answer can ask for a wait
            raise BackendError(
                f"The server at {self.endpoint} asked to wait {asked:g} seconds before the next "
                f"request (status {answer.status_code}). That is longer than reliacode waits on "
                "its own, so it stopped. Wait, then run again; if this keeps happening, the "
                f"service's rate limit or daily quota may be used up. {_said(answer, key)}",
                reason="rate_limited",
                endpoint=self.endpoint,
                status=answer.status_code,
            )
        return asked

    def _gave_up(
        self,
        answer: httpx.Response | None,
        failure: httpx.TransportError | None,
        tries: int,
        elapsed: float,
        key: str | None,
    ) -> BackendError:
        times = "once" if tries == 1 else f"{tries} times in a row"
        span = f" over {elapsed:.0f} seconds" if tries > 1 else ""
        if answer is not None and answer.status_code == 429:
            return BackendError(
                f'The server at {self.endpoint} answered "too many requests" (status 429) '
                f"{times}{span}. Wait a while and run again, send fewer requests at a time, or "
                "raise max_retries. On a free plan, the day's limit may be used up. "
                + _said(answer, key),
                reason="rate_limited",
                endpoint=self.endpoint,
                status=429,
            )
        if answer is not None:
            return BackendError(
                f"The server at {self.endpoint} had a problem of its own (status "
                f"{answer.status_code}) {times}{span}. This is usually temporary: try again "
                "later. " + _said(answer, key),
                reason="server_error",
                endpoint=self.endpoint,
                status=answer.status_code,
            )
        if isinstance(failure, httpx.TimeoutException):
            return BackendError(
                f"The server at {self.endpoint} did not answer within {self.timeout:g} seconds, "
                f"{times}. A local model can take a long time to load on its first request, and "
                "a long answer takes long to write: raise timeout, lower max_tokens, or try a "
                "smaller model.",
                reason="timeout",
                endpoint=self.endpoint,
            )
        return BackendError(
            f"The connection to the server at {self.endpoint} broke off before an answer came, "
            f"{times} ({_scrub(str(failure), key) or type(failure).__name__}). Check that the "
            "server is still running and the network is up, then try again.",
            reason="connection_lost",
            endpoint=self.endpoint,
        )

    def _read_key(self) -> str | None:
        name = self.api_key_env
        if name is None:
            return None
        value = os.environ.get(name)
        if value is None or not value.strip():
            state = "is not set" if value is None else "is set but empty"
            raise BackendError(
                f"The environment variable {name} {state}, so there is no API key to send to "
                f"{self.endpoint}. Set it before you start Python, for example with "
                f"export {name}=... in a terminal, or load it from a .env file that you never "
                "share. Do not write the key into your code.",
                reason="missing_key",
                endpoint=self.endpoint,
            )
        key = value.strip()
        if not _KEY_CHARACTERS.fullmatch(key):
            raise BackendError(
                f"The environment variable {name} holds spaces, line breaks or other characters "
                "that an API key cannot have, so it cannot be sent. Check its value (this "
                "message does not show it).",
                reason="bad_key",
                endpoint=self.endpoint,
            )
        return key

    def _parse(
        self,
        answer: httpx.Response,
        parameters: RequestParameters,
        latency: float,
        attempts: int,
        key: str | None,
    ) -> Response:
        try:
            data = answer.json()
        except ValueError:
            raise self._unexpected("the answer is not JSON", answer, key) from None
        if not isinstance(data, dict):
            raise self._unexpected("the answer is not a JSON object", answer, key)
        if "choices" not in data and "error" in data:
            raise BackendError(
                f"The server at {self.endpoint} answered with an error instead of a chat "
                f"completion. {_said(answer, key)} Try again later; if it keeps happening, check "
                "the model name and the service's status page.",
                reason="server_error",
                endpoint=self.endpoint,
                status=answer.status_code,
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._unexpected('it has no "choices" list', answer, key)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._unexpected('its first choice has no "message"', answer, key)
        text = _content_text(message.get("content"))
        if text is None:
            raise self._unexpected('the message "content" is not text', answer, key)
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return Response(
            text=text,
            model=_text_or_none(data.get("model")),
            parameters=parameters,
            prompt_tokens=_count(usage.get("prompt_tokens")),
            completion_tokens=_count(usage.get("completion_tokens")),
            total_tokens=_count(usage.get("total_tokens")),
            finish_reason=_text_or_none(choice.get("finish_reason")),
            response_id=_text_or_none(data.get("id")),
            request_id=_text_or_none(answer.headers.get("x-request-id")),
            system_fingerprint=_text_or_none(data.get("system_fingerprint")),
            latency_seconds=latency,
            attempts=attempts,
            created_at=utc_now(),
        )

    def _unexpected(self, what: str, answer: httpx.Response, key: str | None) -> BackendError:
        snippet = " ".join(_scrub(answer.text, key)[:_SNIPPET].split())
        return BackendError(
            f"The server at {self.endpoint} answered, but not in the shape of a chat completion: "
            f"{what}. Check that base_url points to an OpenAI-compatible server; its address "
            f'usually ends in /v1. The answer began: "{snippet}"',
            reason="unexpected_answer",
            endpoint=self.endpoint,
            status=answer.status_code,
        )

    def close(self) -> None:
        """Close the backend's connections. A closed backend cannot send more requests."""
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleBackend(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key_env={self.api_key_env!r})"
        )


def _check_base_url(base_url: object) -> str:
    example = 'such as "http://localhost:11434/v1" for a local Ollama server'
    if not isinstance(base_url, str) or not base_url.strip():
        raise TypeError(f"base_url should be the server's address, {example}.")
    url = base_url.strip().rstrip("/")
    url = url.removesuffix(_CHAT_PATH)
    try:
        parts = urlsplit(url)
        has_login = parts.username is not None or parts.password is not None
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError for a port that is not a number
    except ValueError:
        raise ValueError(f"base_url is not a web address that can be read, {example}.") from None
    # The address is quoted in error messages and kept in every result, so it may not hold a key.
    if has_login or parts.query or parts.fragment:
        raise ValueError(
            "base_url holds a user name, a password or a query string. A key does not belong in "
            "the address: keep it in an environment variable and give the variable's name as "
            "api_key_env."
        )
    if parts.scheme not in ("http", "https") or not host:
        raise ValueError(f"base_url should start with http:// or https:// and a server, {example}.")
    return url


def _check_key_name(name: object) -> str | None:
    if name is None:
        return None
    # The value is never quoted back: if it is a key rather than a name, it must not be shown.
    if not isinstance(name, str) or not _KEY_NAME.fullmatch(name):
        raise ValueError(
            "api_key_env should be the name of the environment variable that holds the key, "
            "written in capitals, such as OPENROUTER_API_KEY, and not the key itself. Put the "
            "key in that variable, then pass the variable's name."
        )
    return name


def _worth_retrying(status: int) -> bool:
    """Too many requests, or a server error: a later try may succeed."""
    return status == 429 or status >= 500


def _retry_after(answer: httpx.Response | None) -> float | None:
    """The wait in seconds that the server asked for, if it gave one as a number."""
    if answer is None:
        return None
    try:
        seconds = float(answer.headers.get("retry-after", ""))
    except ValueError:
        return None  # missing, or given as a date, which is rare: the usual waits apply
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _what_happened(answer: httpx.Response | None, failure: Exception | None) -> str:
    if answer is not None:
        return f"answered with status {answer.status_code}"
    if isinstance(failure, httpx.TimeoutException):
        return "did not answer in time"
    return "dropped the connection"


def _server_message(answer: httpx.Response, key: str | None) -> str:
    """The server's own explanation, short and with the key removed, or ""."""
    try:
        data = answer.json()
    except ValueError:
        data = None
    message: object = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        message = message or data.get("message") or data.get("detail")
    if not isinstance(message, str):
        message = answer.text if not isinstance(data, (dict, list)) else ""
    message = " ".join(_scrub(message, key).split())
    return message[:_SNIPPET] + ("..." if len(message) > _SNIPPET else "")


def _said(answer: httpx.Response, key: str | None) -> str:
    message = _server_message(answer, key)
    return f'The server said: "{message}"' if message else "The server gave no reason."


def _scrub(text: str, key: str | None) -> str:
    """``text`` with every copy of the key replaced, for text that a server sent back."""
    return text.replace(key, _HIDDEN) if key else text


def _check_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise TypeError('model should be the name of a model on the server, such as "qwen3:4b".')
    return model.strip()


def _check_number(name: str, value: object, *, minimum: float, inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} should be a number, but it is {value!r}.")
    if value < minimum or (not inclusive and value == minimum):
        relation = "at least" if inclusive else "more than"
        raise ValueError(f"{name} should be {relation} {minimum}, but it is {value}.")
    return float(value)


def _check_whole(name: str, value: object, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} should be a whole number or None, but it is {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} should be at least {minimum}, but it is {value}.")
    return value


def _content_text(content: object) -> str | None:
    """The text of a message's content: a string, nothing (no text), or a list of text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(part, dict) for part in content):
        texts = [part.get("text") for part in content if part.get("type") == "text"]
        if all(isinstance(text, str) for text in texts):
            return "".join(texts)
    return None


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: object) -> int | None:
    """A token count as the server gave it, or None. A missing count never becomes zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value

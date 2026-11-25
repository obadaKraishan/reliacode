"""A backend for Anthropic's own Messages API.

:class:`AnthropicBackend` sends each request to ``POST /v1/messages`` over httpx, with no vendor
library. It follows the same :class:`~reliacode.Backend` interface as the other backends, so
:func:`reliacode.code` works with it unchanged.

The chat messages reliacode builds are turned into the Messages API's form: the system messages
at the start become the request's ``system`` field, and the rest stay as messages. Only the text
of an answer is kept.
"""

import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from reliacode.backends import (
    _FIRST_WAIT,
    _KEY_CHARACTERS,
    _LOCAL_HOSTS,
    _LONGEST_RETRY_AFTER,
    _LONGEST_WAIT,
    _MISSING_WORDS,
    _MODEL_WORD,
    _REFUSED_WORDS,
    _SCHEMA_WORDS,
    _SNIPPET,
    _TEMPERATURE_WORD,
    BackendError,
    Message,
    RequestParameters,
    Response,
    _check_key_name,
    _check_model,
    _check_number,
    _check_whole,
    _count,
    _refuse_credential_fields,
    _retry_after,
    _said,
    _scrub,
    _server_message,
    _text_or_none,
    _what_happened,
    check_messages,
    utc_now,
)

logger = logging.getLogger(__name__)

#: The ``anthropic-version`` header every request sends: the latest version Anthropic's API
#: versions page listed when this backend was written. It is recorded in every response.
API_VERSION = "2026-10-21"
_API_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MESSAGES_PATH = "/v1/messages"
_DEFAULT_BASE_URL = "https://api.anthropic.com"
#: Statuses worth another try, as Anthropic's errors page lists them for its own clients.
_RETRY_STATUSES = {408, 409, 429}
#: The cache's times to live that the Messages API offers (``cache_control.ttl``).
_CACHE_TTLS = ("5m", "1h")


class AnthropicBackend:
    """A backend for Anthropic's own Messages API (``POST /v1/messages``).

    Use it to code with a Claude model directly, rather than through Anthropic's
    OpenAI-compatible endpoint::

        backend = AnthropicBackend("claude-haiku-5-5", max_tokens=4000)
        run = code(codebook, texts, backend=backend)

    Args:
        model: the model's name, such as ``"claude-haiku-5-5"``.
        api_key_env: the *name* of the environment variable that holds the API key. The key is
            read from it each time a request is sent, goes only into the request's
            ``Authorization`` header, and is never stored, logged or shown in an error.
        max_tokens: the most tokens an answer may use. The Messages API needs a limit. On
            models that think before they answer, the thinking counts toward it, so leave room.
        cache: ``"5m"`` or ``"1h"`` to cache the system prompt (the codebook part that every
            call shares) for five minutes or one hour, or ``None`` (the default) for no caching.
            The cache marker goes on the system prompt only, never on the text being coded, so
            caching changes no prompt text and no prompt hash. A prompt shorter than the
            model's minimum is not cached, and the API says nothing; reliacode logs a warning
            when an answer shows that nothing was read from or written to the cache.
        temperature: the sampling temperature. The default, ``None``, sends none, and the record
            says none was sent. API version 2026-10-21 refuses a temperature on every model, so
            give one only with an older ``api_version`` and a model that takes it.
        extra_body: extra request fields, sent as given in every request and recorded in every
            response's ``parameters.extra_body``. Fields the backend sets itself are refused,
            and so is any field whose name looks like a credential.
        base_url: the API's address, ``"https://api.anthropic.com"`` by default.
        api_version: the ``anthropic-version`` header, pinned to :data:`API_VERSION` by default
            and recorded in every response.
        timeout: how many seconds to wait for each answer.
        max_retries: how many times to try again after "overloaded" (status 529), "too many
            requests" (429), another server error, a timeout or a broken connection. Waits grow
            from 1 second to 2, 4 and so on, up to 30 seconds, unless the API says how long to
            wait (``retry-after``). When the API says not to try again (``x-should-retry:
            false``), it is not tried again.
        transport: an httpx transport. Tests pass ``httpx.MockTransport`` here, so that no
            request leaves the test.
        sleep: the function used to wait between tries. Tests replace it.
        clock: a function that returns seconds, used to measure the latency. Tests replace it.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
        cache: Literal["5m", "1h"] | None = None,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        api_version: str = API_VERSION,
        timeout: float = 600.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        # First, so that a field that looks like a credential is refused before anything else.
        self._extra_body = _check_extra_body(extra_body)
        self.model = _check_model(model)
        if api_key_env is None:
            raise ValueError(
                "AnthropicBackend needs an API key: give the name of the environment variable "
                'that holds it, such as api_key_env="ANTHROPIC_API_KEY".'
            )
        self.api_key_env = _check_key_name(api_key_env)
        max_tokens = _check_whole("max_tokens", max_tokens, minimum=1)
        if max_tokens is None:
            raise TypeError("max_tokens should be a whole number: the Messages API needs a limit.")
        self.max_tokens = max_tokens
        if cache is not None and cache not in _CACHE_TTLS:
            shown = repr(cache) if isinstance(cache, str) else type(cache).__name__
            raise ValueError(
                'cache should be "5m" (five minutes), "1h" (one hour) or None (no caching), '
                f"the times to live the Messages API offers, but it is {shown}."
            )
        self.cache = cache
        self._warned_uncached = False
        self.temperature = (
            None if temperature is None else _check_number("temperature", temperature, minimum=0)
        )
        self.base_url = _check_base_url(base_url)
        if not isinstance(api_version, str) or not _API_DATE.fullmatch(api_version):
            raise ValueError(
                f'api_version should be a date written as "YYYY-MM-DD", such as "{API_VERSION}".'
            )
        self.api_version = api_version
        self.timeout = _check_number("timeout", timeout, minimum=0, inclusive=False)
        self.max_retries = _check_whole("max_retries", max_retries, minimum=0)
        self._sleep = sleep
        self._clock = clock
        self._client = httpx.Client(transport=transport, timeout=self.timeout)

    @property
    def endpoint(self) -> str:
        """The full address that requests are sent to."""
        return self.base_url + _MESSAGES_PATH

    @property
    def extra_body(self) -> dict[str, Any]:
        """A copy of the extra request fields sent with every request."""
        return json.loads(self._extra_body)

    def complete(
        self, messages: Sequence[Message], *, json_schema: Mapping[str, Any] | None = None
    ) -> Response:
        """Send ``messages`` to the Messages API and return its answer as a :class:`Response`.

        Raises:
            BackendError: the API could not be reached, refused the request, or answered in a
                shape that is not a message.
            ValueError: a system message comes after another message: this backend sends system
                messages at the start only.
        """
        body, parameters = self._request(messages, json_schema)
        key = self._read_key()  # at call time, so a changed variable takes effect
        answer, latency, attempts = self._send("POST", self.endpoint, key, body=body)
        if answer.status_code >= 300:
            raise self._failed(answer, attempts, key)
        data = self._json(answer, key)
        return self._response(
            data,
            parameters,
            latency=latency,
            attempts=attempts,
            request_id=_text_or_none(answer.headers.get("request-id")),
            key=key,
        )

    # The request ------------------------------------------------------------------------------

    def _request(
        self, messages: Sequence[Message], json_schema: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], RequestParameters]:
        """The request body for ``messages``, and the record of what it asks."""
        checked = check_messages(messages)
        system, conversation = _split_system(checked)
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        cached = None
        if system:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": text} for text in system]
            if self.cache is not None:
                # On the shared part only: the last system block ends the cached prefix.
                blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": self.cache}
                cached = self.cache
            body["system"] = blocks
        body["messages"] = conversation
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)  # never one of the fields above: those are refused
        parameters = RequestParameters(
            endpoint=self.endpoint,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
            api_version=self.api_version,
            cache=cached,
        )
        return body, parameters

    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "anthropic-version": self.api_version}

    # Sending ----------------------------------------------------------------------------------

    def _send(
        self,
        method: str,
        url: str,
        key: str,
        *,
        body: dict[str, Any] | None = None,
        retry_lost: bool = True,
    ) -> tuple[httpx.Response, float, int]:
        """Send one request, trying again after a passing problem. Returns the answer, the
        latency of the try that got it, and the number of tries.

        With ``retry_lost=False``, a timeout or a broken connection is not tried again, because
        the API may have acted on the request before the answer was lost.
        """
        tries = (self.max_retries or 0) + 1
        first_started = self._clock()
        headers = self._headers(key)
        for attempt in range(1, tries + 1):
            logger.debug("%s %s (model %s, try %d of %d)", method, url, self.model, attempt, tries)
            started = self._clock()
            answer: httpx.Response | None = None
            failure: httpx.TransportError | None = None
            try:
                answer = self._client.request(method, url, json=body, headers=headers)
            except httpx.ConnectError as exc:
                raise self._no_server(exc, key) from None
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if not retry_lost:
                    raise self._gave_up(None, exc, 1, 0.0, key) from None
                failure = exc
            if answer is not None and not _worth_retrying(answer):
                return answer, max(self._clock() - started, 0.0), attempt
            if attempt == tries:
                elapsed = self._clock() - first_started
                raise self._gave_up(answer, failure, tries, elapsed, key)
            wait = self._wait_before(attempt + 1, answer, key)
            logger.warning(
                "The API at %s %s; trying again in %.1f seconds (try %d of %d).",
                url,
                _what_happened(answer, failure),
                wait,
                attempt + 1,
                tries,
            )
            self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises

    def _read_key(self) -> str:
        name = self.api_key_env
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

    def _wait_before(self, attempt: int, answer: httpx.Response | None, key: str) -> float:
        asked = _retry_after(answer)
        if asked is None:
            return min(_FIRST_WAIT * 2 ** (attempt - 2), _LONGEST_WAIT)
        if asked > _LONGEST_RETRY_AFTER:
            assert answer is not None  # only an answer can ask for a wait
            raise BackendError(
                f"The API at {self.endpoint} asked to wait {asked:g} seconds before the next "
                f"request (status {answer.status_code}). That is longer than reliacode waits on "
                "its own, so it stopped. Wait, then run again; if this keeps happening, a rate "
                f"limit or a spend limit of your account may be used up. {_said(answer, key)}",
                reason="rate_limited",
                endpoint=self.endpoint,
                status=answer.status_code,
            )
        return asked

    # Answers ----------------------------------------------------------------------------------

    def _json(self, answer: httpx.Response, key: str) -> dict[str, Any]:
        try:
            data = answer.json()
        except ValueError:
            raise self._unexpected("the answer is not JSON", answer.text, key) from None
        if not isinstance(data, dict):
            raise self._unexpected("the answer is not a JSON object", answer.text, key)
        return data

    def _response(
        self,
        data: dict[str, Any],
        parameters: RequestParameters,
        *,
        latency: float,
        attempts: int,
        request_id: str | None,
        key: str | None,
    ) -> Response:
        """The record of one message, as the Messages API returns it."""
        if data.get("type") == "error" or ("content" not in data and "error" in data):
            message = _error_message(data, key)
            said = f'The API said: "{message}"' if message else "The API gave no reason."
            raise BackendError(
                f"The API at {self.endpoint} answered with an error instead of a message. "
                f"{said} Try again later; if it keeps happening, check the model name and "
                "Anthropic's status page.",
                reason="server_error",
                endpoint=self.endpoint,
            )
        content = data.get("content")
        if not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
            raise self._unexpected('it has no "content" list', json.dumps(data), key)
        # Only text blocks are read: a thinking block's text is never kept.
        texts = [block.get("text") for block in content if block.get("type") == "text"]
        if not all(isinstance(text, str) for text in texts):
            raise self._unexpected("a text block holds no text", json.dumps(data), key)
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("output_tokens_details")
        details = details if isinstance(details, dict) else {}
        uncached = _count(usage.get("input_tokens"))
        written = _count(usage.get("cache_creation_input_tokens"))
        read = _count(usage.get("cache_read_input_tokens"))
        by_ttl = usage.get("cache_creation")
        written_1h = (
            _count(by_ttl.get("ephemeral_1h_input_tokens")) if isinstance(by_ttl, dict) else None
        )
        if parameters.cache is not None and written == 0 and read == 0:
            self._warn_uncached()
        output = _count(usage.get("output_tokens"))
        # The whole input is the three counts together, as the API documents it.
        prompt = None if uncached is None else uncached + (written or 0) + (read or 0)
        return Response(
            text="".join(texts),  # type: ignore[arg-type] - every one is a str, checked above
            model=_text_or_none(data.get("model")),
            parameters=parameters,
            prompt_tokens=prompt,
            completion_tokens=output,
            total_tokens=prompt + output if prompt is not None and output is not None else None,
            finish_reason=_text_or_none(data.get("stop_reason")),
            response_id=_text_or_none(data.get("id")),
            request_id=request_id,
            latency_seconds=latency,
            attempts=attempts,
            created_at=utc_now(),
            reasoning_tokens=_count(details.get("thinking_tokens")),
            cache_write_tokens=written,
            cache_write_1h_tokens=written_1h,
            cache_read_tokens=read,
        )

    def _warn_uncached(self) -> None:
        if self._warned_uncached:
            return
        self._warned_uncached = True
        logger.warning(
            "Nothing was read from or written to the prompt cache for model %s, although the "
            "system prompt was marked for caching. A prompt shorter than the model's minimum "
            "cacheable length is not cached, and the API gives no error: see Anthropic's prompt "
            "caching page for the minimum. Calls are priced as uncached input.",
            self.model,
        )

    # Errors -----------------------------------------------------------------------------------

    def _no_server(self, failure: httpx.ConnectError, key: str | None) -> BackendError:
        detail = _scrub(str(failure), key)
        detail = f" ({detail})" if detail else ""
        if urlsplit(self.base_url).hostname in _LOCAL_HOSTS:
            advice = "Check that the server is running and listens on this port, then try again."
        else:
            advice = "Check the address in base_url and your internet connection, then try again."
        return BackendError(
            f"There is no server answering at {self.base_url}{detail}. {advice}",
            reason="no_server",
            endpoint=self.endpoint,
        )

    def _failed(self, answer: httpx.Response, attempts: int, key: str | None) -> BackendError:
        """The error for an answer that is not a success, once no more tries are left."""
        if answer.status_code == 429 or answer.status_code >= 500:
            return self._gave_up(answer, None, attempts, 0.0, key)
        return self._refused(answer, key)

    def _refused(self, answer: httpx.Response, key: str | None) -> BackendError:
        """The error for an answer that is a redirect or a refusal (status 300 to 499)."""
        status = answer.status_code
        message = _server_message(answer, key)
        said = _said(answer, key).replace("The server said", "The API said")
        said = said.replace("The server gave no reason", "The API gave no reason")
        url = _url_of(answer, self.endpoint)
        if status < 400:
            location = (
                _scrub(answer.headers.get("location", ""), key) or "an address it did not give"
            )
            return BackendError(
                f"The API at {url} sent the request on to {location} (status {status}). Use "
                "that address as base_url: reliacode does not follow redirects, so a key is "
                "never sent to an address you did not choose.",
                reason="redirected",
                endpoint=self.endpoint,
                status=status,
            )
        if status in (401, 403):
            return BackendError(
                f"The API at {url} refused the request (status {status}), which usually means "
                f"a missing or wrong API key, or a key that may not use this model. Check that "
                f"{self.api_key_env} holds a current key for Anthropic's API, and that the key "
                f'may use the model "{self.model}". {said}',
                reason="key_refused",
                endpoint=self.endpoint,
                status=status,
            )
        if status == 402:
            return BackendError(
                f"The API at {url} refused the request because of a billing problem (status "
                f"402). Check the payment details of your account in the Claude Console. {said}",
                reason="billing",
                endpoint=self.endpoint,
                status=status,
            )
        about_temperature = bool(_TEMPERATURE_WORD.search(message))
        if (
            status in (400, 404)
            and _MODEL_WORD.search(message)
            and _MISSING_WORDS.search(message)
            and not about_temperature
        ):
            return BackendError(
                f'The API at {self.base_url} has no model called "{self.model}" (status '
                f"{status}). {said} Check the name against Anthropic's list of models, such as "
                '"claude-haiku-5-5".',
                reason="model_not_found",
                endpoint=self.endpoint,
                status=status,
            )
        if status == 404:
            return BackendError(
                f"The API at {self.base_url} has nothing at {url} (status 404). Check base_url: "
                f"it should be the API's address without a path, such as {_DEFAULT_BASE_URL}.",
                reason="no_endpoint",
                endpoint=self.endpoint,
                status=status,
            )
        hint = ""
        if status == 413:
            hint = " The request is larger than the API accepts."
        elif about_temperature and self.temperature is not None and _REFUSED_WORDS.search(message):
            hint = (
                " This model or API version may not take a temperature: create the backend "
                "with temperature=None (the default), and the field is left out of the request."
            )
        elif _SCHEMA_WORDS.search(message):
            hint = (
                " The model may not support structured output: create the backend with "
                "structured_output=False, and the expected answer is described in the prompt only."
            )
        return BackendError(
            f"The API at {url} refused the request (status {status}). {said}{hint}",
            reason="bad_request",
            endpoint=self.endpoint,
            status=status,
        )

    def _gave_up(
        self,
        answer: httpx.Response | None,
        failure: Exception | None,
        tries: int,
        elapsed: float,
        key: str | None,
    ) -> BackendError:
        times = "once" if tries == 1 else f"{tries} times in a row"
        span = f" over {elapsed:.0f} seconds" if tries > 1 else ""
        url = self.endpoint
        if answer is not None:
            url = _url_of(answer, url)
            said = _said(answer, key).replace("The server said", "The API said")
            said = said.replace("The server gave no reason", "The API gave no reason")
            if answer.status_code == 429:
                return BackendError(
                    f'The API at {url} answered "too many requests" (status 429) {times}{span}. '
                    "Wait a while and run again, send fewer requests at a time, or raise "
                    "max_retries. A 429 without a wait to honor can also mean that your "
                    f"account's monthly spend limit is reached. {said}",
                    reason="rate_limited",
                    endpoint=self.endpoint,
                    status=429,
                )
            what = (
                "was overloaded (status 529)"
                if answer.status_code == 529
                else f"had a problem of its own (status {answer.status_code})"
            )
            return BackendError(
                f"The API at {url} {what} {times}{span}. This is usually temporary: try again "
                f"later. {said}",
                reason="server_error",
                endpoint=self.endpoint,
                status=answer.status_code,
            )
        if isinstance(failure, httpx.TimeoutException):
            return BackendError(
                f"The API at {url} did not answer within {self.timeout:g} seconds, {times}. A "
                "long answer takes long to write: raise timeout or lower max_tokens.",
                reason="timeout",
                endpoint=self.endpoint,
            )
        return BackendError(
            f"The connection to the API at {url} broke off before an answer came, {times} "
            f"({_scrub(str(failure), key) or type(failure).__name__}). Check the network, then "
            "try again.",
            reason="connection_lost",
            endpoint=self.endpoint,
        )

    def _unexpected(self, what: str, text: str, key: str | None) -> BackendError:
        snippet = " ".join(_scrub(text, key)[:_SNIPPET].split())
        return BackendError(
            f"The API at {self.endpoint} answered, but not in the shape of a message: {what}. "
            f'Check that base_url points to Anthropic\'s API. The answer began: "{snippet}"',
            reason="unexpected_answer",
            endpoint=self.endpoint,
        )

    def close(self) -> None:
        """Close the backend's connections. A closed backend cannot send more requests."""
        self._client.close()

    def __enter__(self) -> "AnthropicBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model!r}, api_key_env={self.api_key_env!r})"


def _worth_retrying(answer: httpx.Response) -> bool:
    """Whether a later try may succeed: the API's own ``x-should-retry`` header decides when it
    is there; otherwise the statuses Anthropic's errors page lists (408, 409, 429, 500 and up)."""
    said = answer.headers.get("x-should-retry", "").strip().lower()
    if said == "true":
        return True
    if said == "false":
        return False
    return answer.status_code in _RETRY_STATUSES or answer.status_code >= 500


def _url_of(answer: httpx.Response, fallback: str) -> str:
    """The address a request went to, as the answer records it."""
    try:
        return str(answer.request.url)
    except RuntimeError:  # pragma: no cover - httpx always sets it on a client's answers
        return fallback


def _split_system(messages: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    """The texts of the system messages at the start, and the messages after them."""
    system: list[str] = []
    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        if messages[index]["content"]:
            system.append(messages[index]["content"])
        index += 1
    rest = messages[index:]
    for number, message in enumerate(rest, index + 1):
        if message["role"] == "system":
            raise ValueError(
                f"Message {number} is a system message after another message. AnthropicBackend "
                "sends system messages at the start of the conversation only: put them first."
            )
    return system, rest


def _error_message(data: dict[str, Any], key: str | None) -> str:
    error = data.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return ""
    message = " ".join(_scrub(message, key).split())
    return message[:_SNIPPET] + ("..." if len(message) > _SNIPPET else "")


def _check_base_url(base_url: object) -> str:
    example = f'such as "{_DEFAULT_BASE_URL}"'
    if not isinstance(base_url, str) or not base_url.strip():
        raise TypeError(f"base_url should be the API's address, {example}.")
    url = base_url.strip().rstrip("/")
    url = url.removesuffix(_MESSAGES_PATH).removesuffix("/v1")
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


#: The request fields the backend sets itself, and what to use instead.
_OWN_FIELDS = {
    "model": "Give the model as the backend's model argument.",
    "messages": "The messages are the ones given to complete().",
    "system": "The system prompt comes from the system messages given to complete().",
    "max_tokens": "Use the backend's max_tokens setting.",
    "stream": "The backend reads one whole answer, so it never streams.",
    "temperature": "Use the backend's temperature setting.",
    "thinking": "Use the backend's thinking setting.",
    "output_config": "Use the backend's effort and structured_output settings.",
    "cache_control": "Use the backend's cache setting.",
}
_EXTRA_EXAMPLE = '{"metadata": {"user_id": "study-a"}}'


def _check_extra_body(extra_body: object) -> str:
    """The extra request fields, checked, as JSON text (so every use gets a fresh copy)."""
    if extra_body is None:
        return "{}"
    if not isinstance(extra_body, Mapping):
        raise TypeError(
            f"extra_body should be a mapping of field names to values, such as {_EXTRA_EXAMPLE}."
        )
    _refuse_credential_fields(extra_body, ValueError)
    for name in extra_body:
        if name in _OWN_FIELDS:
            raise ValueError(
                f'extra_body names "{name}", a field the backend sets itself. {_OWN_FIELDS[name]}'
            )
    try:
        return json.dumps(dict(extra_body), allow_nan=False)
    except (TypeError, ValueError):
        raise TypeError(
            "extra_body should hold only JSON values: text, numbers, True or False, None, and "
            f"lists and mappings of them, such as {_EXTRA_EXAMPLE}."
        ) from None

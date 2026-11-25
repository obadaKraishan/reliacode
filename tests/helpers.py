"""Shared test helpers: paths, small YAML snippets, a mock HTTP server, a test clock, and
answers and ledger lines to build records from."""

import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from reliacode.backends import RequestParameters, Response

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples" / "codebooks"

#: Invented hashes for ledger lines that are not tied to a real codebook or prompt.
CODEBOOK_HASH = "sha256:" + "c" * 64
PROMPT_HASH = "sha256:" + "d" * 64
#: A fixed time for records, so tests do not depend on the clock.
NOON = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)


def response(
    text: str = '{"code": "positive"}',
    *,
    model: str = "example-model",
    prompt_tokens: int | None = 40,
    completion_tokens: int | None = 6,
    **fields: object,
) -> Response:
    """An answer as a backend returns it, with invented counts. ``fields`` replace the rest."""
    values: dict[str, object] = {
        "text": text,
        "model": model,
        "parameters": RequestParameters(
            endpoint="http://localhost:11434/v1/chat/completions",
            model=model,
            temperature=0.0,
            structured_output=True,
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": "stop",
        "latency_seconds": 0.25,
        "created_at": NOON,
    }
    values.update(fields)
    return Response.model_validate(values)


class TickingClock:
    """A UTC clock the tests control: every reading is ``step`` seconds after the last one."""

    def __init__(self, start: datetime = NOON, step: float = 1.0) -> None:
        self.now = start
        self.step = timedelta(seconds=step)

    def __call__(self) -> datetime:
        self.now += self.step
        return self.now


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to open a network connection (see conftest.py).

    It is not an OSError, so no client code can mistake it for a server that is down.
    """


def yaml_text(text: str) -> str:
    """Dedent a triple-quoted YAML snippet so tests can indent it naturally."""
    return textwrap.dedent(text).lstrip("\n")


MINIMAL = yaml_text(
    """
    name: tone-demo
    variables:
      - name: tone
        definition: The overall tone of the text toward its main subject.
        categories:
          - code: positive
            definition: The text mostly praises or supports its subject.
          - code: negative
            definition: The text mostly criticises or opposes its subject.
    """
)


def completion(text: str = '{"code": "positive"}', **fields: object) -> dict[str, object]:
    """A chat completion as an OpenAI-compatible server returns it. ``fields`` replace the top."""
    body: dict[str, object] = {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "created": 1795000000,
        "model": "tiny-model-served",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 6, "total_tokens": 48},
    }
    body.update(fields)
    return body


def anthropic_message(
    text: str = '{"code": "positive"}', *, usage: dict[str, object] | None = None, **fields: object
) -> dict[str, object]:
    """A message as Anthropic's Messages API returns it, with invented ids and counts. ``usage``
    replaces the counts, and ``fields`` replace the rest."""
    body: dict[str, object] = {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 7,
        },
    }
    body.update(fields)
    return body


class FakeServer:
    """A mock HTTP server for httpx: answers from a script and keeps every request it gets.

    Each answer is an ``httpx.Response``, an exception to raise, or a function of the request.
    Answers are used in order, and the last one repeats, so ``FakeServer(error_500)`` keeps
    failing. Pass ``server.transport`` to a backend; no socket is ever opened.
    """

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers) or [httpx.Response(200, json=completion())]
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(request)
        # A fresh copy each time, so a repeated answer is never a response that was already read.
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def body(self, number: int = -1) -> dict[str, object]:
        """The JSON body of a request the server got (the last one by default)."""
        return json.loads(self.requests[number].content)


class FakeClock:
    """A clock the tests control: every reading moves time on by ``step`` seconds."""

    def __init__(self, step: float = 0.5) -> None:
        self.now = 1000.0
        self.step = step
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


#: A codebook for reliability tests: a nominal and an ordinal variable.
AGREEMENT = yaml_text(
    """
    name: agreement-demo
    variables:
      - name: tone
        definition: The overall tone of the text toward its main subject.
        categories:
          - code: positive
            definition: The text mostly praises or supports its subject.
          - code: neutral
            definition: The text neither praises nor criticises its subject.
          - code: negative
            definition: The text mostly criticises or opposes its subject.
      - name: severity
        definition: How strong the incivility in the text is.
        type: ordinal
        categories:
          - code: none
            definition: No incivility.
            level: 0
          - code: mild
            definition: Incivility a reader could miss.
            level: 1
          - code: strong
            definition: Incivility no reader could miss.
            level: 2
    """
)


def matrix_rows(variable: str, matrix: dict[str, list[object]]) -> list[dict[str, object]]:
    """Rows for ``read_codes`` from a reliability data matrix: for each coder, the values of
    units 1, 2, 3, ... in order, with ``None`` for a missing value."""
    return [
        {"unit": str(number), "coder": coder, "variable": variable, "value": value}
        for coder, values in matrix.items()
        for number, value in enumerate(values, 1)
    ]


def one_variable(codes: list[str], *, ordinal: bool = False, name: str = "v") -> str:
    """A codebook with one variable whose codes are ``codes``. An ordinal variable gets the
    levels 0, 1, 2, ... in the order given."""
    lines = ["name: one-variable", "variables:", f"  - name: {name}", "    definition: Invented."]
    if ordinal:
        lines.append("    type: ordinal")
    lines.append("    categories:")
    for level, value in enumerate(codes):
        lines += [f'      - code: "{value}"', "        definition: Invented."]
        if ordinal:
            lines.append(f"        level: {level}")
    return "\n".join(lines) + "\n"

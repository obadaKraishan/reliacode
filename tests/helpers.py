"""Shared test helpers: paths, small YAML snippets, a mock HTTP server and a test clock."""

import json
import textwrap
from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples" / "codebooks"


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

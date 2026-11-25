"""The examples in docs/backends.md for request fields and one call per unit run, and print what
the page says they print. The HTTP example answers from a mock server: no request leaves the
test."""

import contextlib
import io
import itertools
import json
import re
import textwrap
from pathlib import Path

import httpx
import pytest

import reliacode
from helpers import completion
from reliacode import FakeBackend, Ledger, code, load_codebook

ROOT = Path(__file__).parent.parent
DOC = (ROOT / "docs" / "backends.md").read_text(encoding="utf-8")
BLOCKS = re.findall(r"```(\w+)\n(.*?)```", DOC, flags=re.DOTALL)


def block_with(marker: str) -> str:
    (body,) = [body for language, body in BLOCKS if language == "python" and marker in body]
    return textwrap.dedent(body)  # a block inside a list item is indented


@pytest.fixture
def mock_server(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every httpx client answers from an invented server; the request bodies are kept."""
    bodies: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        frame = "dominant_frame" in body["messages"][0]["content"].split("Variable: ")[1]
        text = '{"code": "economic"}' if frame else '{"code": "government"}'
        usage = {
            "prompt_tokens": 700,
            "completion_tokens": 104,
            "total_tokens": 804,
            "completion_tokens_details": {"reasoning_tokens": 96},
        }
        return httpx.Response(200, json=completion(text, usage=usage))

    real_client = httpx.Client

    def client(*, transport: object = None, timeout: float) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(answer), timeout=timeout)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.chdir(ROOT)
    return bodies


def test_each_example_prints_what_the_page_shows(mock_server: list[dict]) -> None:
    namespace: dict[str, object] = {}
    checked = 0
    for (language, body), (next_language, shown) in itertools.pairwise(BLOCKS):
        if language != "python" or next_language != "text":
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(body, "docs/backends.md", "exec"), namespace)
        assert printed.getvalue() == shown
        checked += 1
    assert checked == 3
    # The request-fields example sent no temperature, and the reasoning effort as given.
    assert len(mock_server) == 2
    for body in mock_server:
        assert "temperature" not in body
        assert body["reasoning_effort"] == "low"
        assert body["model"] == "openai/gpt-oss-120b"


def test_the_anthropic_backend_on_the_page_is_valid_and_sends_nothing_when_made() -> None:
    namespace: dict[str, object] = {"reliacode": reliacode}
    exec(compile(block_with("api.anthropic.com"), "docs/backends.md", "exec"), namespace)
    backend = namespace["backend"]
    assert isinstance(backend, reliacode.OpenAICompatibleBackend)
    assert backend.temperature is None
    assert backend.endpoint == "https://api.anthropic.com/v1/chat/completions"
    thinking = backend.extra_body["thinking"]
    assert 1024 <= thinking["budget_tokens"] < backend.max_tokens  # as the quoted page asks


def test_the_unit_prompt_on_the_page_states_the_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {}
    exec(compile(block_with("severity_rule = Rule("), "docs/backends.md", "exec"), namespace)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(block_with("build_unit_prompt"), "docs/backends.md", "exec"), namespace)
    assert (
        "Rules across variables (the codes must follow every one):\n"
        "- Severity is none if and only if incivility_form is none.\n" in printed.getvalue()
    )


def test_the_ledger_line_names_every_variable_as_the_page_says(tmp_path: Path) -> None:
    codebook = load_codebook(ROOT / "examples" / "codebooks" / "incivility.yaml")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    backend = FakeBackend(['{"incivility_form": "none", "severity": "none"}'])
    code(codebook, ["Thanks."], backend=backend, recipe="per_unit", ledger=ledger)
    (entry,) = ledger.entries()
    assert entry.variable == "incivility_form+severity"
    assert "(here `incivility_form+severity`)" in DOC

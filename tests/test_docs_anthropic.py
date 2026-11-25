"""The examples in the section of docs/backends.md on Anthropic's own API run, and print what the
page says they print. They answer from an invented Messages and Batches API: no request leaves
the test, and the counts are invented."""

import contextlib
import io
import itertools
import json
import re
import shutil
from pathlib import Path

import httpx
import pytest

import reliacode
from helpers import FakeBatchAPI, anthropic_message

ROOT = Path(__file__).parent.parent
FULL = (ROOT / "docs" / "backends.md").read_text(encoding="utf-8")
SECTION = FULL[FULL.index("\n## Anthropic's own API") :]
BLOCKS = re.findall(r"```(\w+)\n(.*?)```", SECTION, flags=re.DOTALL)
CODES = {
    "Thanks for posting the new library hours.": ("none", "none"),
    "Only a clown would vote for this bus plan.": ("name_calling", "mild"),
    "The ferry schedule changes on Monday.": ("none", "none"),
    "Stop lying about the budget, you hacks.": ("lying", "strong"),
    "Great work on the new crossing.": ("none", "none"),
}


def coded(params: dict) -> str:
    text = params["messages"][0]["content"].split("\n\n", 1)[1]
    form, severity = CODES[text]
    return json.dumps({"incivility_form": form, "severity": severity})


class InventedAPI(FakeBatchAPI):
    """The synchronous calls write the shared prompt to the cache once, then read it; the batch's
    requests read it."""

    def __init__(self) -> None:
        super().__init__(
            lambda custom_id, params: coded(params), polls=0, prefix="msgbatch_example"
        )
        self.sync_calls = 0
        self.usage = {
            "input_tokens": 40,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1712,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 200,
            "output_tokens_details": {"thinking_tokens": 170},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/messages":
            request.read()
            self.requests.append(request)
            self.sync_calls += 1
            first = self.sync_calls == 1
            usage = {
                "input_tokens": 46 if first else 51,
                "cache_creation_input_tokens": 1712 if first else 0,
                "cache_read_input_tokens": 0 if first else 1712,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 1712 if first else 0,
                },
                "output_tokens": 210 if first else 190,
                "output_tokens_details": {"thinking_tokens": 180 if first else 160},
            }
            text = coded(json.loads(request.content))
            return httpx.Response(200, json=anthropic_message(text, usage=usage))
        return super().__call__(request)


@pytest.fixture
def invented_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> InventedAPI:
    api = InventedAPI()
    real_client = httpx.Client

    def client(*, transport: object = None, timeout: float) -> httpx.Client:
        return real_client(transport=api.transport, timeout=timeout)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-the-docs")
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    monkeypatch.chdir(tmp_path)  # the examples write ledger.jsonl and run-state.json here
    return api


def test_each_example_prints_what_the_page_shows(invented_api: InventedAPI) -> None:
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
    created = invented_api.created()
    assert len(created) == 1  # one batch; the second call sent nothing
    for request in created[0]["requests"]:
        system = request["params"]["system"]
        assert system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert request["params"]["thinking"] == {"type": "adaptive"}
        assert "temperature" not in request["params"]
    state = Path("run-state.json").read_text(encoding="utf-8")
    assert "lying" not in state and "budget" not in state


def test_the_first_block_makes_the_backend_the_page_describes() -> None:
    (first,) = [body for language, body in BLOCKS[:1] if language == "python"]
    namespace: dict[str, object] = {"reliacode": reliacode}
    exec(compile(first, "docs/backends.md", "exec"), namespace)
    backend = namespace["backend"]
    assert isinstance(backend, reliacode.AnthropicBackend)
    assert (backend.model, backend.max_tokens, backend.cache) == ("claude-haiku-5-5", 4000, "1h")
    assert backend.thinking == {"type": "adaptive"}
    assert backend.effort is None and backend.temperature is None
    assert backend.api_version == "2026-10-21"


def test_the_page_names_the_settings_the_backend_has() -> None:
    import inspect

    names = [n for n in inspect.signature(reliacode.AnthropicBackend.__init__).parameters]
    for name in names[2:]:
        if name not in ("transport", "sleep", "clock"):
            assert f"`{name}`" in SECTION, name
    for field in ("api_version", "thinking", "effort", "cache"):
        assert f"`{field}`" in SECTION

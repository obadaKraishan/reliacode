"""The default recipe (one call per variable) builds the same prompts and sends the same messages,
byte for byte, as the version recorded in ``tests/fixtures/default_prompts.json``.

That file was made before the one-call-per-unit recipe was added (see
``tests/reference/make_prompt_reference.py``), so these tests show that adding it changed
nothing that the default recipe sends or reports. If one fails, the default prompts changed:
every reported prompt hash would change, which needs a new PROMPT_VERSION and a CHANGELOG entry.
"""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from helpers import FIXTURES
from reliacode import (
    Codebook,
    FakeBackend,
    OpenAICompatibleBackend,
    code,
    load_codebook,
    parse_codebook,
)
from reliacode.coding import (
    INSTRUCTION,
    PROMPT_VERSION,
    RETRY_TEMPLATE,
    USER_TEMPLATE,
    answer_schema,
    build_prompt,
)

ROOT = Path(__file__).parent.parent
REFERENCE = json.loads((FIXTURES / "default_prompts.json").read_text(encoding="utf-8"))
CODEBOOKS = REFERENCE["codebooks"]
TEXT = REFERENCE["text"]


def codebook_of(entry: dict) -> Codebook:
    if "file" in entry:
        return load_codebook(ROOT / entry["file"])
    return parse_codebook(entry["yaml"])


def by_name(entry: dict) -> str:
    return entry["name"]


def asked_again(messages: list[dict[str, str]]) -> str:
    """The same scripted answers as the reference script: unusable, then the first code."""
    if len(messages) == 2:
        return REFERENCE["first_answer"]
    first_code = messages[0]["content"].split("where the code is one of: ")[-1].split(",")[0]
    return json.dumps({"code": first_code.split(".")[0]})


def test_the_reference_was_made_before_the_second_recipe() -> None:
    assert REFERENCE["made_with"]["commit"] == "60410f5e66993eab5de986f812dc679e88ce5c22"
    assert len(CODEBOOKS) == 4
    assert sum(len(entry["prompts"]) for entry in CODEBOOKS) == 7


def test_the_recipe_constants_are_unchanged() -> None:
    assert REFERENCE["constants"] == {
        "PROMPT_VERSION": PROMPT_VERSION,
        "INSTRUCTION": INSTRUCTION,
        "USER_TEMPLATE": USER_TEMPLATE,
        "RETRY_TEMPLATE": RETRY_TEMPLATE,
    }
    assert PROMPT_VERSION == 1


@pytest.mark.parametrize("entry", CODEBOOKS, ids=by_name)
def test_every_prompt_is_byte_for_byte_the_same(entry: dict) -> None:
    codebook = codebook_of(entry)
    assert codebook.content_hash() == entry["content_hash"]
    for pinned in entry["prompts"]:
        prompt = build_prompt(codebook, pinned["variable"])
        assert prompt.system.encode("utf-8") == pinned["system"].encode("utf-8")
        assert prompt.user_template == pinned["user_template"]
        assert prompt.retry_template == pinned["retry_template"]
        assert prompt.json_schema == pinned["json_schema"]
        assert answer_schema(codebook.variable(pinned["variable"])) == pinned["json_schema"]
        assert prompt.hash == pinned["hash"]
        assert prompt.messages(TEXT) == pinned["messages"]


@pytest.mark.parametrize("entry", CODEBOOKS, ids=by_name)
def test_code_sends_the_same_messages_and_second_asks(entry: dict) -> None:
    codebook = codebook_of(entry)
    backend = FakeBackend(asked_again)
    run = code(codebook, {"u1": TEXT}, backend=backend)
    pinned = entry["asked_again"]
    assert backend.calls == pinned["calls"]
    assert backend.json_schemas == pinned["json_schemas"]
    assert [result.prompt_hash for result in run.results] == pinned["prompt_hashes"]
    assert [result.code for result in run.results] == pinned["codes"]
    assert [prompt.hash for prompt in run.prompts] == [p["hash"] for p in entry["prompts"]]


@pytest.mark.parametrize("entry", CODEBOOKS, ids=by_name)
def test_an_answer_never_usable_gives_the_same_messages_and_reasons(entry: dict) -> None:
    codebook = codebook_of(entry)
    backend = FakeBackend(lambda messages: REFERENCE["unusable_answer"])
    run = code(codebook, {"u1": TEXT}, backend=backend)
    assert backend.calls == entry["missing"]["calls"]
    assert [result.missing_reason for result in run.results] == entry["missing"]["missing_reasons"]


# The HTTP requests, recorded with the source of 60410f5 -------------------------------------------

REQUESTS = json.loads((FIXTURES / "default_requests.json").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    "make_request_reference", ROOT / "tests" / "reference" / "make_request_reference.py"
)
assert _spec is not None and _spec.loader is not None
REFERENCE_SCRIPT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REFERENCE_SCRIPT)


def in_order(body: str) -> object:
    """A request body with the order of its fields kept, to compare field by field."""
    return json.loads(body, object_pairs_hook=list)


@pytest.mark.parametrize(
    "pinned", REQUESTS["runs"], ids=lambda run: f"{Path(run['file']).stem}-{run['settings']}"
)
def test_the_http_requests_are_the_same_field_for_field_and_in_order(pinned: dict) -> None:
    sent: list[str] = []

    def server(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        return REFERENCE_SCRIPT.answer(request)

    backend = OpenAICompatibleBackend(
        "http://localhost:11434/v1",
        "reference-model",
        transport=httpx.MockTransport(server),
        **REQUESTS["settings"][pinned["settings"]],
    )
    run = code(load_codebook(ROOT / pinned["file"]), REQUESTS["texts"], backend=backend)
    assert run.missing() == ()
    assert len(sent) == len(pinned["requests"]) == 8
    assert [in_order(body) for body in sent] == [in_order(body) for body in pinned["requests"]]
    if httpx.__version__ == REQUESTS["made_with"]["httpx"]:
        assert sent == pinned["requests"]  # byte for byte, with the same JSON encoder


def test_the_request_reference_was_made_with_the_old_source() -> None:
    assert REQUESTS["made_with"]["source_commit"] == "60410f5e66993eab5de986f812dc679e88ce5c22"
    assert len(REQUESTS["runs"]) == 4
    first = in_order(REQUESTS["runs"][0]["requests"][0])
    assert [name for name, _ in first] == [
        "model",
        "messages",
        "temperature",
        "stream",
        "response_format",
    ]

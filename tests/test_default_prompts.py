"""The default recipe (one call per variable) builds the same prompts and sends the same messages,
byte for byte, as the version recorded in ``tests/fixtures/default_prompts.json``.

That file was made before the one-call-per-unit recipe was added (see
``tests/reference/make_prompt_reference.py``), so these tests show that adding it changed
nothing that the default recipe sends or reports. If one fails, the default prompts changed:
every reported prompt hash would change, which needs a new PROMPT_VERSION and a CHANGELOG entry.
"""

import json
from pathlib import Path

import pytest

from helpers import FIXTURES
from reliacode import Codebook, FakeBackend, code, load_codebook, parse_codebook
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

"""The content hash identifies what a codebook says, not how its YAML is written."""

import hashlib
import json
import re

import pytest

from helpers import FIXTURES, MINIMAL, yaml_text
from reliacode import load_codebook, parse_codebook
from reliacode.codebook import HASH_VERSION

#: Pinned hash of tests/fixtures/golden.yaml. If this test fails, the hash recipe changed:
#: that breaks every published hash, so it needs a new HASH_VERSION and a CHANGELOG entry.
GOLDEN_HASH = "sha256:e5864654c321c8ac46171bfafc4aef896c50fcd9f5da3a591c6f6c60c9bff8be"


def hash_of(text: str) -> str:
    return parse_codebook(text).content_hash()


def test_hash_format() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", hash_of(MINIMAL))


def test_pinned_golden_hash() -> None:
    assert load_codebook(FIXTURES / "golden.yaml").content_hash() == GOLDEN_HASH


def test_documented_recipe_can_be_reproduced_without_reliacode() -> None:
    # Compact JSON, sorted keys, UTF-8, defaults left out, wrapped with the recipe version.
    content = {
        "name": "tone-demo",
        "variables": [
            {
                "name": "tone",
                "definition": "The overall tone of the text toward its main subject.",
                "categories": [
                    {
                        "code": "positive",
                        "definition": "The text mostly praises or supports its subject.",
                    },
                    {
                        "code": "negative",
                        "definition": "The text mostly criticises or opposes its subject.",
                    },
                ],
            }
        ],
    }
    blob = json.dumps(
        {"reliacode_codebook_hash": HASH_VERSION, "codebook": content},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    codebook = parse_codebook(MINIMAL)
    assert codebook.canonical_json().encode("utf-8") == blob
    assert codebook.content_hash() == "sha256:" + hashlib.sha256(blob).hexdigest()


def test_file_and_text_give_the_same_hash(tmp_path) -> None:
    path = tmp_path / "tone.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert load_codebook(path).content_hash() == hash_of(MINIMAL)


def test_hash_is_the_same_on_every_call() -> None:
    codebook = parse_codebook(MINIMAL)
    assert codebook.content_hash() == codebook.content_hash() == hash_of(MINIMAL)


# --------------------------------------------------------------------- formatting does not matter

SAME_CONTENT = {
    "keys in another order": """
        variables:
          - categories:
              - definition: The text mostly praises or supports its subject.
                code: positive
              - definition: The text mostly criticises or opposes its subject.
                code: negative
            definition: The overall tone of the text toward its main subject.
            name: tone
        name: tone-demo
        """,
    "comments and other indentation": """
        # A comment at the top
        name: tone-demo   # trailing comment
        variables:
        -   name: tone
            definition: The overall tone of the text toward its main subject.
            categories:
            -   code: positive
                definition: The text mostly praises or supports its subject.
            -   code: negative   # another comment
                definition: The text mostly criticises or opposes its subject.
        """,
    "flow style and quotes": """
        name: "tone-demo"
        variables: [{name: 'tone',
          definition: "The overall tone of the text toward its main subject.",
          categories: [
            {code: "positive", definition: 'The text mostly praises or supports its subject.'},
            {code: negative, definition: The text mostly criticises or opposes its subject.}]}]
        """,
    "block scalars and surrounding whitespace": """
        name: "  tone-demo  "
        variables:
          - name: tone
            definition: |
              The overall tone of the text toward its main subject.
            categories:
              - code: positive
                definition: >
                  The text mostly praises
                  or supports its subject.
              - code: negative
                definition: "The text mostly criticises or opposes its subject.   "
        """,
    "defaults written out": """
        name: tone-demo
        version: null
        authors: []
        variables:
          - name: tone
            type: nominal
            decision_rules: []
            definition: The overall tone of the text toward its main subject.
            categories:
              - code: positive
                label:
                examples: []
                definition: The text mostly praises or supports its subject.
              - code: negative
                definition: The text mostly criticises or opposes its subject.
        """,
}


@pytest.mark.parametrize("variant", list(SAME_CONTENT), ids=list(SAME_CONTENT))
def test_formatting_does_not_change_the_hash(variant: str) -> None:
    assert hash_of(yaml_text(SAME_CONTENT[variant])) == hash_of(MINIMAL)


def test_example_shorthand_and_mapping_give_the_same_hash() -> None:
    base = MINIMAL.replace(
        "            definition: The text mostly praises or supports its subject.\n",
        "            definition: The text mostly praises or supports its subject.\n"
        "            examples:\n"
        "              - {example}\n",
    )
    assert hash_of(base.format(example="A fine result.")) == hash_of(
        base.format(example="{text: A fine result.}")
    )


def test_unicode_normal_forms_give_the_same_hash() -> None:
    composed = MINIMAL.replace("tone-demo", "café")
    decomposed = MINIMAL.replace("tone-demo", "café")
    assert composed != decomposed
    assert hash_of(composed) == hash_of(decomposed)


# --------------------------------------------------------------------- content does matter

CHANGED_CONTENT = {
    "a definition": ("mostly praises", "largely praises"),
    "a code": ("code: negative", "code: critical"),
    "the name": ("name: tone-demo", "name: tone-demo-2"),
    "inner whitespace": ("mostly praises or", "mostly  praises or"),
    "an added optional field": ("name: tone-demo\n", "name: tone-demo\nversion: '1'\n"),
}


@pytest.mark.parametrize("change", list(CHANGED_CONTENT), ids=list(CHANGED_CONTENT))
def test_changing_content_changes_the_hash(change: str) -> None:
    old, new = CHANGED_CONTENT[change]
    assert old in MINIMAL
    assert hash_of(MINIMAL.replace(old, new)) != hash_of(MINIMAL)


def test_reordering_categories_changes_the_hash() -> None:
    reordered = yaml_text(
        """
        name: tone-demo
        variables:
          - name: tone
            definition: The overall tone of the text toward its main subject.
            categories:
              - code: negative
                definition: The text mostly criticises or opposes its subject.
              - code: positive
                definition: The text mostly praises or supports its subject.
        """
    )
    assert parse_codebook(reordered).variables[0].codes == ("negative", "positive")
    assert hash_of(reordered) != hash_of(MINIMAL)


def test_changing_an_ordinal_level_changes_the_hash() -> None:
    text = (FIXTURES / "golden.yaml").read_text(encoding="utf-8")
    assert "level: 1" in text
    assert hash_of(text.replace("level: 1", "level: 2")) != GOLDEN_HASH

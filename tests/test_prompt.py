"""The prompt for each variable is built from the codebook alone, and its hash is pinned."""

import hashlib
import json

import pytest

from helpers import EXAMPLES, FIXTURES, MINIMAL, yaml_text
from reliacode import load_codebook, parse_codebook
from reliacode.coding import (
    PROMPT_VERSION,
    RETRY_TEMPLATE,
    USER_TEMPLATE,
    answer_schema,
    build_prompt,
)

#: Pinned prompt hash for the "severity" variable of tests/fixtures/golden.yaml. If this test
#: fails, the prompt changed: that changes every reported prompt hash, so bump PROMPT_VERSION
#: and add a CHANGELOG entry.
GOLDEN_PROMPT_HASH = "sha256:5ad0d44f599d32bdeb693663ec5ccae0c1939bc1d316f995da660256fcce9b0c"

GOLDEN_PROMPT = """\
You are a careful coder in a quantitative content analysis. Code one variable for the text you \
are given, following the codebook below exactly.

Codebook: golden, version 1.0
Language of the texts: en

Variable: severity
Definition: How strongly uncivil the comment is.
The categories are ordered: a higher level means more of what is measured.
Decision rules:
- If unsure between two levels, choose the lower one.

Categories (choose exactly one):
- none (level 0): No uncivil language.
  Example: "Merci, très clair."
- strong (level 1, قوي): Insults aimed at a person.
  Example: "You are an idiot." (A direct insult.)

Answer with only a JSON object of the form {"code": "..."}, where the code is one of: none, \
strong. Write nothing else."""

FULL = yaml_text(
    """
    name: park-coverage
    title: Coverage of the new city park
    description: How local news describes the new city park.
    unit_of_analysis: One news article.
    instructions: Read the whole article before coding.
    variables:
      - name: stance
        label: Stance toward the park
        definition: Whether the article favors or opposes the park.
        instructions: Judge the article, not the people it quotes.
        decision_rules:
          - If the article is balanced, code neutral.
          - Ignore the headline when it disagrees with the body.
        categories:
          - code: favor
            label: In favor
            definition: The article presents the park as good for the city.
            examples:
              - The new park gives families a place to meet.
          - code: oppose
            definition: The article presents the park as a mistake.
            examples:
              - text: Residents say the park cost too much.
                note: A cost complaint counts as opposition.
          - code: neutral
            definition: The article takes no side.
    """
)


def test_the_golden_prompt_is_pinned() -> None:
    prompt = build_prompt(load_codebook(FIXTURES / "golden.yaml"), "severity")
    assert prompt.system == GOLDEN_PROMPT
    assert prompt.hash == GOLDEN_PROMPT_HASH
    assert PROMPT_VERSION == 1


def test_every_part_of_the_codebook_reaches_the_prompt() -> None:
    system = build_prompt(parse_codebook(FULL), "stance").system
    for part in [
        "Codebook: Coverage of the new city park\n",
        "About this codebook: How local news describes the new city park.",
        "Unit of analysis: One news article.",
        "General instructions: Read the whole article before coding.",
        "Variable: stance (Stance toward the park)",
        "Definition: Whether the article favors or opposes the park.",
        "Instructions: Judge the article, not the people it quotes.",
        "Decision rules:\n- If the article is balanced, code neutral.\n- Ignore the headline",
        "- favor (In favor): The article presents the park as good for the city.",
        '  Example: "The new park gives families a place to meet."',
        "- oppose: The article presents the park as a mistake.",
        '  Example: "Residents say the park cost too much." (A cost complaint counts as',
        "- neutral: The article takes no side.",
        "where the code is one of: favor, oppose, neutral. Write nothing else.",
    ]:
        assert part in system, part
    assert "ordered" not in system  # only ordinal variables are described as ordered


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.stem)
def test_the_example_prompts_hold_every_code_rule_and_example(path) -> None:
    codebook = load_codebook(path)
    for variable in codebook.variables:
        system = build_prompt(codebook, variable.name).system
        assert all(rule in system for rule in variable.decision_rules)
        for category in variable.categories:
            assert f"- {category.code} (" in system
            assert category.definition in system
            assert all(example.text in system for example in category.examples)


def test_the_hash_recipe_can_be_reproduced_without_reliacode() -> None:
    codebook = parse_codebook(MINIMAL)
    prompt = build_prompt(codebook, "tone")
    payload = {
        "reliacode_prompt": 1,
        "system": prompt.system,
        "user": "Text to code:\n\n{text}",
        "retry": RETRY_TEMPLATE,
        "json_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "enum": ["positive", "negative"]}},
            "required": ["code"],
            "additionalProperties": False,
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert prompt.hash == "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_the_hash_follows_the_content_not_the_formatting() -> None:
    reformatted = MINIMAL.replace("name: tone-demo", "name:   'tone-demo'   # a comment")
    changed = MINIMAL.replace("mostly praises", "mainly praises")
    original = build_prompt(parse_codebook(MINIMAL), "tone").hash
    assert build_prompt(parse_codebook(reformatted), "tone").hash == original
    assert build_prompt(parse_codebook(changed), "tone").hash != original


def test_each_variable_has_its_own_prompt_and_hash() -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    form, severity = (build_prompt(codebook, name) for name in ("incivility_form", "severity"))
    assert form.variable == "incivility_form"
    assert form.hash != severity.hash
    assert "Variable: severity" in severity.system
    assert "Variable: severity" not in form.system


def test_the_schema_allows_only_the_codes() -> None:
    codebook = parse_codebook(MINIMAL)
    assert answer_schema(codebook.variable("tone")) == {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": ["positive", "negative"]}},
        "required": ["code"],
        "additionalProperties": False,
    }
    assert build_prompt(codebook, "tone").json_schema == answer_schema(codebook.variable("tone"))


def test_the_messages_carry_the_prompt_and_the_text() -> None:
    prompt = build_prompt(parse_codebook(MINIMAL), "tone")
    text = "The mayor's plan {finally} works."  # braces in a text are kept as they are
    assert prompt.messages(text) == [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": f"Text to code:\n\n{text}"},
    ]
    assert prompt.user_template == USER_TEMPLATE


def test_an_unknown_variable_is_named() -> None:
    with pytest.raises(KeyError, match=r'no variable "frame"\. Its variables are: tone'):
        build_prompt(parse_codebook(MINIMAL), "frame")

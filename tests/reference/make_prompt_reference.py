"""Record the prompts of the default recipe (one call per variable), byte for byte.

The output, ``tests/fixtures/default_prompts.json``, was made once, before the second recipe
(one call per unit) was added, and must never be made again with a later version: it is the
record that later versions are compared with. ``tests/test_default_prompts.py`` checks that the
default recipe still builds exactly these prompts and sends exactly these messages.

    uv run python tests/reference/make_prompt_reference.py

For each codebook below it records:

- every variable's prompt: the system text, the templates, the JSON Schema and the hash;
- the messages ``code()`` sends when the first answer cannot be used and the second can;
- the messages and the missing reason when no answer can be used.

The texts and the scripted answers are invented.
"""

import json
import platform
import subprocess
from pathlib import Path

from reliacode import FakeBackend, code, load_codebook, parse_codebook
from reliacode.coding import (
    INSTRUCTION,
    PROMPT_VERSION,
    RETRY_TEMPLATE,
    USER_TEMPLATE,
    build_prompt,
)

ROOT = Path(__file__).parent.parent.parent
OUTPUT = ROOT / "tests" / "fixtures" / "default_prompts.json"
FILES = [
    "examples/codebooks/incivility.yaml",
    "examples/codebooks/news_frames.yaml",
    "tests/fixtures/golden.yaml",
]
#: An invented codebook that uses every field the prompt can show, and codes written as numbers.
PARK = """\
name: park-coverage
version: "2.1"
title: Coverage of the new city park
description: How local news describes the new city park.
unit_of_analysis: One news article.
language: en
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
  - name: prominence
    label: How prominent the park is
    definition: How much of the article is about the park.
    type: ordinal
    categories:
      - code: "1"
        label: A mention
        definition: The park is mentioned in passing.
        level: 1
      - code: "2"
        definition: The park is one of several subjects.
        level: 2
      - code: "3"
        definition: The park is the main subject.
        level: 3
"""
TEXT = (
    'The new park opened on Saturday, and families filled the lawns by noon. "Finally," said one.'
)
FIRST_ANSWER = "I think it is positive."
UNUSABLE_ANSWER = '{"code": "Not-A-Code"}'


def asked_again(messages: list[dict[str, str]]) -> str:
    """Unusable at first, then the variable's first code."""
    if len(messages) == 2:
        return FIRST_ANSWER
    first_code = messages[0]["content"].split("where the code is one of: ")[-1].split(",")[0]
    return json.dumps({"code": first_code.split(".")[0]})


def record(name: str, codebook, source: dict[str, str]) -> dict[str, object]:
    prompts = []
    for variable in codebook.variables:
        prompt = build_prompt(codebook, variable.name)
        prompts.append(
            {
                "variable": prompt.variable,
                "system": prompt.system,
                "user_template": prompt.user_template,
                "retry_template": prompt.retry_template,
                "json_schema": prompt.json_schema,
                "hash": prompt.hash,
                "messages": prompt.messages(TEXT),
            }
        )
    again = FakeBackend(asked_again)
    run = code(codebook, {"u1": TEXT}, backend=again)
    assert run.missing() == (), name
    unusable = FakeBackend(lambda messages: UNUSABLE_ANSWER)
    missing = code(codebook, {"u1": TEXT}, backend=unusable)
    return {
        "name": name,
        **source,
        "content_hash": codebook.content_hash(),
        "prompts": prompts,
        "asked_again": {
            "calls": again.calls,
            "json_schemas": again.json_schemas,
            "prompt_hashes": [result.prompt_hash for result in run.results],
            "codes": [result.code for result in run.results],
        },
        "missing": {
            "calls": unusable.calls,
            "missing_reasons": [result.missing_reason for result in missing.results],
        },
    }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    codebooks = [record(path, load_codebook(ROOT / path), {"file": path}) for path in FILES]
    codebooks.append(record("park-coverage", parse_codebook(PARK), {"yaml": PARK}))
    data = {
        "about": (
            "The prompts of the default recipe (one call per variable), recorded before the "
            "one-call-per-unit recipe was added. Never regenerate this file."
        ),
        "made_with": {"commit": commit, "python": platform.python_version()},
        "text": TEXT,
        "first_answer": FIRST_ANSWER,
        "unusable_answer": UNUSABLE_ANSWER,
        "constants": {
            "PROMPT_VERSION": PROMPT_VERSION,
            "INSTRUCTION": INSTRUCTION,
            "USER_TEMPLATE": USER_TEMPLATE,
            "RETRY_TEMPLATE": RETRY_TEMPLATE,
        },
        "codebooks": codebooks,
    }
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} at {commit[:7]}")


if __name__ == "__main__":
    main()

"""Coding texts with a codebook: the prompt for each variable, and the check of each answer.

Each variable gets its own prompt, built only from the codebook: the codebook's title,
description, unit of analysis, language and instructions, then the variable's definition,
instructions and decision rules, and each category with its code, label, level, definition and
examples. The model is asked to answer with a JSON object, ``{"code": "..."}``.

The prompt hash identifies the exact prompt, so it can be reported with the results, next to the
codebook's content hash. It covers the prompt text, the templates used around each text and for
a second try, the JSON Schema of the answer, and the version of this recipe.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from reliacode.codebook import Codebook, Variable

#: Version of the prompt recipe. It is part of every prompt hash, so it changes every hash.
#: Bump it whenever the wording or layout of the prompt, the templates or the schema changes.
PROMPT_VERSION = 1

#: The first lines of every prompt.
INSTRUCTION = (
    "You are a careful coder in a quantitative content analysis. Code one variable for the "
    "text you are given, following the codebook below exactly."
)
#: The message that carries each text.
USER_TEMPLATE = "Text to code:\n\n{text}"
#: The message sent after an answer that cannot be used, before the model answers again.
RETRY_TEMPLATE = (
    "That answer cannot be used: {problem} Answer again with only a JSON object of the form "
    '{{"code": "..."}}, where the code is one of: {codes}.'
)


class Prompt(BaseModel):
    """The prompt for one variable, as it is sent, with its hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable: str
    system: str
    user_template: str
    retry_template: str
    json_schema: dict[str, Any]
    hash: str

    def messages(self, text: str) -> list[dict[str, str]]:
        """The chat messages that ask for the code of ``text``."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_template.format(text=text)},
        ]


def answer_schema(variable: Variable) -> dict[str, Any]:
    """The JSON Schema of a valid answer: an object whose ``code`` is one of the variable's codes.

    It follows the rules for strict structured output (every field required, no other fields).
    """
    return {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": list(variable.codes)}},
        "required": ["code"],
        "additionalProperties": False,
    }


def build_prompt(codebook: Codebook, variable: str) -> Prompt:
    """The prompt that asks for the code of ``variable``, built from ``codebook``.

    Raises:
        KeyError: the codebook has no variable with that name.
    """
    chosen = codebook.variable(variable)
    system = _system_text(codebook, chosen)
    schema = answer_schema(chosen)
    payload = {
        "reliacode_prompt": PROMPT_VERSION,
        "system": system,
        "user": USER_TEMPLATE,
        "retry": RETRY_TEMPLATE,
        "json_schema": schema,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return Prompt(
        variable=chosen.name,
        system=system,
        user_template=USER_TEMPLATE,
        retry_template=RETRY_TEMPLATE,
        json_schema=schema,
        hash="sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    )


def _system_text(codebook: Codebook, variable: Variable) -> str:
    title = codebook.title or codebook.name
    version = f", version {codebook.version}" if codebook.version else ""
    lines = [INSTRUCTION, "", f"Codebook: {title}{version}"]
    for heading, value in (
        ("About this codebook", codebook.description),
        ("Unit of analysis", codebook.unit_of_analysis),
        ("Language of the texts", codebook.language),
        ("General instructions", codebook.instructions),
    ):
        if value:
            lines.append(f"{heading}: {value}")

    label = f" ({variable.label})" if variable.label else ""
    lines += ["", f"Variable: {variable.name}{label}", f"Definition: {variable.definition}"]
    if variable.type == "ordinal":
        lines.append("The categories are ordered: a higher level means more of what is measured.")
    if variable.instructions:
        lines.append(f"Instructions: {variable.instructions}")
    if variable.decision_rules:
        lines.append("Decision rules:")
        lines += [f"- {rule}" for rule in variable.decision_rules]

    lines += ["", "Categories (choose exactly one):"]
    for category in variable.categories:
        details = [f"level {category.level}"] if category.level is not None else []
        details += [category.label] if category.label else []
        about = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {category.code}{about}: {category.definition}")
        for example in category.examples:
            note = f" ({example.note})" if example.note else ""
            lines.append(f'  Example: "{example.text}"{note}')

    codes = ", ".join(variable.codes)
    lines += [
        "",
        f'Answer with only a JSON object of the form {{"code": "..."}}, where the code is one '
        f"of: {codes}. Write nothing else.",
    ]
    return "\n".join(lines)

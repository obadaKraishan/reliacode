"""Coding texts with a codebook: the prompt for each variable, the check of each answer, and
:func:`code`, which codes every text for every variable with a backend.

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
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reliacode.backends import Backend, RequestParameters, Response, utc_now
from reliacode.codebook import Codebook, Variable
from reliacode.errors import describe

logger = logging.getLogger(__name__)

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
#: How much of a bad answer a problem message quotes.
_QUOTE = 60
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)

#: The message sent after an answer that cannot be used, before the model answers again.
RETRY_TEMPLATE = (
    "That answer cannot be used: {problem} Answer again with only a JSON object of the form "
    '{{"code": "..."}}, where the code is one of: {codes}.'
)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Prompt(_Record):
    """The prompt for one variable, as it is sent, with its hash."""

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


def check_answer(text: str, codes: Sequence[str]) -> tuple[str | None, str | None]:
    """Read the code from a model's answer, and say why when it cannot be used.

    A valid answer is a JSON object whose ``code`` is exactly one of ``codes``, such as
    ``{"code": "conflict"}``. Other fields are ignored. Spaces around the answer or the code, a
    code fence (```json ... ```) around the object, and a ``<think>...</think>`` block before it
    are allowed. A whole number is read as text, for codes such as "1". Nothing else is changed:
    upper and lower case must match, and a near miss is never corrected to the closest code.

    Returns:
        ``(code, None)`` for a valid answer, or ``(None, problem)``, where ``problem`` says in
        plain words what is wrong with the answer.
    """
    cleaned = _THINKING.sub("", text).strip()
    fence = _FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        return None, "The answer is empty."
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None, f"The answer is not a JSON object: {_quoted(cleaned)}."
    if not isinstance(data, dict):
        return None, f'The answer is not a JSON object with a "code" field: {_quoted(cleaned)}.'
    if "code" not in data:
        return None, f'The answer has no "code" field: {_quoted(cleaned)}.'
    value = data["code"]
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None, f'The "code" in the answer is not text: {_quoted(json.dumps(value))}.'
    value = value.strip()
    if value not in codes:
        return None, f"{_quoted(value)} is not one of the codes."
    return value, None


def _quoted(text: str) -> str:
    short = " ".join(text.split())
    if len(short) > _QUOTE:
        short = short[: _QUOTE - 3] + "..."
    return f'"{short}"'


#: A text's id: its position in a list, or its key in a mapping.
UnitId = str | int


class CodingResult(_Record):
    """The code of one variable for one text, with the record of how it was obtained.

    ``code`` is ``None`` when no answer could be used after every try. ``missing_reason`` then
    says why, and ``answers`` keeps every raw answer, so an answer outside the codebook's codes
    never enters the results unseen. ``responses`` holds the full record of each answer: the
    model, the endpoint, the parameters, the token counts and the time.
    """

    unit: UnitId
    variable: str
    code: str | None
    missing_reason: str | None = None
    answers: tuple[str, ...]
    responses: tuple[Response, ...] = Field(min_length=1)
    codebook_hash: str
    prompt_hash: str

    @property
    def is_missing(self) -> bool:
        """True when no answer could be used."""
        return self.code is None

    @property
    def model(self) -> str:
        """The model that gave the last answer: the name the server returned, or else the name
        that was asked for."""
        last = self.responses[-1]
        return last.model or last.parameters.model

    @property
    def endpoint(self) -> str:
        """Where the requests went."""
        return self.responses[-1].parameters.endpoint

    @property
    def parameters(self) -> RequestParameters:
        """The settings of the last request: model, temperature, seed and token limit."""
        return self.responses[-1].parameters

    @property
    def created_at(self) -> datetime:
        """When the last answer came, in UTC."""
        return self.responses[-1].created_at


class CodingRun(_Record):
    """Everything one call of :func:`code` produced: the results, the prompts and what was used."""

    codebook_name: str
    codebook_version: str | None = None
    codebook_hash: str
    prompts: tuple[Prompt, ...]
    results: tuple[CodingResult, ...]
    started_at: datetime
    finished_at: datetime
    reliacode_version: str

    def table(self) -> list[dict[str, UnitId | str | None]]:
        """One row per text, in the order given: the text's id under ``"unit"``, then one column
        per variable with its code, or ``None`` where it is missing.

        ``pandas.DataFrame(run.table())`` turns it into a data frame.
        """
        rows: dict[UnitId, dict[str, UnitId | str | None]] = {}
        for result in self.results:
            rows.setdefault(result.unit, {"unit": result.unit})[result.variable] = result.code
        return list(rows.values())

    def missing(self) -> tuple[CodingResult, ...]:
        """The results with no usable answer, to check by hand."""
        return tuple(result for result in self.results if result.code is None)


def code(
    codebook: Codebook,
    texts: Sequence[str] | Mapping[UnitId, str],
    *,
    backend: Backend,
    variables: Sequence[str] | str | None = None,
    attempts: int = 2,
) -> CodingRun:
    """Code every text for every variable of the codebook (or the ones named) with ``backend``.

    Each text is coded for one variable at a time, with that variable's prompt
    (:func:`build_prompt`), and each answer is checked against the variable's codes
    (:func:`check_answer`). When an answer cannot be used, the model is shown its answer and the
    reason and asked again, up to ``attempts`` answers in all. If none can be used, the result's
    code is ``None``, with the reason and every raw answer kept.

    Args:
        codebook: the codebook to code with, from :func:`reliacode.load_codebook`.
        texts: the texts to code: a list (each text's id is its position, counting from 0) or a
            mapping from ids to texts, such as ``dict(zip(df["id"], df["text"]))``.
        backend: where the prompts go, such as an :class:`~reliacode.OpenAICompatibleBackend`
            or a :class:`~reliacode.FakeBackend`.
        variables: the name of one variable, or a list of names, to code only those. All
            variables are coded by default.
        attempts: the most answers to ask for, for each text and variable. The default, 2,
            allows one more try after an answer that cannot be used.

    Raises:
        BackendError: the backend could not answer, for example because no server is running.
            The run stops there, since the requests that follow would fail the same way.
        TypeError, ValueError: ``texts``, ``variables`` or ``attempts`` are not as described.
    """
    if not isinstance(codebook, Codebook):
        raise TypeError(
            'codebook should be a codebook loaded with reliacode.load_codebook("file.yaml").'
        )
    if not isinstance(backend, Backend):
        raise TypeError(
            "backend should be a reliacode backend, such as OpenAICompatibleBackend(...) or "
            f"FakeBackend(...), but it is {describe(backend)}."
        )
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(
            f"attempts should be a whole number of at least 1, but it is {attempts!r}."
        )
    units = _units(texts)
    names = _variable_names(codebook, variables)
    prompts = {name: build_prompt(codebook, name) for name in names}
    codebook_hash = codebook.content_hash()
    started = utc_now()
    results = [
        _code_one(
            backend, codebook.variable(name), prompts[name], unit, text, attempts, codebook_hash
        )
        for unit, text in units
        for name in names
    ]
    from reliacode import __version__  # here, because the package imports this module

    return CodingRun(
        codebook_name=codebook.name,
        codebook_version=codebook.version,
        codebook_hash=codebook_hash,
        prompts=tuple(prompts.values()),
        results=tuple(results),
        started_at=started,
        finished_at=utc_now(),
        reliacode_version=__version__,
    )


def _code_one(
    backend: Backend,
    variable: Variable,
    prompt: Prompt,
    unit: UnitId,
    text: str,
    attempts: int,
    codebook_hash: str,
) -> CodingResult:
    messages = prompt.messages(text)
    answers: list[str] = []
    responses: list[Response] = []
    problem: str | None = None
    for attempt in range(1, attempts + 1):
        response = backend.complete(messages, json_schema=prompt.json_schema)
        if not isinstance(response, Response):
            raise TypeError(
                "The backend's complete() should return a reliacode Response, but it returned "
                f"{describe(response)}."
            )
        responses.append(response)
        answers.append(response.text)
        value, problem = check_answer(response.text, variable.codes)
        if value is not None:
            break
        logger.info(
            "%s, variable %s: answer %d of %d cannot be used. %s",
            _unit_name(unit),
            variable.name,
            attempt,
            attempts,
            problem,
        )
        retry = prompt.retry_template.format(problem=problem, codes=", ".join(variable.codes))
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": retry},
        ]
    else:
        value = None
    reason = None
    if value is None:
        reason = (
            f"The answer could not be used. {problem}"
            if attempts == 1
            else f"None of the {attempts} answers could be used. The last one: {problem}"
        )
    return CodingResult(
        unit=unit,
        variable=variable.name,
        code=value,
        missing_reason=reason,
        answers=tuple(answers),
        responses=tuple(responses),
        codebook_hash=codebook_hash,
        prompt_hash=prompt.hash,
    )


def _units(texts: object) -> list[tuple[UnitId, str]]:
    shape = "a list of texts, or a mapping from ids to texts"
    if isinstance(texts, (str, bytes)):
        raise TypeError(
            f"texts should be {shape}, not a single text. To code one text, write [text]."
        )
    if isinstance(texts, Mapping):
        items = list(texts.items())
    elif isinstance(texts, Iterable):
        items = list(enumerate(texts))
    else:
        raise TypeError(f"texts should be {shape}, but it is {describe(texts)}.")
    if not items:
        raise ValueError("texts is empty, so there is nothing to code.")
    for unit, text in items:
        if isinstance(unit, bool) or not isinstance(unit, (str, int)):
            raise TypeError(
                f"Text ids should be text or whole numbers, but one is {describe(unit)}."
            )
        if not isinstance(text, str):
            raise TypeError(
                f"{_unit_name(unit)} should be text, but it is {describe(text)}. Remove missing "
                "texts (such as NaN in a data frame) before coding."
            )
        if not text.strip():
            raise ValueError(
                f"{_unit_name(unit)} is empty. Remove it before coding, or record it as missing "
                "yourself."
            )
    return items


def _variable_names(codebook: Codebook, variables: Sequence[str] | str | None) -> list[str]:
    if variables is None:
        return [variable.name for variable in codebook.variables]
    names = [variables] if isinstance(variables, str) else list(variables)
    if not names:
        raise ValueError("variables is empty. Name at least one, or leave it out to code them all.")
    for name in names:
        codebook.variable(name)  # a KeyError that lists the codebook's variables
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'The variable "{repeated[0]}" is named more than once in variables.')
    return names


def _unit_name(unit: UnitId) -> str:
    return f'The text "{unit}"' if isinstance(unit, str) else f"Text {unit}"

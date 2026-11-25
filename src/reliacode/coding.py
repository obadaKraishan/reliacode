"""Coding texts with a codebook: the prompts, the check of each answer, and :func:`code`, which
codes every text for every variable with a backend.

There are two recipes. By default each variable gets its own prompt and its own call (one call
per variable), built only from the codebook: the codebook's title, description, unit of
analysis, language and instructions, then the variable's definition, instructions and decision
rules, and each category with its code, label, level, definition and examples. The model is
asked to answer with a JSON object, ``{"code": "..."}``.

With ``recipe="per_unit"``, one call per text asks for all the chosen variables at once, with
the same parts of the codebook for each, and the answer is one JSON object with one code per
variable. Rules across variables can be given with it (:class:`Rule`); they are stated in the
prompt and checked in the answer.

The prompt hash identifies the exact prompt, so it can be reported with the results, next to the
codebook's content hash. It covers the prompt text, the templates used around each text and for
a second try, the JSON Schema of the answer, and the version of the recipe.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliacode.backends import Backend, RequestParameters, Response, utc_now
from reliacode.codebook import Codebook, Variable
from reliacode.errors import describe
from reliacode.ledger import BudgetError, Ledger, LedgerEntry, Outcome

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

#: The two recipes: one call per variable (the default), or one call per text (unit) that asks
#: for all the chosen variables together.
Recipe = Literal["per_variable", "per_unit"]
_RECIPES = ("per_variable", "per_unit")

#: Version of the one-call-per-unit recipe. It is part of every unit prompt's hash. Bump it
#: whenever the wording or layout of that prompt, its templates or its schema changes.
UNIT_PROMPT_VERSION = 1
#: The first lines of every prompt that asks for several variables in one call.
UNIT_INSTRUCTION = (
    "You are a careful coder in a quantitative content analysis. Code every variable below for "
    "the text you are given, following the codebook below exactly."
)
#: The message sent after an answer that cannot be used as it is, before the model answers
#: again. ``{problems}`` lists each problem on a line of its own; none quotes the text.
UNIT_RETRY_TEMPLATE = (
    "That answer cannot be used as it is:\n{problems}\nAnswer again with only a JSON object "
    "with one field for each variable ({names}), each holding one of that variable's codes. "
    "Write nothing else."
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


class Rule(_Record):
    """A rule across variables, for one call per unit: a sentence for the prompt, the variables
    it names, and a check on their codes.

    Give rules with ``code(..., recipe="per_unit", rules=[...])``::

        Rule(
            text="Severity is none if and only if incivility_form is none.",
            variables=["incivility_form", "severity"],
            check=lambda codes: (codes["severity"] == "none")
            == (codes["incivility_form"] == "none"),
        )

    The sentence is stated in the prompt, under "Rules across variables". The check gets the
    codes of the variables the rule names, as a dict such as ``{"incivility_form": "none",
    "severity": "none"}``, and returns ``True`` when they follow the rule. It is run only when
    each of those variables has a valid code. When it returns ``False``, the model is asked
    again; if the last answer still breaks the rule, the variables it names are missing and the
    others keep their codes.

    The prompt hash covers the sentence and the variables, but not the check's code, so keep the
    check with your analysis script. Spaces and line breaks in the sentence are written as single
    spaces.
    """

    text: str
    variables: tuple[str, ...]
    check: Callable[[dict[str, str]], bool] = Field(exclude=True, repr=False)

    @field_validator("text")
    @classmethod
    def _one_sentence(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("give the rule as a sentence for the prompt")
        return value

    @field_validator("variables")
    @classmethod
    def _named(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("name the variables the rule is about, such as ['form', 'severity']")
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(f'the variable "{repeated[0]}" is named more than once')
        return value


class UnitPrompt(_Record):
    """The prompt that asks for several variables of one text in one call, with its hash.

    It is used by the one-call-per-unit recipe (``code(..., recipe="per_unit")``).
    ``variables`` are the variables asked, in order, and ``rules`` maps each rule's sentence to
    the variables it names. The hash covers the prompt text, the templates, the JSON Schema, the
    rules' sentences and variables, and the recipe's version.
    """

    recipe: Literal["per_unit"] = "per_unit"
    recipe_version: int
    variables: tuple[str, ...]
    system: str
    user_template: str
    retry_template: str
    json_schema: dict[str, Any]
    rules: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    hash: str

    def messages(self, text: str) -> list[dict[str, str]]:
        """The chat messages that ask for every variable's code of ``text``."""
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


def unit_answer_schema(variables: Sequence[Variable]) -> dict[str, Any]:
    """The JSON Schema of a valid answer that codes several variables at once: an object with
    one field per variable, whose value is one of that variable's codes.

    It follows the rules for strict structured output (every field required, no other fields).
    """
    return {
        "type": "object",
        "properties": {
            variable.name: {"type": "string", "enum": list(variable.codes)}
            for variable in variables
        },
        "required": [variable.name for variable in variables],
        "additionalProperties": False,
    }


def build_unit_prompt(
    codebook: Codebook,
    variables: Sequence[str] | str | None = None,
    rules: Sequence[Rule] | None = None,
) -> UnitPrompt:
    """The prompt that asks for all of ``codebook``'s variables (or the ones named) of a text in
    one call, built from the codebook, with the rules across variables stated in it.

    Raises:
        KeyError: the codebook has no variable with a name given.
        TypeError, ValueError: ``variables`` or ``rules`` are not as described in :func:`code`.
    """
    names = _variable_names(codebook, variables)
    chosen = [codebook.variable(name) for name in names]
    stated = _check_rules(rules, names)
    system = _unit_system_text(codebook, chosen, stated)
    schema = unit_answer_schema(chosen)
    payload = {
        "reliacode_prompt": UNIT_PROMPT_VERSION,
        "recipe": "per_unit",
        "system": system,
        "user": USER_TEMPLATE,
        "retry": UNIT_RETRY_TEMPLATE,
        "json_schema": schema,
        "rules": [[rule.text, list(rule.variables)] for rule in stated],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return UnitPrompt(
        recipe_version=UNIT_PROMPT_VERSION,
        variables=tuple(names),
        system=system,
        user_template=USER_TEMPLATE,
        retry_template=UNIT_RETRY_TEMPLATE,
        json_schema=schema,
        rules={rule.text: rule.variables for rule in stated},
        hash="sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    )


def _check_rules(rules: object, names: Sequence[str]) -> tuple[Rule, ...]:
    """The rules, checked against the variables asked, before any call is made."""
    if rules is None:
        return ()
    shape = 'a list of Rule(text="...", variables=[...], check=...)'
    if isinstance(rules, (str, bytes, Rule)) or not isinstance(rules, Sequence):
        raise TypeError(f"rules should be {shape}, but it is {describe(rules)}.")
    checked: list[Rule] = []
    for number, rule in enumerate(rules, 1):
        if not isinstance(rule, Rule):
            raise TypeError(f"Rule {number} should be a Rule(...), but it is {describe(rule)}.")
        for name in rule.variables:
            if name not in names:
                raise ValueError(
                    f'Rule {number} ("{rule.text}") names "{name}", which is not among the '
                    f"variables coded ({', '.join(names)}). Code that variable too, or leave the "
                    "rule out."
                )
        for earlier, other in enumerate(checked, 1):
            if other.text == rule.text:
                raise ValueError(
                    f"Rule {number} has the same sentence as rule {earlier}. Give each rule once."
                )
        checked.append(rule)
    return tuple(checked)


def _unit_system_text(
    codebook: Codebook, variables: Sequence[Variable], rules: Sequence[Rule]
) -> str:
    lines = [UNIT_INSTRUCTION, "", *_codebook_lines(codebook)]
    for variable in variables:
        lines += _variable_lines(variable)
    if rules:
        lines += ["", "Rules across variables (the codes must follow every one):"]
        lines += [f"- {rule.text}" for rule in rules]
    form = json.dumps({variable.name: "..." for variable in variables}, ensure_ascii=False)
    lines += ["", "Answer with only a JSON object with one field for each variable, in this form:"]
    lines.append(form)
    lines += [
        f"The codes for {variable.name} are: {', '.join(variable.codes)}." for variable in variables
    ]
    lines.append("Write nothing else.")
    return "\n".join(lines)


def _system_text(codebook: Codebook, variable: Variable) -> str:
    lines = [INSTRUCTION, "", *_codebook_lines(codebook), *_variable_lines(variable)]
    codes = ", ".join(variable.codes)
    lines += [
        "",
        f'Answer with only a JSON object of the form {{"code": "..."}}, where the code is one '
        f"of: {codes}. Write nothing else.",
    ]
    return "\n".join(lines)


def _codebook_lines(codebook: Codebook) -> list[str]:
    """The codebook's title and version, then its description, unit, language and instructions."""
    title = codebook.title or codebook.name
    version = f", version {codebook.version}" if codebook.version else ""
    lines = [f"Codebook: {title}{version}"]
    for heading, value in (
        ("About this codebook", codebook.description),
        ("Unit of analysis", codebook.unit_of_analysis),
        ("Language of the texts", codebook.language),
        ("General instructions", codebook.instructions),
    ):
        if value:
            lines.append(f"{heading}: {value}")
    return lines


def _variable_lines(variable: Variable) -> list[str]:
    """A blank line, the variable with its definition, instructions and decision rules, then a
    blank line and its categories with their examples."""
    label = f" ({variable.label})" if variable.label else ""
    lines = ["", f"Variable: {variable.name}{label}", f"Definition: {variable.definition}"]
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
    return lines


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
    value, problem, _ = _check(text, codes)
    return value, problem


def _check(text: str, codes: Sequence[str]) -> tuple[str | None, str | None, str | None]:
    """:func:`check_answer`, plus the kind of problem in words that quote nothing from the
    answer, for the log."""
    cleaned = _cleaned(text)
    if not cleaned:
        return None, "The answer is empty.", "the answer is empty"
    try:
        data = json.loads(cleaned)
    except ValueError:
        return (
            None,
            f"The answer is not a JSON object: {_quoted(cleaned)}.",
            "the answer is not a JSON object",
        )
    if not isinstance(data, dict):
        return (
            None,
            f'The answer is not a JSON object with a "code" field: {_quoted(cleaned)}.',
            'the answer is not a JSON object with a "code" field',
        )
    if "code" not in data:
        return (
            None,
            f'The answer has no "code" field: {_quoted(cleaned)}.',
            'the answer has no "code" field',
        )
    value = data["code"]
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return (
            None,
            f'The "code" in the answer is not text: {_quoted(json.dumps(value))}.',
            'the "code" in the answer is not text',
        )
    value = value.strip()
    if value not in codes:
        return (
            None,
            f"{_quoted(value)} is not one of the codes.",
            "the code in the answer is not one of the codes",
        )
    return value, None, None


def _cleaned(text: str) -> str:
    """An answer without a thinking block before it or a code fence around it."""
    cleaned = _THINKING.sub("", text).strip()
    fence = _FENCE.match(cleaned)
    return fence.group(1).strip() if fence else cleaned


def _quoted(text: str) -> str:
    short = " ".join(text.split())
    if len(short) > _QUOTE:
        short = short[: _QUOTE - 3] + "..."
    return f'"{short}"'


class _UnitAnswer(NamedTuple):
    """What one answer for several variables gave: the usable codes, why each other variable
    cannot be used, each problem once for the second ask, and the kinds of problem for the log.
    None of these words quotes the answer or the text."""

    codes: dict[str, str]
    problems: dict[str, str]
    retry: list[str]
    kinds: list[str]


def _check_unit(
    text: str, variables: Sequence[Variable], rules: Sequence[Rule] = ()
) -> _UnitAnswer:
    """Read every variable's code from an answer that should be one JSON object with one field
    per variable, then check the rules. The same leniency as :func:`check_answer` applies to
    each code. A broken rule makes the variables it names unusable."""
    cleaned = _cleaned(text)
    data: object = None
    if cleaned:
        try:
            data = json.loads(cleaned)
        except ValueError:
            data = None
    if not isinstance(data, dict):
        problem, kind = (
            ("The answer is not a JSON object.", "the answer is not a JSON object")
            if cleaned
            else ("The answer is empty.", "the answer is empty")
        )
        every = {variable.name: problem for variable in variables}
        return _UnitAnswer({}, every, [problem], [kind])
    answer = _UnitAnswer({}, {}, [], [])
    for variable in variables:
        name = variable.name
        value = data.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if name not in data:
            problem, kind = f'The answer has no field for "{name}".', f"no field for {name}"
        elif not isinstance(value, str):
            problem = f'The value for "{name}" is not text.'
            kind = f"the value for {name} is not text"
        elif value.strip() not in variable.codes:
            codes = ", ".join(variable.codes)
            problem = f'The code for "{name}" is not one of its codes: {codes}.'
            kind = f"the code for {name} is not one of its codes"
        else:
            answer.codes[name] = value.strip()
            continue
        answer.problems[name] = problem
        answer.retry.append(problem)
        answer.kinds.append(kind)
    broken = [
        (number, rule)
        for number, rule in enumerate(rules, 1)
        if all(name in answer.codes for name in rule.variables)
        and not _follows(number, rule, {name: answer.codes[name] for name in rule.variables})
    ]
    for number, rule in broken:
        problem = f"The codes break this rule: {rule.text}"
        answer.retry.append(problem)
        answer.kinds.append(f"rule {number} is broken")
        for name in rule.variables:
            answer.codes.pop(name, None)
            earlier = answer.problems.get(name)
            answer.problems[name] = f"{earlier} {problem}" if earlier else problem
    return answer


def _follows(number: int, rule: Rule, codes: dict[str, str]) -> bool:
    """Run a rule's check on the codes of the variables it names."""
    try:
        kept = rule.check(dict(codes))
    except Exception as error:
        example = json.dumps(dict.fromkeys(rule.variables, "..."))
        raise ValueError(
            f'The check of rule {number} ("{rule.text}") failed with {type(error).__name__}: '
            f"{error}. A check gets the codes of the variables the rule names, as a dict such as "
            f"{example}, and returns True when they follow the rule."
        ) from error
    if not isinstance(kept, bool):
        raise TypeError(
            f'The check of rule {number} ("{rule.text}") should return True or False, but it '
            f"returned {describe(kept)}."
        )
    return kept


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
    """Everything one call of :func:`code` produced: the results, the prompts and what was used.

    ``recipe`` says how the texts were asked: ``"per_variable"`` (one call per variable, with
    one :class:`Prompt` per variable) or ``"per_unit"`` (one call per text for all the variables,
    with one :class:`UnitPrompt`). Either way there is one result per text and variable.
    """

    codebook_name: str
    codebook_version: str | None = None
    codebook_hash: str
    prompts: tuple[Prompt | UnitPrompt, ...]
    results: tuple[CodingResult, ...]
    started_at: datetime
    finished_at: datetime
    reliacode_version: str
    recipe: Recipe = "per_variable"

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
    ledger: Ledger | None = None,
    recipe: Recipe = "per_variable",
    rules: Sequence[Rule] | None = None,
) -> CodingRun:
    """Code every text for every variable of the codebook (or the ones named) with ``backend``.

    By default each text is coded for one variable at a time, with that variable's prompt
    (:func:`build_prompt`), and each answer is checked against the variable's codes
    (:func:`check_answer`). When an answer cannot be used, the model is shown its answer and the
    reason and asked again, up to ``attempts`` answers in all. If none can be used, the result's
    code is ``None``, with the reason and every raw answer kept.

    With ``recipe="per_unit"``, one call per text asks for all the variables together
    (:func:`build_unit_prompt`), and the answer must be one JSON object with one code per
    variable, each checked against its variable's codes. An answer that cannot be used as it is
    gets one more ask (with ``attempts=2``) that names each problem without quoting the text.
    The last answer decides: a variable it gives no usable code for is missing, with the reason,
    and the other variables keep their codes. There is still one result per text and variable.
    Rules across variables (:class:`Rule`) are stated in the prompt and checked in each answer;
    a broken rule is asked again, and if the last answer still breaks it, the variables it names
    are missing.

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
        ledger: a :class:`~reliacode.Ledger` that records every call, including second asks
            and failures, and keeps its budget. Without one, nothing is recorded. With one call
            per text, a line's ``variable`` names every variable asked, joined by ``+``.
        recipe: ``"per_variable"`` (the default: one call per variable) or ``"per_unit"`` (one
            call per text, for all the variables together).
        rules: rules across variables, as a list of :class:`Rule`, for ``recipe="per_unit"``
            only. Each names only variables that are coded.

    Raises:
        BackendError: the backend could not answer, for example because no server is running.
            The run stops there, since the requests that follow would fail the same way.
        BudgetError: the ledger's budget refused a call. The run stops before that call, and
            the error's ``run`` holds the results finished before the stop. A cost limit for
            a model with no price is refused before the first call, with no ``run``.
        TypeError, ValueError: ``texts``, ``variables``, ``attempts``, ``ledger``, ``recipe``
            or ``rules`` are not as described, or a rule's check fails or does not return True
            or False.
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
    plan = _plan(
        codebook,
        texts,
        variables=variables,
        attempts=attempts,
        ledger=ledger,
        recipe=recipe,
        rules=rules,
        backend=backend,
    )
    started = utc_now()
    results: list[CodingResult] = []
    try:
        for task in plan.tasks(codebook):
            while not task.done:
                task.take(_ask(backend, task.messages, task.work, task.text, task.ask))
            results.extend(task.results())
    except BudgetError as error:
        error.run = plan.run(codebook, results, started)
        raise
    return plan.run(codebook, results, started)


class _Plan(NamedTuple):
    """What a run of :func:`code` (or of :func:`reliacode.code_in_batches`) is made of, once
    every argument is checked."""

    units: list[tuple["UnitId", str]]
    names: list[str]
    rules: tuple[Rule, ...]
    prompts: dict[str, "Prompt | UnitPrompt"]
    codebook_hash: str
    ledger: Ledger | None
    model: str | None
    recipe: "Recipe"
    attempts: int

    def tasks(self, codebook: Codebook) -> Iterator["_VariableTask | _UnitTask"]:
        """One task per text and variable (or per text, for one call per unit), in order."""
        for unit, text in self.units:
            if self.recipe == "per_unit":
                (unit_prompt,) = self.prompts.values()
                assert isinstance(unit_prompt, UnitPrompt)
                chosen = tuple(codebook.variable(name) for name in self.names)
                work = _UnitWork(
                    chosen, self.rules, unit_prompt, self.codebook_hash, self.ledger, self.model
                )
                yield _UnitTask(work, unit, text, self.attempts)
                continue
            for name in self.names:
                prompt = self.prompts[name]
                assert isinstance(prompt, Prompt)
                one = _Work(
                    codebook.variable(name), prompt, self.codebook_hash, self.ledger, self.model
                )
                yield _VariableTask(one, unit, text, self.attempts)

    def run(
        self, codebook: Codebook, results: Sequence["CodingResult"], started: datetime
    ) -> "CodingRun":
        return _run(codebook, self.codebook_hash, self.prompts, results, started, self.recipe)


def _plan(
    codebook: Codebook,
    texts: object,
    *,
    variables: Sequence[str] | str | None,
    attempts: object,
    ledger: object,
    recipe: object,
    rules: Sequence[Rule] | None,
    backend: object,
) -> _Plan:
    """Check the arguments of a run, in the order :func:`code` has always checked them, and
    build its prompts. A cost limit for a model with no price is refused here."""
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(
            f"attempts should be a whole number of at least 1, but it is {attempts!r}."
        )
    if ledger is not None and not isinstance(ledger, Ledger):
        raise TypeError(
            'ledger should be a reliacode Ledger, such as Ledger("ledger.jsonl"), but it is '
            f"{describe(ledger)}."
        )
    if recipe not in _RECIPES:
        shown = repr(recipe) if isinstance(recipe, str) else describe(recipe)
        raise ValueError(
            'recipe should be "per_variable" (one call per variable, the default) or "per_unit" '
            f"(one call per text, for all the variables together), but it is {shown}."
        )
    if recipe == "per_variable" and rules:
        raise ValueError(
            "rules are checked only when one call asks for all the variables together: use "
            'recipe="per_unit" with them. With one call per variable, the model never sees its '
            "answers for the other variables."
        )
    units = _units(texts)
    names = _variable_names(codebook, variables)
    prompts: dict[str, Prompt | UnitPrompt]
    stated: tuple[Rule, ...] = ()
    if recipe == "per_unit":
        stated = _check_rules(rules, names)
        prompts = {"+".join(names): build_unit_prompt(codebook, names, stated)}
    else:
        prompts = {name: build_prompt(codebook, name) for name in names}
    codebook_hash = codebook.content_hash()
    model = getattr(backend, "model", None)
    model = model if isinstance(model, str) else None
    if ledger is not None:
        ledger.check_price(model)
    return _Plan(units, names, stated, prompts, codebook_hash, ledger, model, recipe, attempts)


def _run(
    codebook: Codebook,
    codebook_hash: str,
    prompts: Mapping[str, Prompt | UnitPrompt],
    results: Sequence[CodingResult],
    started: datetime,
    recipe: Recipe,
) -> CodingRun:
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
        recipe=recipe,
    )


class _Work(NamedTuple):
    """What every call for one variable shares: the variable, its prompt and the ledger."""

    variable: Variable
    prompt: Prompt
    codebook_hash: str
    ledger: Ledger | None
    model: str | None

    @property
    def label(self) -> str:
        """The variable's name, for the ledger line."""
        return self.variable.name


class _UnitWork(NamedTuple):
    """What every call for one text shares when all its variables are asked together."""

    variables: tuple[Variable, ...]
    rules: tuple[Rule, ...]
    prompt: UnitPrompt
    codebook_hash: str
    ledger: Ledger | None
    model: str | None

    @property
    def label(self) -> str:
        """Every variable asked, joined by "+" (a variable's name cannot hold one)."""
        return "+".join(variable.name for variable in self.variables)


def _ask(
    backend: Backend,
    messages: list[dict[str, str]],
    work: _Work | _UnitWork,
    text: str,
    attempt: int,
) -> Response:
    """One call: the budget is checked first, and a call that fails is recorded before the
    error goes on."""
    ledger = work.ledger
    if ledger is not None:
        ledger.check()
    try:
        response = backend.complete(messages, json_schema=work.prompt.json_schema)
        if not isinstance(response, Response):
            raise TypeError(
                "The backend's complete() should return a reliacode Response, but it "
                f"returned {describe(response)}."
            )
    except Exception as error:
        if ledger is not None:
            ledger.record(
                LedgerEntry.from_failure(
                    error, model=work.model, text=text, ask=attempt, **_ties(work, ledger)
                )
            )
        raise
    return response


def _outcome(usable: bool, attempt: int, attempts: int) -> Outcome:
    if usable:
        return "answered"
    return "asked_again" if attempt < attempts else "missing"


class _VariableTask:
    """One text and one variable: the messages to send next, and every answer so far.

    :meth:`take` handles one answer as :func:`code` always has: it checks the answer, records
    the call in the ledger, and either ends the task or builds the second ask.
    """

    def __init__(self, work: _Work, unit: UnitId, text: str, attempts: int) -> None:
        self.work = work
        self.unit = unit
        self.text = text
        self.attempts = attempts
        self.messages = work.prompt.messages(text)
        self.ask = 1
        self.answers: list[str] = []
        self.responses: list[Response] = []
        self.done = False
        self._value: str | None = None
        self._problem: str | None = None

    def take(self, response: Response, *, record: bool = True) -> None:
        """Handle the answer to the current ask. ``record=False`` leaves the ledger alone, for
        an answer whose line is already there."""
        attempt, work = self.ask, self.work
        variable, prompt, ledger = work.variable, work.prompt, work.ledger
        self.responses.append(response)
        self.answers.append(response.text)
        value, problem, kind = _check(response.text, variable.codes)
        self._value, self._problem = value, problem
        if ledger is not None and record:
            outcome = _outcome(value is not None, attempt, self.attempts)
            ledger.record(
                LedgerEntry.from_response(
                    response, outcome=outcome, text=self.text, ask=attempt, **_ties(work, ledger)
                )
            )
        if value is not None:
            self.done = True
            return
        # Only the kind of problem: an answer can echo the coded text, which may be private.
        logger.info(
            "%s, variable %s: answer %d of %d cannot be used: %s.",
            _unit_name(self.unit),
            variable.name,
            attempt,
            self.attempts,
            kind,
        )
        retry = prompt.retry_template.format(problem=problem, codes=", ".join(variable.codes))
        self.messages = [
            *self.messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": retry},
        ]
        if attempt == self.attempts:
            self.done = True
        else:
            self.ask += 1

    def results(self) -> list[CodingResult]:
        value, problem, attempts = self._value, self._problem, self.attempts
        reason = None
        if value is None:
            reason = (
                f"The answer could not be used. {problem}"
                if attempts == 1
                else f"None of the {attempts} answers could be used. The last one: {problem}"
            )
        return [
            CodingResult(
                unit=self.unit,
                variable=self.work.variable.name,
                code=value,
                missing_reason=reason,
                answers=tuple(self.answers),
                responses=tuple(self.responses),
                codebook_hash=self.work.codebook_hash,
                prompt_hash=self.work.prompt.hash,
            )
        ]


class _UnitTask:
    """One text with all its variables asked together. The last answer decides every code."""

    def __init__(self, work: _UnitWork, unit: UnitId, text: str, attempts: int) -> None:
        self.work = work
        self.unit = unit
        self.text = text
        self.attempts = attempts
        self.messages = work.prompt.messages(text)
        self.ask = 1
        self.answers: list[str] = []
        self.responses: list[Response] = []
        self.done = False
        self._checked: _UnitAnswer | None = None

    def take(self, response: Response, *, record: bool = True) -> None:
        """Handle the answer to the current ask (see :meth:`_VariableTask.take`)."""
        attempt, work = self.ask, self.work
        prompt, ledger = work.prompt, work.ledger
        self.responses.append(response)
        self.answers.append(response.text)
        try:
            checked = _check_unit(response.text, work.variables, work.rules)
        except Exception:
            # A rule's check failed: the call was made, so it is recorded before the run stops.
            if ledger is not None and record:
                ledger.record(
                    LedgerEntry.from_response(
                        response,
                        outcome="missing",
                        text=self.text,
                        ask=attempt,
                        **_ties(work, ledger),
                    )
                )
            raise
        self._checked = checked
        if ledger is not None and record:
            outcome = _outcome(not checked.problems, attempt, self.attempts)
            ledger.record(
                LedgerEntry.from_response(
                    response, outcome=outcome, text=self.text, ask=attempt, **_ties(work, ledger)
                )
            )
        if not checked.problems:
            self.done = True
            return
        # The kinds of problem name variables and codes only, never words of the answer.
        logger.info(
            "%s: answer %d of %d cannot be used as it is: %s.",
            _unit_name(self.unit),
            attempt,
            self.attempts,
            "; ".join(checked.kinds),
        )
        if attempt < self.attempts:
            retry = prompt.retry_template.format(
                problems="\n".join(f"- {problem}" for problem in checked.retry),
                names=", ".join(prompt.variables),
            )
            self.messages = [
                *self.messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": retry},
            ]
            self.ask += 1
        else:
            self.done = True

    def results(self) -> list[CodingResult]:
        checked, attempts = self._checked, self.attempts
        assert checked is not None  # only a task that took an answer has results
        results = []
        for variable in self.work.variables:
            name = variable.name
            reason = None
            if name not in checked.codes:
                reason = (
                    f"The answer could not be used for this variable. {checked.problems[name]}"
                    if attempts == 1
                    else f"The last of the {attempts} answers could not be used for this "
                    f"variable. {checked.problems[name]}"
                )
            results.append(
                CodingResult(
                    unit=self.unit,
                    variable=name,
                    code=checked.codes.get(name),
                    missing_reason=reason,
                    answers=tuple(self.answers),
                    responses=tuple(self.responses),
                    codebook_hash=self.work.codebook_hash,
                    prompt_hash=self.work.prompt.hash,
                )
            )
        return results


def _ties(work: _Work | _UnitWork, ledger: Ledger) -> dict[str, Any]:
    """What ties a ledger line to the work: the variable (or variables), the hashes and the
    time."""
    return {
        "variable": work.label,
        "codebook_hash": work.codebook_hash,
        "prompt_hash": work.prompt.hash,
        "at": ledger.clock(),
    }


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

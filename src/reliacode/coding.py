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
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

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


class UnitPrompt(_Record):
    """The prompt that asks for several variables of one text in one call, with its hash.

    It is used by the one-call-per-unit recipe (``code(..., recipe="per_unit")``).
    ``variables`` are the variables asked, in order. The hash covers the prompt text, the
    templates, the JSON Schema and the recipe's version.
    """

    recipe: Literal["per_unit"] = "per_unit"
    recipe_version: int
    variables: tuple[str, ...]
    system: str
    user_template: str
    retry_template: str
    json_schema: dict[str, Any]
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
    codebook: Codebook, variables: Sequence[str] | str | None = None
) -> UnitPrompt:
    """The prompt that asks for all of ``codebook``'s variables (or the ones named) of a text in
    one call, built from the codebook.

    Raises:
        KeyError: the codebook has no variable with a name given.
        TypeError, ValueError: ``variables`` is not as described in :func:`code`.
    """
    names = _variable_names(codebook, variables)
    chosen = [codebook.variable(name) for name in names]
    system = _unit_system_text(codebook, chosen)
    schema = unit_answer_schema(chosen)
    payload = {
        "reliacode_prompt": UNIT_PROMPT_VERSION,
        "recipe": "per_unit",
        "system": system,
        "user": USER_TEMPLATE,
        "retry": UNIT_RETRY_TEMPLATE,
        "json_schema": schema,
        "rules": [],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return UnitPrompt(
        recipe_version=UNIT_PROMPT_VERSION,
        variables=tuple(names),
        system=system,
        user_template=USER_TEMPLATE,
        retry_template=UNIT_RETRY_TEMPLATE,
        json_schema=schema,
        hash="sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    )


def _unit_system_text(codebook: Codebook, variables: Sequence[Variable]) -> str:
    lines = [UNIT_INSTRUCTION, "", *_codebook_lines(codebook)]
    for variable in variables:
        lines += _variable_lines(variable)
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


def _check_unit(text: str, variables: Sequence[Variable]) -> _UnitAnswer:
    """Read every variable's code from an answer that should be one JSON object with one field
    per variable. The same leniency as :func:`check_answer` applies to each code."""
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
    return answer


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

    Raises:
        BackendError: the backend could not answer, for example because no server is running.
            The run stops there, since the requests that follow would fail the same way.
        BudgetError: the ledger's budget refused a call. The run stops before that call, and
            the error's ``run`` holds the results finished before the stop. A cost limit for
            a model with no price is refused before the first call, with no ``run``.
        TypeError, ValueError: ``texts``, ``variables``, ``attempts``, ``ledger`` or
            ``recipe`` are not as described.
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
    units = _units(texts)
    names = _variable_names(codebook, variables)
    prompts: dict[str, Prompt | UnitPrompt]
    if recipe == "per_unit":
        unit_prompt = build_unit_prompt(codebook, names)
        prompts = {"+".join(names): unit_prompt}
    else:
        prompts = {name: build_prompt(codebook, name) for name in names}
    codebook_hash = codebook.content_hash()
    model = getattr(backend, "model", None)
    model = model if isinstance(model, str) else None
    if ledger is not None:
        ledger.check_price(model)
    started = utc_now()
    results: list[CodingResult] = []
    try:
        for unit, text in units:
            if recipe == "per_unit":
                chosen = tuple(codebook.variable(name) for name in names)
                unit_work = _UnitWork(chosen, unit_prompt, codebook_hash, ledger, model)
                results.extend(_code_unit(backend, unit_work, unit, text, attempts))
                continue
            for name in names:
                prompt = prompts[name]
                assert isinstance(prompt, Prompt)
                work = _Work(codebook.variable(name), prompt, codebook_hash, ledger, model)
                results.append(_code_one(backend, work, unit, text, attempts))
    except BudgetError as error:
        error.run = _run(codebook, codebook_hash, prompts, results, started, recipe)
        raise
    return _run(codebook, codebook_hash, prompts, results, started, recipe)


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


def _code_one(
    backend: Backend, work: _Work, unit: UnitId, text: str, attempts: int
) -> CodingResult:
    variable, prompt, ledger = work.variable, work.prompt, work.ledger
    messages = prompt.messages(text)
    answers: list[str] = []
    responses: list[Response] = []
    problem: str | None = None
    for attempt in range(1, attempts + 1):
        response = _ask(backend, messages, work, text, attempt)
        responses.append(response)
        answers.append(response.text)
        value, problem, kind = _check(response.text, variable.codes)
        if ledger is not None:
            outcome = _outcome(value is not None, attempt, attempts)
            ledger.record(
                LedgerEntry.from_response(
                    response, outcome=outcome, text=text, ask=attempt, **_ties(work, ledger)
                )
            )
        if value is not None:
            break
        # Only the kind of problem: an answer can echo the coded text, which may be private.
        logger.info(
            "%s, variable %s: answer %d of %d cannot be used: %s.",
            _unit_name(unit),
            variable.name,
            attempt,
            attempts,
            kind,
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
        codebook_hash=work.codebook_hash,
        prompt_hash=prompt.hash,
    )


def _code_unit(
    backend: Backend, work: _UnitWork, unit: UnitId, text: str, attempts: int
) -> list[CodingResult]:
    """Every variable of one text, from one call (and a second ask if the answer cannot be used
    as it is). The last answer decides every code."""
    prompt, ledger = work.prompt, work.ledger
    messages = prompt.messages(text)
    answers: list[str] = []
    responses: list[Response] = []
    for attempt in range(1, attempts + 1):
        response = _ask(backend, messages, work, text, attempt)
        responses.append(response)
        answers.append(response.text)
        checked = _check_unit(response.text, work.variables)
        if ledger is not None:
            outcome = _outcome(not checked.problems, attempt, attempts)
            ledger.record(
                LedgerEntry.from_response(
                    response, outcome=outcome, text=text, ask=attempt, **_ties(work, ledger)
                )
            )
        if not checked.problems:
            break
        # The kinds of problem name variables and codes only, never words of the answer.
        logger.info(
            "%s: answer %d of %d cannot be used as it is: %s.",
            _unit_name(unit),
            attempt,
            attempts,
            "; ".join(checked.kinds),
        )
        if attempt < attempts:
            retry = prompt.retry_template.format(
                problems="\n".join(f"- {problem}" for problem in checked.retry),
                names=", ".join(prompt.variables),
            )
            messages = [
                *messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": retry},
            ]
    results = []
    for variable in work.variables:
        name = variable.name
        reason = None
        if name not in checked.codes:
            reason = (
                f"The answer could not be used for this variable. {checked.problems[name]}"
                if attempts == 1
                else f"The last of the {attempts} answers could not be used for this variable. "
                f"{checked.problems[name]}"
            )
        results.append(
            CodingResult(
                unit=unit,
                variable=name,
                code=checked.codes.get(name),
                missing_reason=reason,
                answers=tuple(answers),
                responses=tuple(responses),
                codebook_hash=work.codebook_hash,
                prompt_hash=prompt.hash,
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

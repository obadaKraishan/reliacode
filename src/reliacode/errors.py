"""Codebook errors written for researchers, not programmers.

Pydantic reports problems as paths such as ``variables.0.categories.2.definition`` with messages
such as "Field required". This module turns each one into a sentence that names the variable and
category involved, gives the line in the YAML file, and says how to fix it.
"""

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

#: Separates several messages raised together by one cross-field check.
RULE_SEPARATOR = "\n"

# How list items are named in messages, and which field identifies them.
_ITEM_NOUNS = {
    "variables": "variable",
    "categories": "category",
    "examples": "example",
    "decision_rules": "decision rule",
    "authors": "author",
    "references": "reference",
}
_ITEM_IDS = {"variables": "name", "categories": "code"}

_MISSING_HINTS = {
    ("codebook", "name"): "Give the codebook a short name, such as news-frames.",
    ("codebook", "variables"): 'List at least one variable to code, each starting with "- name:".',
    ("variable", "name"): (
        "Every variable needs a name, such as dominant_frame, which becomes a column in the "
        "coded data."
    ),
    ("variable", "definition"): "Every variable needs a definition of what it measures.",
    ("variable", "categories"): (
        'List the categories coders can choose from, each starting with "- code:".'
    ),
    ("category", "code"): (
        "Every category needs a short code, such as conflict or 1. The code is what gets "
        "recorded in the data."
    ),
    ("category", "definition"): (
        "Every category needs a definition that tells coders when to choose it."
    ),
    ("example", "text"): 'Give the example text, either as "text: ..." or as a plain string.',
}
_EXTRA_HINTS = {
    ("codebook", "decision_rules"): (
        "Decision rules belong to a variable. For rules that apply to the whole codebook, use "
        '"instructions".'
    ),
    ("codebook", "categories"): "Categories belong to a variable, inside its list of variables.",
    ("variable", "level"): "Levels belong to the categories of an ordinal variable.",
    ("variable", "examples"): "Examples belong to a category, not to the variable itself.",
}
_SHAPE_HINTS = {
    "variable": 'Write each variable as "- name: ..." followed by its definition and categories.',
    "category": (
        'Write each category as "- code: ..." followed by "definition: ..." on the next line.'
    ),
    "example": 'Give an example as plain text, or as "text: ..." with an optional "note: ...".',
}
_TOO_SHORT_HINTS = {
    "categories": (
        "Coders need something to choose between: for a present/absent variable, list both "
        "categories."
    ),
}
_YAML_BOOLEANS = (
    "YAML reads unquoted yes, no, on, off, true and false as true/false values. Put the value "
    'in quotes, for example "no" instead of no.'
)


@dataclass(frozen=True)
class Problem:
    """One problem in a codebook: what is wrong, where, and on which line of the file."""

    message: str
    location: str = ""
    line: int | None = None

    def __str__(self) -> str:
        where = ", ".join(p for p in (f"line {self.line}" if self.line else "", self.location) if p)
        if not where:
            return self.message
        return f"{where[0].upper()}{where[1:]}: {self.message}"


class CodebookError(ValueError):
    """A codebook could not be loaded. ``problems`` lists every problem found, in file order."""

    def __init__(self, problems: Sequence[Problem], source: str | None = None) -> None:
        self.problems = tuple(problems)
        self.source = source
        super().__init__(self._render())

    def _render(self) -> str:
        count = len(self.problems)
        where = f" in {self.source}" if self.source and self.source != "<string>" else ""
        lines = [f"The codebook{where} has {count} {'problem' if count == 1 else 'problems'}:"]
        lines += [f"  {number}. {problem}" for number, problem in enumerate(self.problems, 1)]
        return "\n".join(lines)


def describe(value: Any) -> str:
    """A short, plain description of a YAML value for use in a sentence."""
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return f"the value {str(value).lower()}"
    if isinstance(value, str):
        short = value if len(value) <= 40 else value[:37] + "..."
        return f'the text "{short}"'
    if isinstance(value, (int, float)):
        return f"the number {value}"
    if isinstance(value, Mapping):
        return "a block of fields"
    if isinstance(value, (list, tuple)):
        return "a list"
    return repr(value)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _nested_model(arg)
        if found is not None:
            return found
    return None


@dataclass
class _Place:
    parts: list[str]  # e.g. ['variable "frame"', 'category "conflict"']
    owner: type[BaseModel]  # the model that holds ``field`` (or the failing item)
    field: str | None  # trailing field name, if the error is about one field
    noun: str | None  # noun of the last list item, if the error is about a whole item


def _locate(loc: tuple[str | int, ...], root: type[BaseModel], data: Any) -> _Place:
    place = _Place(parts=[], owner=root, field=None, noun=None)
    node, list_field = data, None
    for index, part in enumerate(loc):
        if isinstance(part, str):
            if index == len(loc) - 1:
                place.field = part
                break
            info = place.owner.model_fields.get(part)
            list_field = part
            node = node.get(part) if isinstance(node, Mapping) else None
            if info is not None and isinstance(loc[index + 1], int):
                continue  # the item model is entered at the index below
            nested = _nested_model(info.annotation) if info is not None else None
            place.owner = nested or place.owner
        else:
            item = node[part] if isinstance(node, list) and 0 <= part < len(node) else None
            noun = _ITEM_NOUNS.get(list_field or "", "item")
            id_key = _ITEM_IDS.get(list_field or "")
            ident = item.get(id_key) if id_key and isinstance(item, Mapping) else None
            if isinstance(ident, (str, int)) and not isinstance(ident, bool):
                place.parts.append(f'{noun} "{ident}"')
            else:
                place.parts.append(f"{noun} {part + 1}")
            info = place.owner.model_fields.get(list_field or "")
            nested = _nested_model(info.annotation) if info is not None else None
            place.owner = nested or place.owner
            place.noun = noun
            node = item
    return place


def _line(loc: tuple[str | int, ...], lines: Mapping[tuple[str | int, ...], int]) -> int | None:
    for end in range(len(loc), 0, -1):
        line = lines.get(loc[:end])
        if line is not None:
            return line
    return None


def _messages(err: Mapping[str, Any], place: _Place) -> list[str]:
    kind, value, ctx = err["type"], err["input"], err.get("ctx") or {}
    owner = place.owner.__name__.lower()
    field = place.field
    subject = f'"{field}"' if field else f"This {place.noun or owner}"
    hint = _MISSING_HINTS.get((owner, field or ""), "")

    def sentence(*texts: str) -> str:
        return " ".join(t for t in texts if t)

    if kind == "codebook_rule":
        return err["msg"].split(RULE_SEPARATOR)
    if kind == "missing":
        return [sentence(f'The required field "{field}" is missing.', hint)]
    if kind == "extra_forbidden":
        allowed = list(place.owner.model_fields)
        match = _suggestion(field or "", place.owner)
        special = _EXTRA_HINTS.get((owner, field or ""), "")
        if match:
            advice = f'Did you mean "{match}"?'
        elif special:
            advice = special
        else:
            advice = f"The fields a {owner} can have are: {', '.join(allowed)}."
        return [sentence(f'"{field}" is not a recognized field.', advice)]
    if kind == "empty_text" or (kind == "string_type" and value is None):
        return [sentence(f"{subject} is empty.", hint)]
    if kind == "string_type" and isinstance(value, bool):
        return [sentence(f"{subject} was read as {describe(value)}, not as text.", _YAML_BOOLEANS)]
    if kind == "string_type":
        return [f"{subject} should be text, but it is {describe(value)}."]
    if kind == "string_pattern_mismatch" and field == "name":
        return [
            f'"{value}" cannot be used as a variable name. Variable names become column names in '
            "the coded data, so use only letters, digits and underscores, starting with a letter "
            "(for example dominant_frame)."
        ]
    if kind in {"tuple_type", "list_type"}:
        advice = 'Put each entry on its own line, starting with "- ".'
        return [sentence(f"{subject} should be a list, but it is {describe(value)}.", advice)]
    if kind in {"model_type", "model_attributes_type", "dict_type"}:
        advice = _SHAPE_HINTS.get(place.noun or owner, "")
        return [
            sentence(f"{subject} should be a block of fields, but it is {describe(value)}.", advice)
        ]
    if kind == "too_short":
        minimum, actual = ctx.get("min_length", 1), ctx.get("actual_length", 0)
        entries = "entry" if minimum == 1 else "entries"
        hint = _TOO_SHORT_HINTS.get(field or "", "")
        return [
            sentence(f"{subject} needs at least {minimum} {entries}, but it has {actual}.", hint)
        ]
    if kind == "literal_error":
        expected = str(ctx.get("expected", "")).replace("'", '"')
        advice = (
            "Use ordinal when the categories have a natural order (such as none, mild, strong), "
            "and nominal otherwise."
            if field == "type"
            else ""
        )
        return [sentence(f"{subject} must be {expected}, but it is {describe(value)}.", advice)]
    if kind in {"int_type", "int_parsing", "int_from_float"}:
        quoted = isinstance(value, str) and value.strip().lstrip("-").isdigit()
        return [
            sentence(
                f"{subject} should be a whole number, such as 0, 1 or 2, but it is "
                f"{describe(value)}.",
                "Remove the quotes around it." if quoted else "",
            )
        ]
    return [f"{subject}: {err['msg']}."]


def _suggestion(field: str, owner: type[BaseModel]) -> str | None:
    """The allowed field that ``field`` is most likely a misspelling of."""
    match = difflib.get_close_matches(field, list(owner.model_fields), n=1, cutoff=0.6)
    return match[0] if match else None


def _is_cascade(err: Mapping[str, Any]) -> bool:
    """Pydantic also reports "too short" when list items failed; those repeat other errors."""
    if err["type"] != "too_short":
        return False
    value, minimum = err["input"], (err.get("ctx") or {}).get("min_length", 0)
    return isinstance(value, (list, tuple)) and len(value) >= minimum


def problems_from_validation_error(
    exc: ValidationError,
    root: type[BaseModel],
    data: Any,
    lines: Mapping[tuple[str | int, ...], int] | None = None,
) -> list[Problem]:
    """Translate a pydantic ``ValidationError`` into plain-language problems in file order.

    Args:
        exc: the error raised by ``root.model_validate(data)``.
        root: the model that was validated (used to name fields and suggest spellings).
        data: the raw input, used to name variables and categories in locations.
        lines: 1-based line numbers by path, from the YAML loader.
    """
    errors = [err for err in exc.errors(include_url=False) if not _is_cascade(err)]
    places = [_locate(tuple(err["loc"]), root, data) for err in errors]
    # A misspelled field ("defintion") also makes the real one ("definition") missing. The
    # "did you mean" message says all there is to say, so the "missing" one is dropped.
    misspelled = {
        (tuple(err["loc"][:-1]), _suggestion(place.field or "", place.owner))
        for err, place in zip(errors, places, strict=True)
        if err["type"] == "extra_forbidden"
    }
    problems = []
    for err, place in zip(errors, places, strict=True):
        loc = tuple(err["loc"])
        if err["type"] == "missing" and (loc[:-1], place.field) in misspelled:
            continue
        location = " > ".join(place.parts)
        line = _line(loc, lines or {})
        problems.extend(Problem(message, location, line) for message in _messages(err, place))
    return sorted(problems, key=lambda p: (p.line is None, p.line or 0))  # stable: keeps ties

"""The codebook: variables, categories, definitions, examples and decision rules.

A codebook is written in YAML and loaded with :func:`load_codebook`. The models are immutable,
so a loaded codebook cannot drift from the file it came from.
"""

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from reliacode._yaml import YamlError, load_yaml
from reliacode.errors import (
    RULE_SEPARATOR,
    CodebookError,
    Problem,
    describe,
    problems_from_validation_error,
)

#: Version of the content-hash recipe. It is part of the hashed data, so it changes every hash.
#: Bump it whenever what the hash covers, or how it is computed, changes.
HASH_VERSION = 1


def _clean_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).strip()
    if not value:
        raise PydanticCustomError("empty_text", "must not be empty")
    return value


#: Text: surrounding whitespace is removed, Unicode is normalised to NFC, and it may not be empty.
Text = Annotated[str, AfterValidator(_clean_text)]

#: Variable names become column names in coded data.
VariableName = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
]

Level = Annotated[int, Strict()]


def _rule_error(messages: list[str]) -> PydanticCustomError:
    return PydanticCustomError("codebook_rule", RULE_SEPARATOR.join(messages))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, coerce_numbers_to_str=True)


class Example(_Model):
    """An example of text that belongs in a category, with an optional note on why."""

    text: Text
    note: Text | None = None

    @model_validator(mode="before")
    @classmethod
    def _text_shorthand(cls, data: Any) -> Any:
        # In YAML an example may be just the text: `- "The senator attacked..."`.
        return {"text": data} if isinstance(data, str) else data


class Category(_Model):
    """One value a coder can choose for a variable."""

    code: Text
    label: Text | None = None
    definition: Text
    examples: tuple[Example, ...] = ()
    level: Level | None = None


class Variable(_Model):
    """One thing coders record for every unit, such as the dominant frame of an article."""

    name: VariableName
    label: Text | None = None
    definition: Text
    type: Literal["nominal", "ordinal"] = "nominal"
    instructions: Text | None = None
    decision_rules: tuple[Text, ...] = ()
    categories: tuple[Category, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_categories(self) -> "Variable":
        messages = []
        codes = [category.code for category in self.categories]
        for code in sorted({code for code in codes if codes.count(code) > 1}):
            messages.append(
                f'The code "{code}" is used by more than one category. Each category needs its '
                "own code, because the code is what gets recorded in the data."
            )
        if self.type == "ordinal":
            missing = [c.code for c in self.categories if c.level is None]
            if missing:
                messages.append(
                    f'This variable is ordinal, so every category needs a "level": a whole number '
                    f"giving its rank, such as 0, 1, 2. Missing for: {_quoted(missing)}."
                )
            levels = [c.level for c in self.categories if c.level is not None]
            for level in sorted({level for level in levels if levels.count(level) > 1}):
                shared = [c.code for c in self.categories if c.level == level]
                messages.append(
                    f"Level {level} is given to more than one category ({_quoted(shared)}). "
                    "Each level of an ordinal variable must be different."
                )
        else:
            with_level = [c.code for c in self.categories if c.level is not None]
            if with_level:
                messages.append(
                    f"This variable is nominal, but {_quoted(with_level)} "
                    f'{"has" if len(with_level) == 1 else "have"} a "level". Levels are only '
                    'used for ordinal variables: remove them, or add "type: ordinal".'
                )
        if messages:
            raise _rule_error(messages)
        return self

    @property
    def codes(self) -> tuple[str, ...]:
        """The category codes, in codebook order."""
        return tuple(category.code for category in self.categories)


class Codebook(_Model):
    """A content-analysis codebook."""

    name: Text
    version: Text | None = None
    title: Text | None = None
    description: Text | None = None
    language: Text | None = None
    unit_of_analysis: Text | None = None
    instructions: Text | None = None
    authors: tuple[Text, ...] = ()
    references: tuple[Text, ...] = ()
    variables: tuple[Variable, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_variables(self) -> "Codebook":
        names = [variable.name for variable in self.variables]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise _rule_error(
                [
                    f'The variable name "{name}" is used more than once. Each variable needs '
                    "its own name, because the name becomes a column in the coded data."
                    for name in repeated
                ]
            )
        return self

    def variable(self, name: str) -> Variable:
        """The variable called ``name``."""
        for variable in self.variables:
            if variable.name == name:
                return variable
        known = ", ".join(v.name for v in self.variables)
        raise KeyError(f'This codebook has no variable "{name}". Its variables are: {known}.')

    def canonical_json(self) -> str:
        """The exact text that :meth:`content_hash` hashes.

        It is the validated codebook as compact JSON with sorted keys, leaving out every field
        that still has its default value, wrapped with the hash recipe version:
        ``{"codebook": {...}, "reliacode_codebook_hash": 1}``.
        """
        payload = {
            "reliacode_codebook_hash": HASH_VERSION,
            "codebook": self.model_dump(mode="json", exclude_defaults=True),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def content_hash(self) -> str:
        """A stable fingerprint of the codebook's content, as ``"sha256:<64 hex digits>"``.

        The hash depends only on what the codebook says, not on how the YAML is written:
        key order, comments, indentation, quoting, block or flow style, surrounding whitespace,
        and writing out default values all leave it unchanged. Any change to a text, code or
        level changes it, and so does reordering variables or categories, because order reaches
        the prompts a model sees. Report it with results so readers can check which codebook
        was used.
        """
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _quoted(items: list[str]) -> str:
    return ", ".join(f'"{item}"' for item in items)


def parse_codebook(text: str, source: str = "<string>") -> Codebook:
    """Read a codebook from YAML text. ``source`` names the text in error messages.

    Raises:
        CodebookError: the text is not valid YAML or not a valid codebook. The error lists every
            problem found, each with its line number and a suggested fix.
    """
    try:
        loaded = load_yaml(text)
    except YamlError as exc:
        raise CodebookError([Problem(exc.message, line=exc.line)], source) from None
    data = loaded.data
    if data is None:
        message = "The file is empty. A codebook needs at least a name and a list of variables."
        raise CodebookError([Problem(message)], source)
    if not isinstance(data, dict):
        message = (
            "A codebook should be a block of fields such as name: and variables:, but this file "
            f"holds {describe(data)}."
        )
        raise CodebookError([Problem(message, line=1)], source)
    try:
        return Codebook.model_validate(data)
    except ValidationError as exc:
        problems = problems_from_validation_error(exc, Codebook, data, loaded.lines)
        raise CodebookError(problems, source) from None


def load_codebook(path: str | os.PathLike[str]) -> Codebook:
    """Read a codebook from a YAML file (UTF-8, with or without a byte-order mark).

    Raises:
        FileNotFoundError: there is no file at ``path``.
        CodebookError: the file is not UTF-8 text, not valid YAML, or not a valid codebook.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        message = (
            "The file is not saved as UTF-8 text. Open it in a text editor and save it again "
            'with the encoding set to "UTF-8".'
        )
        raise CodebookError([Problem(message)], str(path)) from None
    return parse_codebook(text, source=str(path))

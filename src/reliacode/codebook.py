"""The codebook: variables, categories, definitions, examples and decision rules.

A codebook is written in YAML and loaded with :func:`load_codebook`. The models are immutable,
so a loaded codebook cannot drift from the file it came from.
"""

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
    model_validator,
)
from pydantic_core import PydanticCustomError

from reliacode._yaml import load_yaml

#: Separates several rule messages raised by one cross-field check.
RULE_SEPARATOR = "\n"


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


def _quoted(items: list[str]) -> str:
    return ", ".join(f'"{item}"' for item in items)


def parse_codebook(text: str, source: str = "<string>") -> Codebook:
    """Read a codebook from YAML text. ``source`` names the text in error messages."""
    loaded = load_yaml(text)
    return Codebook.model_validate(loaded.data)


def load_codebook(path: str | os.PathLike[str]) -> Codebook:
    """Read a codebook from a YAML file (UTF-8, with or without a byte-order mark)."""
    path = Path(path)
    return parse_codebook(path.read_text(encoding="utf-8-sig"), source=str(path))

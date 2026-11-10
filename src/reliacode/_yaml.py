"""Strict YAML reading for codebooks.

PyYAML's safe loader is used, with two additions that matter for hand-written codebooks:

* duplicate keys are an error (PyYAML silently keeps the last one, so a second ``definition:``
  in the same block would quietly replace the first);
* the line number of every key and list item is kept, so later validation errors can point to
  the line a researcher needs to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

YamlPath = tuple[str | int, ...]

_MERGE_TAG = "tag:yaml.org,2002:merge"


class YamlError(Exception):
    """The text could not be read as YAML. ``line`` and ``column`` are 1-based when known."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column


@dataclass(frozen=True)
class LoadedYaml:
    """Parsed YAML data plus the 1-based line of every mapping key and list item."""

    data: Any
    lines: dict[YamlPath, int] = field(default_factory=dict)

    def line_for(self, path: YamlPath) -> int | None:
        """The line of ``path``, or of its deepest part that exists in the document."""
        for end in range(len(path), -1, -1):
            line = self.lines.get(tuple(path[:end]))
            if line is not None:
                return line
        return None


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys in a mapping."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: dict[str, int] = {}
        for key_node, _value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == _MERGE_TAG:
                continue
            line = key_node.start_mark.line + 1
            if key_node.value in seen:
                raise YamlError(
                    f'The key "{key_node.value}" appears twice in the same block (lines '
                    f"{seen[key_node.value]} and {line}). YAML would keep only the second one "
                    "and silently drop the first, so remove or rename one of them.",
                    line=line,
                )
            seen[key_node.value] = line
        return super().construct_mapping(node, deep=deep)


_SYNTAX_HINTS = (
    (
        "mapping values are not allowed",
        "A common cause is a colon followed by a space inside a text value. Put the whole text "
        'in quotes, for example definition: "Conflict: disagreement between actors".',
    ),
    (
        "cannot start any token",
        "YAML does not allow tab characters for indentation. Indent with spaces instead.",
    ),
    (
        "expected <block end>",
        "Check the indentation of this line and the lines above it: entries at the same level "
        "must start in the same column.",
    ),
    (
        "could not find expected ':'",
        "A key may be missing its colon, or a quote may be left open on an earlier line.",
    ),
    (
        "found unexpected end of stream",
        "A quote or bracket is probably left open. Check that every opening quote is closed.",
    ),
)


def _syntax_message(exc: yaml.MarkedYAMLError) -> str:
    problem = (exc.problem or exc.context or "the text could not be parsed").strip()
    if "expected a single document" in (exc.context or ""):
        return (
            "The file contains more than one YAML document (separated by a line with ---). "
            "A codebook file must hold exactly one codebook."
        )
    message = f"This is not valid YAML: {problem}."
    for needle, hint in _SYNTAX_HINTS:
        if needle in problem:
            return f"{message} {hint}"
    return f"{message} Check the indentation and quoting around this line."


def _record_lines(
    node: yaml.Node, path: YamlPath, lines: dict[YamlPath, int], ancestors: frozenset[int]
) -> None:
    lines.setdefault(path, node.start_mark.line + 1)
    if id(node) in ancestors:  # a recursive alias; its lines are already recorded
        return
    ancestors = ancestors | {id(node)}
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.tag != _MERGE_TAG:
                child = (*path, key_node.value)
                lines.setdefault(child, key_node.start_mark.line + 1)
                _record_lines(value_node, child, lines, ancestors)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _record_lines(item, (*path, index), lines, ancestors)


def load_yaml(text: str) -> LoadedYaml:
    """Parse one YAML document safely, rejecting duplicate keys and recording line numbers.

    Raises:
        YamlError: the text is not valid YAML, holds several documents, uses a tag the safe
            loader does not allow, or repeats a key within one mapping.
    """
    loader = _StrictLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            return LoadedYaml(data=None)
        lines: dict[YamlPath, int] = {}
        _record_lines(node, (), lines, frozenset())
        data = loader.construct_document(node)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        line = mark.line + 1 if mark else None
        column = mark.column + 1 if mark else None
        raise YamlError(_syntax_message(exc), line=line, column=column) from None
    except yaml.YAMLError as exc:
        raise YamlError(f"This is not valid YAML: {exc}") from None
    finally:
        loader.dispose()
    return LoadedYaml(data=data, lines=lines)

"""The codebook API stays as it is, so code that already loads codebooks keeps working.

New modules may add names, but the codebook names, their signatures, their fields and the
example codebooks' content hashes do not change. A change here is a breaking change: it needs a
CHANGELOG entry, and a new HASH_VERSION if a hash changes.
"""

import dataclasses
import inspect

import pytest

import reliacode
from helpers import EXAMPLES
from reliacode import (
    Category,
    Codebook,
    CodebookError,
    Example,
    Problem,
    Variable,
    load_codebook,
    parse_codebook,
)
from reliacode.codebook import HASH_VERSION

CODEBOOK_NAMES = [
    "Category",
    "Codebook",
    "CodebookError",
    "Example",
    "Problem",
    "Variable",
    "__version__",
    "load_codebook",
    "parse_codebook",
]

#: The published content hashes of the example codebooks (see CHANGELOG.md).
EXAMPLE_HASHES = {
    "incivility.yaml": "sha256:0a56109f1da29f0242fae1edf057785de91e1f0d2474227e977764aaa2b234dd",
    "news_frames.yaml": "sha256:b4bdbacd8a7fae5da8725e4bd5c2f5f5a44187922e5ec380a2b21f2ee491bdb1",
}


def parameters(function: object) -> list[tuple[str, str, object]]:
    signature = inspect.signature(function)
    return [(p.name, p.kind.name, p.default) for p in signature.parameters.values()]


def test_every_codebook_name_is_still_exported() -> None:
    for name in CODEBOOK_NAMES:
        assert name in reliacode.__all__, name
        assert hasattr(reliacode, name), name


def test_the_loading_functions_keep_their_signatures() -> None:
    assert parameters(load_codebook) == [("path", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty)]
    assert parameters(parse_codebook) == [
        ("text", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("source", "POSITIONAL_OR_KEYWORD", "<string>"),
    ]


def test_the_models_keep_their_fields() -> None:
    assert list(Codebook.model_fields) == [
        "name",
        "version",
        "title",
        "description",
        "language",
        "unit_of_analysis",
        "instructions",
        "authors",
        "references",
        "variables",
    ]
    assert list(Variable.model_fields) == [
        "name",
        "label",
        "definition",
        "type",
        "instructions",
        "decision_rules",
        "categories",
    ]
    assert list(Category.model_fields) == ["code", "label", "definition", "examples", "level"]
    assert list(Example.model_fields) == ["text", "note"]


def test_the_codebook_methods_are_still_there() -> None:
    assert parameters(Codebook.variable) == [
        ("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("name", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ]
    assert callable(Codebook.canonical_json)
    assert callable(Codebook.content_hash)
    assert isinstance(Variable.codes, property)


def test_the_errors_keep_their_shape() -> None:
    assert issubclass(CodebookError, ValueError)
    assert [field.name for field in dataclasses.fields(Problem)] == ["message", "location", "line"]
    error = CodebookError([Problem("Something is wrong.", "variable 1", 3)], "codebook.yaml")
    assert (error.source, error.problems[0].line) == ("codebook.yaml", 3)
    assert str(error) == (
        "The codebook in codebook.yaml has 1 problem:\n  1. Line 3, variable 1: Something is wrong."
    )


def test_the_hash_recipe_version_is_unchanged() -> None:
    assert HASH_VERSION == 1


@pytest.mark.parametrize("name", sorted(EXAMPLE_HASHES))
def test_the_example_codebooks_load_with_their_published_hashes(name: str) -> None:
    codebook = load_codebook(EXAMPLES / name)
    assert codebook.content_hash() == EXAMPLE_HASHES[name]
    assert parse_codebook((EXAMPLES / name).read_text(encoding="utf-8")) == codebook

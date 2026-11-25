"""The codebook and backend API stays as it is, so code that already uses reliacode keeps
working.

New modules may add names, and functions may gain keyword arguments at the end with defaults
that keep the old behavior. The existing names, their signatures, their fields and the example
codebooks' content hashes do not change. A change here is a breaking change: it needs a
CHANGELOG entry, and a new HASH_VERSION if a hash changes.
"""

import dataclasses
import inspect
import time
from pathlib import Path

import pytest

import reliacode
from helpers import EXAMPLES
from reliacode import (
    Category,
    Codebook,
    CodebookError,
    CodingResult,
    CodingRun,
    Example,
    FakeBackend,
    Ledger,
    OpenAICompatibleBackend,
    Problem,
    RequestParameters,
    Response,
    Variable,
    code,
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

BACKEND_NAMES = [
    "Backend",
    "BackendError",
    "CodingResult",
    "CodingRun",
    "FakeBackend",
    "OpenAICompatibleBackend",
    "RequestParameters",
    "Response",
    "code",
    "Rule",
    "AnthropicBackend",
]

LEDGER_NAMES = [
    "Budget",
    "BudgetError",
    "Ledger",
    "LedgerEntry",
    "LedgerError",
    "LedgerSummary",
    "ModelPrice",
    "PriceTable",
    "read_ledger",
    "summarize",
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


# The backend and coding API ----------------------------------------------------------------------

EMPTY = inspect.Parameter.empty


@pytest.mark.parametrize("name", BACKEND_NAMES + LEDGER_NAMES)
def test_every_backend_and_ledger_name_is_exported(name: str) -> None:
    assert name in reliacode.__all__
    assert hasattr(reliacode, name)


def test_code_keeps_its_parameters_and_adds_the_ledger_at_the_end() -> None:
    assert parameters(code) == [
        ("codebook", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("texts", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("backend", "KEYWORD_ONLY", EMPTY),
        ("variables", "KEYWORD_ONLY", None),
        ("attempts", "KEYWORD_ONLY", 2),
        ("ledger", "KEYWORD_ONLY", None),
        ("recipe", "KEYWORD_ONLY", "per_variable"),
        ("rules", "KEYWORD_ONLY", None),
    ]


def test_the_backends_keep_their_parameters() -> None:
    assert parameters(OpenAICompatibleBackend.__init__) == [
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("base_url", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("model", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("api_key_env", "KEYWORD_ONLY", None),
        ("timeout", "KEYWORD_ONLY", 60.0),
        ("max_retries", "KEYWORD_ONLY", 3),
        ("temperature", "KEYWORD_ONLY", 0.0),
        ("seed", "KEYWORD_ONLY", None),
        ("max_tokens", "KEYWORD_ONLY", None),
        ("structured_output", "KEYWORD_ONLY", True),
        ("transport", "KEYWORD_ONLY", None),
        ("sleep", "KEYWORD_ONLY", time.sleep),
        ("clock", "KEYWORD_ONLY", time.perf_counter),
        ("extra_body", "KEYWORD_ONLY", None),
    ]
    assert parameters(FakeBackend.__init__) == [
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("answers", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("model", "KEYWORD_ONLY", "fake-model"),
        ("tokens", "KEYWORD_ONLY", None),
    ]


def test_the_anthropic_backend_keeps_its_parameters() -> None:
    from reliacode import AnthropicBackend

    assert parameters(AnthropicBackend.__init__) == [
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("model", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("api_key_env", "KEYWORD_ONLY", "ANTHROPIC_API_KEY"),
        ("max_tokens", "KEYWORD_ONLY", 4096),
        ("cache", "KEYWORD_ONLY", None),
        ("thinking", "KEYWORD_ONLY", None),
        ("effort", "KEYWORD_ONLY", None),
        ("structured_output", "KEYWORD_ONLY", True),
        ("temperature", "KEYWORD_ONLY", None),
        ("extra_body", "KEYWORD_ONLY", None),
        ("base_url", "KEYWORD_ONLY", "https://api.anthropic.com"),
        ("api_version", "KEYWORD_ONLY", "2026-10-21"),
        ("timeout", "KEYWORD_ONLY", 600.0),
        ("max_retries", "KEYWORD_ONLY", 3),
        ("transport", "KEYWORD_ONLY", None),
        ("sleep", "KEYWORD_ONLY", time.sleep),
        ("clock", "KEYWORD_ONLY", time.perf_counter),
    ]


def test_the_backend_and_coding_records_keep_their_fields() -> None:
    assert list(Response.model_fields) == [
        "text",
        "model",
        "parameters",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "finish_reason",
        "response_id",
        "request_id",
        "system_fingerprint",
        "latency_seconds",
        "attempts",
        "created_at",
        "reasoning_tokens",
        # Added for the native Anthropic backend, at the end.
        "cache_write_tokens",
        "cache_write_1h_tokens",
        "cache_read_tokens",
        "batch_id",
        "custom_id",
    ]
    assert list(RequestParameters.model_fields) == [
        "endpoint",
        "model",
        "temperature",
        "seed",
        "max_tokens",
        "structured_output",
        "extra_body",
        # Added for the native Anthropic backend, at the end.
        "api_version",
        "thinking",
        "effort",
        "cache",
    ]
    assert list(CodingResult.model_fields) == [
        "unit",
        "variable",
        "code",
        "missing_reason",
        "answers",
        "responses",
        "codebook_hash",
        "prompt_hash",
    ]
    assert list(CodingRun.model_fields) == [
        "codebook_name",
        "codebook_version",
        "codebook_hash",
        "prompts",
        "results",
        "started_at",
        "finished_at",
        "reliacode_version",
        "recipe",
    ]


def test_a_run_with_a_ledger_codes_exactly_as_a_run_without_one(tmp_path: Path) -> None:
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    texts = ["The council raised the bus fare.", "A storm closed the harbor for a week."]
    script = ['{"code": "economic"}', "not json", '{"code": "government"}'] * 2

    def run(ledger: Ledger | None) -> tuple[CodingRun, FakeBackend]:
        backend = FakeBackend(script)
        return code(codebook, texts, backend=backend, ledger=ledger), backend

    plain, plain_backend = run(None)
    recorded, recorded_backend = run(Ledger(tmp_path / "ledger.jsonl"))
    assert plain_backend.calls == recorded_backend.calls
    assert plain_backend.json_schemas == recorded_backend.json_schemas
    assert plain.table() == recorded.table()
    assert plain.prompts == recorded.prompts
    assert plain.codebook_hash == recorded.codebook_hash
    for without, with_ledger in zip(plain.results, recorded.results, strict=True):
        assert without.answers == with_ledger.answers
        assert without.missing_reason == with_ledger.missing_reason
        assert without.prompt_hash == with_ledger.prompt_hash
        assert [r.text for r in without.responses] == [r.text for r in with_ledger.responses]


# The reliability API -----------------------------------------------------------------------------

RELIABILITY_NAMES = [
    "CodebookReliability",
    "Codes",
    "CodesError",
    "VariableReliability",
    "codebook_reliability",
    "read_codes",
    "variable_reliability",
]


@pytest.mark.parametrize("name", RELIABILITY_NAMES)
def test_every_reliability_name_is_exported(name: str) -> None:
    assert name in reliacode.__all__
    assert hasattr(reliacode, name)


def test_the_reliability_functions_keep_their_parameters() -> None:
    assert parameters(reliacode.read_codes) == [
        ("codebook", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("source", "POSITIONAL_OR_KEYWORD", None),
        ("run", "KEYWORD_ONLY", None),
    ]
    settings = [
        ("resamples", "KEYWORD_ONLY", 2000),
        ("seed", "KEYWORD_ONLY", None),
        ("confidence", "KEYWORD_ONLY", 0.95),
    ]
    assert parameters(reliacode.variable_reliability) == [
        ("codebook", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("codes", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("variable", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("coders", "KEYWORD_ONLY", None),
        *settings,
    ]
    assert parameters(reliacode.codebook_reliability) == [
        ("codebook", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("codes", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("variables", "KEYWORD_ONLY", None),
        ("coders", "KEYWORD_ONLY", None),
        *settings,
    ]


def test_the_reliability_records_keep_their_fields() -> None:
    from reliacode.reliability import Agreement, CodedValue, Interval

    assert list(reliacode.Codes.model_fields) == ["values"]
    assert list(CodedValue.model_fields) == ["unit", "coder", "variable", "value"]
    assert list(Interval.model_fields) == ["low", "high", "confidence", "defined_resamples"]
    assert list(Agreement.model_fields) == [
        "coefficient",
        "value",
        "reason",
        "interval",
        "notes",
        "units",
        "pairable_values",
        "percent_agreement",
        "coder_counts",
        "coder_shares",
    ]
    assert list(reliacode.VariableReliability.model_fields) == [
        "variable",
        "type",
        "coders",
        "alpha",
        "kappa",
        "notes",
        "resamples",
        "seed",
        "confidence",
        "method",
        "codebook_hash",
        "reliacode_version",
    ]
    assert list(reliacode.CodebookReliability.model_fields) == [
        "codebook_name",
        "codebook_version",
        "codebook_hash",
        "coders",
        "variables",
        "resamples",
        "seed",
        "confidence",
        "method",
        "reliacode_version",
    ]
    assert issubclass(reliacode.CodesError, ValueError)


def test_the_reliability_records_are_frozen_and_refuse_unknown_fields() -> None:
    for record in (reliacode.Codes, reliacode.VariableReliability, reliacode.CodebookReliability):
        assert record.model_config["frozen"] is True
        assert record.model_config["extra"] == "forbid"

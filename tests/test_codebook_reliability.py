"""Reliability for a whole codebook, and how long it takes at the size of a typical gold set."""

import random
import re
import time

import pytest

from helpers import AGREEMENT, matrix_rows, yaml_text
from reliacode import parse_codebook
from reliacode.reliability import (
    CodebookReliability,
    codebook_reliability,
    read_codes,
    variable_reliability,
)

CODEBOOK = parse_codebook(AGREEMENT)
ROWS = matrix_rows(
    "tone",
    {
        "anna": ["positive", "neutral", "negative", "neutral", "positive", None],
        "ben": ["positive", "negative", "negative", "neutral", "neutral", "positive"],
    },
) + matrix_rows(
    "severity",
    {
        "anna": ["none", "mild", "strong", "mild", "none"],
        "ben": ["none", "strong", "strong", "mild", "mild"],
    },
)
CODES = read_codes(CODEBOOK, ROWS)


def test_every_variable_is_compared_in_codebook_order_with_one_seed() -> None:
    result = codebook_reliability(CODEBOOK, CODES, seed=9, resamples=300)
    assert isinstance(result, CodebookReliability)
    assert [r.variable for r in result.variables] == ["tone", "severity"]
    for variable in ("tone", "severity"):
        alone = variable_reliability(CODEBOOK, CODES, variable, seed=9, resamples=300)
        assert result.variable(variable) == alone
    assert (result.seed, result.resamples, result.confidence) == (9, 300, 0.95)
    assert result.method == "percentile bootstrap over units"
    assert result.coders == ("anna", "ben")
    assert result.codebook_name == "agreement-demo"
    assert result.codebook_version is None
    assert result.codebook_hash == CODEBOOK.content_hash()


def test_a_drawn_seed_is_shared_by_every_variable() -> None:
    result = codebook_reliability(CODEBOOK, CODES, resamples=100)
    assert {r.seed for r in result.variables} == {result.seed}


def test_variables_and_coders_can_be_chosen() -> None:
    one = codebook_reliability(CODEBOOK, CODES, variables="severity", seed=1, resamples=0)
    assert [r.variable for r in one.variables] == ["severity"]
    both = codebook_reliability(
        CODEBOOK, CODES, variables=["severity", "tone"], coders=["anna"], resamples=0
    )
    assert [r.variable for r in both.variables] == ["severity", "tone"]
    assert both.coders == ("anna",)
    assert both.variable("tone").alpha.value is None


def test_a_variable_without_codes_is_reported_not_skipped() -> None:
    codes = read_codes(CODEBOOK, [row for row in ROWS if row["variable"] == "tone"])
    result = codebook_reliability(CODEBOOK, codes, resamples=0)
    assert result.variable("severity").alpha.reason.startswith("No alpha: there are no values")


def test_the_table_has_one_row_per_variable() -> None:
    result = codebook_reliability(CODEBOOK, CODES, seed=9, resamples=300)
    rows = result.table()
    tone = result.variable("tone")
    assert rows[0] == {
        "variable": "tone",
        "type": "nominal",
        "coders": 2,
        "units": 5,
        "pairable_values": 10,
        "percent_agreement": 60.0,
        "alpha": tone.alpha.value,
        "alpha_low": tone.alpha.interval.low,
        "alpha_high": tone.alpha.interval.high,
        "kappa": tone.kappa.value,
        "kappa_low": tone.kappa.interval.low,
        "kappa_high": tone.kappa.interval.high,
    }
    assert rows[1]["variable"] == "severity"
    assert rows[1]["type"] == "ordinal"


def test_the_table_leaves_empty_what_is_not_there() -> None:
    three = [*ROWS, {"unit": "1", "coder": "cara", "variable": "tone", "value": "positive"}]
    result = codebook_reliability(CODEBOOK, read_codes(CODEBOOK, three), resamples=0)
    row = result.table()[0]
    assert (row["coders"], row["kappa"], row["kappa_low"], row["alpha_low"]) == (
        3,
        None,
        None,
        None,
    )
    assert row["alpha"] is not None


def test_bad_arguments_are_explained() -> None:
    with pytest.raises(TypeError, match="codebook should be a codebook"):
        codebook_reliability("codebook.yaml", CODES)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="codes should be the codes to compare"):
        codebook_reliability(CODEBOOK, ROWS)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match='no variable "tones"'):
        codebook_reliability(CODEBOOK, CODES, variables=["tones"])
    with pytest.raises(ValueError, match='"tone" is named more than once'):
        codebook_reliability(CODEBOOK, CODES, variables=["tone", "tone"])
    with pytest.raises(ValueError, match="variables is empty"):
        codebook_reliability(CODEBOOK, CODES, variables=[])
    with pytest.raises(ValueError, match="resamples should be a whole number"):
        codebook_reliability(CODEBOOK, CODES, resamples=-5)
    with pytest.raises(
        KeyError, match=re.escape('There is no result for "x". The variables are: tone,')
    ):
        codebook_reliability(CODEBOOK, CODES, resamples=0).variable("x")


# Timing ----------------------------------------------------------------------------------------


def gold_set_codebook() -> str:
    """16 variables, as in a typical gold set: three with many categories, one ordinal scale,
    and twelve present/absent flags."""
    lines = ["name: timing", "variables:"]

    def variable(name: str, codes: list[str], ordinal: bool = False) -> None:
        nonlocal lines
        lines += [f"  - name: {name}", "    definition: Invented."]
        lines += ["    type: ordinal"] if ordinal else []
        lines.append("    categories:")
        for level, code in enumerate(codes):
            lines += [f"      - code: {code}", "        definition: Invented."]
            lines += [f"        level: {level}"] if ordinal else []

    variable("topic", [f"t{n}" for n in range(24)])
    variable("second_topic", [f"t{n}" for n in range(25)])
    variable("method", [f"m{n}" for n in range(12)])
    variable("scale", ["low", "some", "much", "most"], ordinal=True)
    for flag in range(12):
        variable(f"flag{flag}", ["absent", "present"])
    return yaml_text("\n".join(lines) + "\n")


def test_a_gold_set_of_300_units_16_variables_and_2000_resamples_is_quick() -> None:
    codebook = parse_codebook(gold_set_codebook())
    assert len(codebook.variables) == 16
    rng = random.Random(3)
    rows = []
    for variable in codebook.variables:
        codes = variable.codes
        for unit in range(300):
            truth = (
                rng.choice(codes) if len(codes) > 2 else rng.choice(["absent"] * 4 + ["present"])
            )
            for coder in ("coder_1", "coder_2"):
                value = truth if rng.random() < 0.95 else rng.choice(codes)
                rows.append(
                    {"unit": f"w{unit}", "coder": coder, "variable": variable.name, "value": value}
                )
    codes = read_codes(codebook, rows)
    start = time.perf_counter()
    result = codebook_reliability(codebook, codes, seed=20261124)
    seconds = time.perf_counter() - start
    print(f"\n300 units, 2 coders, 16 variables, 2,000 resamples: {seconds:.1f} s")
    assert all(r.alpha.interval.defined_resamples == 2000 for r in result.variables)
    assert all(r.kappa is not None and r.kappa.interval is not None for r in result.variables)
    # A generous bound, so a slow CI machine passes; locally it takes a few seconds.
    assert seconds < 120

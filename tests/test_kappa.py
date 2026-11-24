"""Cohen's kappa for two coders: published worked examples, scikit-learn's values, an exact
oracle from the definition, and symmetry in the coders.

Published examples, read from the sources:

- SAS Institute (2014). SAS/STAT 13.2 User's Guide, The FREQ Procedure, "Agreement Study":
  two dermatologists rate the skin condition of 88 patients; simple kappa 0.3449.
- Bland, M. (2015). An Introduction to Medical Statistics (4th ed.), section 20.3, Table 20.5:
  94 children's answers on a self-administered questionnaire against an interview (Bland,
  Bewley, Banks and Pollard, 1975, Health Education Journal 34, 71-78); kappa 0.801.
  Table 20.6 (artificial data): kappa 0.37, 0.00 and 0.00 with 80%, 80% and 68% agreement.
"""

import json
import random
from fractions import Fraction

import pytest

from helpers import AGREEMENT, FIXTURES, matrix_rows, one_variable
from reliacode import parse_codebook
from reliacode.reliability import read_codes, variable_reliability


def cross_table(codes: list[str], table: list[list[int]]) -> list[dict[str, object]]:
    """Rows for two coders from a cross-table of counts: rows are the first coder's codes,
    columns the second's."""
    rows = []
    unit = 0
    for a, counts in zip(codes, table, strict=True):
        for b, count in zip(codes, counts, strict=True):
            for _ in range(count):
                unit += 1
                rows += [
                    {"unit": f"u{unit}", "coder": "first", "variable": "v", "value": a},
                    {"unit": f"u{unit}", "coder": "second", "variable": "v", "value": b},
                ]
    return rows


def kappa_of(codes: list[str], rows: list[dict[str, object]]):
    codebook = parse_codebook(one_variable(codes))
    return variable_reliability(codebook, read_codes(codebook, rows), "v", resamples=0).kappa


def test_the_sas_agreement_study() -> None:
    codes = ["terrible", "poor", "marginal", "clear"]
    table = [[10, 4, 1, 0], [5, 10, 12, 2], [2, 4, 12, 5], [0, 2, 6, 13]]
    kappa = kappa_of(codes, cross_table(codes, table))
    assert round(kappa.value, 4) == 0.3449  # as printed by SAS
    assert kappa.value == pytest.approx((88 * 45 - 1968) / (88**2 - 1968), abs=1e-15)
    assert (kappa.units, kappa.pairable_values) == (88, 176)
    assert kappa.percent_agreement == pytest.approx(100 * 45 / 88)
    assert kappa.coder_counts == {
        "first": {"terrible": 15, "poor": 29, "marginal": 23, "clear": 21},
        "second": {"terrible": 17, "poor": 20, "marginal": 31, "clear": 20},
    }
    assert kappa.coefficient == "cohen_kappa"


def test_bland_table_20_5_questionnaire_against_interview() -> None:
    kappa = kappa_of(["yes", "no"], cross_table(["yes", "no"], [[61, 2], [6, 25]]))
    assert round(kappa.value, 3) == 0.801  # as reported by Bland
    assert round(kappa.percent_agreement / 100, 3) == 0.915
    assert kappa.units == 94


@pytest.mark.parametrize(
    ("table", "printed", "exact", "percent"),
    [
        ([[10, 10], [10, 70]], 0.37, 0.375, 80),
        ([[0, 20], [0, 80]], 0.00, 0.0, 80),
        ([[4, 16], [16, 64]], 0.00, 0.0, 68),
    ],
)
def test_bland_table_20_6_high_agreement_with_kappa_near_zero(
    table: list[list[int]], printed: float, exact: float, percent: float
) -> None:
    kappa = kappa_of(["yes", "no"], cross_table(["yes", "no"], table))
    assert kappa.value == pytest.approx(exact, abs=1e-15)
    assert abs(kappa.value - printed) <= 0.005 + 1e-12  # Bland prints two decimals
    assert kappa.percent_agreement == percent


REFERENCE = json.loads((FIXTURES / "reliability_reference.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("number", range(12))
def test_kappa_matches_scikit_learn(number: int) -> None:
    table = REFERENCE["kappa"][number]
    rows = matrix_rows("v", {"first": table["first"], "second": table["second"]})
    assert kappa_of(table["codes"], rows).value == pytest.approx(table["kappa"], abs=1e-9)


def kappa_by_definition(first: list[object], second: list[object]) -> Fraction | None:
    """(p_o - p_e) / (1 - p_e) on the units both coded, computed exactly."""
    both = [(a, b) for a, b in zip(first, second, strict=True) if a is not None and b is not None]
    if not both:
        return None
    n = len(both)
    observed = Fraction(sum(a == b for a, b in both), n)
    codes = {a for a, _ in both} | {b for _, b in both}
    chance = sum(
        Fraction(sum(a == c for a, _ in both), n) * Fraction(sum(b == c for _, b in both), n)
        for c in codes
    )
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


def invented_coder(
    rng: random.Random, truth: list[str], codes: list[str], skill: float, missing: float
) -> list[object]:
    return [
        None if rng.random() < missing else (code if rng.random() < skill else rng.choice(codes))
        for code in truth
    ]


@pytest.mark.parametrize("seed", range(4))
def test_kappa_equals_the_definition_on_invented_tables_with_missing_values(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        codes = [f"c{n}" for n in range(rng.randint(2, 6))]
        units = rng.randint(1, 40)
        skill, missing = rng.random(), rng.random() * 0.4
        truth = [rng.choice(codes) for _ in range(units)]

        first = invented_coder(rng, truth, codes, skill, missing)
        second = invented_coder(rng, truth, codes, skill, missing)
        exact = kappa_by_definition(first, second)
        kappa = kappa_of(codes, matrix_rows("v", {"first": first, "second": second}))
        if exact is None:
            assert kappa.value is None
        else:
            assert kappa.value == pytest.approx(float(exact), abs=1e-12)


def test_kappa_is_the_same_with_the_coders_swapped() -> None:
    codes = ["terrible", "poor", "marginal", "clear"]
    rows = cross_table(codes, [[10, 4, 1, 0], [5, 10, 12, 2], [2, 4, 12, 5], [0, 2, 6, 13]])
    swapped = [{**row, "coder": {"first": "b", "second": "a"}[row["coder"]]} for row in rows]
    assert kappa_of(codes, swapped).value == pytest.approx(kappa_of(codes, rows).value, abs=1e-15)


def test_kappa_uses_only_the_units_both_coded() -> None:
    rows = matrix_rows("v", {"first": ["a", "b", "a", None], "second": ["a", "b", None, "b"]})
    kappa = kappa_of(["a", "b"], rows)
    assert (kappa.units, kappa.value, kappa.percent_agreement) == (2, 1.0, 100.0)
    assert kappa.coder_counts == {"first": {"a": 1, "b": 1}, "second": {"a": 1, "b": 1}}


def test_kappa_and_alpha_use_the_same_units_with_two_coders() -> None:
    codebook = parse_codebook(AGREEMENT)
    rows = matrix_rows(
        "tone",
        {
            "anna": ["positive", "neutral", "negative", None, "positive"],
            "ben": ["positive", "negative", "negative", "neutral", "positive"],
        },
    )
    result = variable_reliability(codebook, read_codes(codebook, rows), "tone")
    assert result.alpha.units == result.kappa.units == 4
    assert result.alpha.percent_agreement == result.kappa.percent_agreement == 75.0
    assert result.notes == ()


def test_with_more_than_two_coders_there_is_no_kappa_and_the_result_says_why() -> None:
    codebook = parse_codebook(one_variable(["a", "b"]))
    rows = matrix_rows("v", {"x": ["a", "b"], "y": ["a", "b"], "z": ["b", "b"]})
    codes = read_codes(codebook, rows)
    result = variable_reliability(codebook, codes, "v")
    assert result.kappa is None
    assert result.notes == (
        "No Cohen's kappa: it compares exactly two coders, and these codes have 3 (x, y, z). "
        'Choose two with coders=["...", "..."] to compare a pair.',
    )
    assert variable_reliability(codebook, codes, "v", coders=["x", "z"]).kappa.value == 0.0
    one = variable_reliability(codebook, codes, "v", coders=["x"])
    assert "these codes have 1 (x)" in one.notes[0]

"""The percentile bootstrap over units: reproducible with a seed, recorded in the result,
independent of the order of the rows, and with the coverage it claims."""

import random
from fractions import Fraction

import pytest

from helpers import AGREEMENT, matrix_rows
from reliacode import parse_codebook
from reliacode.reliability import METHOD, read_codes, variable_reliability

CODEBOOK = parse_codebook(AGREEMENT)
TONES = ["positive", "neutral", "negative"]


def invented(seed: int, units: int = 40, skill: float = 0.75) -> dict[str, list[object]]:
    """Two coders who give a unit's own tone with probability ``skill``, else any tone."""
    rng = random.Random(seed)
    truth = [rng.choice(TONES) for _ in range(units)]

    def coder() -> list[object]:
        return [tone if rng.random() < skill else rng.choice(TONES) for tone in truth]

    return {"anna": coder(), "ben": coder()}


CODES = read_codes(CODEBOOK, matrix_rows("tone", invented(5)))


def test_the_same_seed_gives_the_same_intervals() -> None:
    first = variable_reliability(CODEBOOK, CODES, "tone", seed=11, resamples=300)
    second = variable_reliability(CODEBOOK, CODES, "tone", seed=11, resamples=300)
    assert first == second
    other = variable_reliability(CODEBOOK, CODES, "tone", seed=12, resamples=300)
    assert other.alpha.interval != first.alpha.interval
    assert other.alpha.value == first.alpha.value


def test_the_intervals_are_pinned_for_a_seed_on_every_python_version() -> None:
    # The draws come from random.Random(seed).random(), whose sequence Python keeps the same
    # across versions, so these numbers hold on every supported Python.
    result = variable_reliability(CODEBOOK, CODES, "tone", seed=20261124, resamples=500)
    assert result.alpha.value == pytest.approx(0.46075085324232085, abs=1e-12)
    assert result.alpha.interval.low == pytest.approx(0.23261350180446316, abs=1e-12)
    assert result.alpha.interval.high == pytest.approx(0.6910475078775086, abs=1e-12)
    assert result.kappa.value == pytest.approx(0.45419103313840153, abs=1e-12)
    assert result.kappa.interval.low == pytest.approx(0.22702020202020212, abs=1e-12)
    assert result.kappa.interval.high == pytest.approx(0.6879646270146912, abs=1e-12)
    assert result.alpha.interval.defined_resamples == 500


def test_the_settings_are_recorded() -> None:
    result = variable_reliability(CODEBOOK, CODES, "tone", seed=3, resamples=200, confidence=0.9)
    assert (result.seed, result.resamples, result.confidence) == (3, 200, 0.9)
    assert result.method == METHOD == "percentile bootstrap over units"
    assert result.alpha.interval.confidence == 0.9


def test_without_a_seed_one_is_drawn_and_recorded_so_the_interval_can_be_repeated() -> None:
    first = variable_reliability(CODEBOOK, CODES, "tone", resamples=200)
    again = variable_reliability(CODEBOOK, CODES, "tone", resamples=200, seed=first.seed)
    assert again.alpha.interval == first.alpha.interval
    assert again.kappa.interval == first.kappa.interval
    assert 0 <= first.seed < 2**32


def test_the_default_is_2000_resamples_at_95_percent() -> None:
    result = variable_reliability(CODEBOOK, CODES, "tone", seed=1)
    assert (result.resamples, result.confidence) == (2000, 0.95)
    assert result.alpha.interval.defined_resamples == 2000


def test_the_order_of_the_rows_and_the_names_of_the_coders_do_not_change_the_interval() -> None:
    rows = matrix_rows("tone", invented(5))
    first = variable_reliability(CODEBOOK, read_codes(CODEBOOK, rows), "tone", seed=4)
    shuffled = rows[:]
    random.Random(2).shuffle(shuffled)
    renamed = [{**row, "coder": {"anna": "zoe", "ben": "adam"}[row["coder"]]} for row in shuffled]
    second = variable_reliability(CODEBOOK, read_codes(CODEBOOK, renamed), "tone", seed=4)
    assert second.alpha.interval == first.alpha.interval
    assert second.kappa.interval.low == pytest.approx(first.kappa.interval.low, abs=1e-15)
    assert second.kappa.interval.high == pytest.approx(first.kappa.interval.high, abs=1e-15)


def test_units_that_cannot_be_paired_do_not_change_the_interval() -> None:
    rows = matrix_rows("tone", invented(5))
    first = variable_reliability(CODEBOOK, read_codes(CODEBOOK, rows), "tone", seed=4)
    lone = [
        {"unit": f"x{n}", "coder": "anna", "variable": "tone", "value": "neutral"} for n in range(9)
    ]
    second = variable_reliability(CODEBOOK, read_codes(CODEBOOK, rows + lone), "tone", seed=4)
    assert second.alpha == first.alpha
    assert second.kappa == first.kappa


def test_a_higher_confidence_gives_a_wider_interval() -> None:
    narrow = variable_reliability(CODEBOOK, CODES, "tone", seed=8, confidence=0.8).alpha.interval
    wide = variable_reliability(CODEBOOK, CODES, "tone", seed=8, confidence=0.99).alpha.interval
    assert wide.low < narrow.low < narrow.high < wide.high


def test_with_no_resamples_there_is_no_interval_and_the_result_says_so() -> None:
    result = variable_reliability(CODEBOOK, CODES, "tone", resamples=0)
    assert result.alpha.interval is None
    assert result.alpha.notes == ("No interval: resamples is 0.",)
    assert result.kappa.notes == ("No interval: resamples is 0.",)
    assert result.alpha.value is not None


def test_more_than_two_coders_get_an_alpha_interval() -> None:
    three = invented(6)
    three["cara"] = [None if n % 4 == 0 else tone for n, tone in enumerate(three["anna"])]
    result = variable_reliability(
        CODEBOOK, read_codes(CODEBOOK, matrix_rows("tone", three)), "tone", seed=2
    )
    assert result.kappa is None
    assert result.alpha.interval.low < result.alpha.value < result.alpha.interval.high


def test_resamples_with_one_code_only_are_counted_and_left_out() -> None:
    # Only unit 10 holds "negative", and about a third of the resamples miss it.
    matrix = {"anna": ["positive"] * 9 + ["negative"], "ben": ["positive"] * 8 + ["neutral"] * 2}
    result = variable_reliability(
        CODEBOOK, read_codes(CODEBOOK, matrix_rows("tone", matrix)), "tone", seed=1
    )
    interval = result.alpha.interval
    assert 0 < interval.defined_resamples < 2000
    left_out = 2000 - interval.defined_resamples
    assert result.alpha.notes == (
        f"{left_out:,} of the 2,000 resamples hold one code only, so alpha is not defined in "
        f"them. The interval comes from the other {interval.defined_resamples:,}.",
    )


def test_no_interval_when_fewer_than_half_of_the_resamples_give_a_value() -> None:
    matrix = {"anna": ["positive", "negative"], "ben": ["positive", "negative"]}
    codes = read_codes(CODEBOOK, matrix_rows("tone", matrix))
    result = variable_reliability(CODEBOOK, codes, "tone", seed=2, resamples=9)
    assert result.alpha.interval is None
    assert result.alpha.notes == (
        "No interval: alpha is defined in only 2 of the 9 resamples, fewer than half. The "
        "others hold one code only.",
        "Every pair of values agrees, so alpha is 1.",
    )
    assert result.kappa.notes[0].startswith("No interval: kappa is defined in only 2 of the 9")


@pytest.mark.parametrize(
    ("settings", "error", "message"),
    [
        ({"resamples": -1}, ValueError, "resamples should be a whole number, 0 or more"),
        ({"resamples": 2.5}, ValueError, "resamples should be a whole number"),
        ({"resamples": True}, ValueError, "resamples should be a whole number"),
        ({"seed": "abc"}, TypeError, "seed should be a whole number"),
        ({"seed": False}, TypeError, "seed should be a whole number"),
        ({"confidence": 95}, ValueError, r"but it is 95\. For a 95% interval, write 0\.95\."),
        ({"confidence": 0}, ValueError, "confidence should be a share between 0 and 1"),
        ({"confidence": 1.0}, ValueError, "confidence should be a share between 0 and 1"),
        ({"confidence": True}, ValueError, "confidence should be a share between 0 and 1"),
    ],
)
def test_bad_settings_are_explained(
    settings: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        variable_reliability(CODEBOOK, CODES, "tone", **settings)  # type: ignore[arg-type]


# Coverage --------------------------------------------------------------------------------------


def population_value(prevalence: list[Fraction], skill: Fraction) -> Fraction:
    """Alpha and kappa in the population (they are equal when both coders work alike).

    Each unit's true code c has probability prevalence[c]. Each coder gives the true code with
    probability ``skill``, and otherwise any of the K codes at random.
    """
    k = len(prevalence)

    def given(c: int, truth: int) -> Fraction:
        return skill * (c == truth) + (1 - skill) / k

    agree = sum(prevalence[t] * sum(given(c, t) ** 2 for c in range(k)) for t in range(k))
    share = [sum(prevalence[t] * given(c, t) for t in range(k)) for c in range(k)]
    chance = sum(s * s for s in share)
    return (agree - chance) / (1 - chance)


def test_the_90_percent_intervals_cover_the_population_value_about_90_percent_of_the_time() -> None:
    # 600 samples of 60 units, each with a 90% interval from 300 resamples. The share that
    # covers the population value has a standard error of about 0.012, so it must fall within
    # 0.04 of 0.90. (Larger runs, 2,000 samples of 100 units, give about 0.895.)
    prevalence = [Fraction(1, 2), Fraction(3, 10), Fraction(1, 5)]
    skill = Fraction(7, 10)
    target = float(population_value(prevalence, skill))
    rng = random.Random(1)
    covered = {"alpha": 0, "kappa": 0}
    samples = 600
    for sample in range(samples):
        truth = rng.choices(range(3), weights=[float(p) for p in prevalence], k=60)

        def coder(truth: list[int] = truth) -> list[object]:
            return [TONES[t] if rng.random() < skill else rng.choice(TONES) for t in truth]

        codes = read_codes(CODEBOOK, matrix_rows("tone", {"anna": coder(), "ben": coder()}))
        result = variable_reliability(
            CODEBOOK, codes, "tone", seed=sample, resamples=300, confidence=0.9
        )
        for name in covered:
            interval = getattr(result, name).interval
            covered[name] += interval.low <= target <= interval.high
    for name, count in covered.items():
        assert abs(count / samples - 0.90) <= 0.04, (name, count / samples)


def test_the_population_value_is_computed_right() -> None:
    # With every coder perfect, alpha and kappa are 1; with coders at random, they are 0.
    prevalence = [Fraction(1, 2), Fraction(1, 2)]
    assert population_value(prevalence, Fraction(1)) == 1
    assert population_value(prevalence, Fraction(0)) == 0

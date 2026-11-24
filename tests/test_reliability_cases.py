"""Undefined and degenerate cases are said in plain words, never hidden as 0 or 1."""

from helpers import AGREEMENT, matrix_rows
from reliacode import parse_codebook
from reliacode.reliability import read_codes, variable_reliability

CODEBOOK = parse_codebook(AGREEMENT)


def tone(matrix: dict[str, list[object]], **settings: object):
    codes = read_codes(CODEBOOK, matrix_rows("tone", matrix))
    return variable_reliability(CODEBOOK, codes, "tone", seed=1, **settings)  # type: ignore[arg-type]


def test_a_variable_without_values_has_no_alpha_and_says_why() -> None:
    codes = read_codes(CODEBOOK, matrix_rows("tone", {"anna": ["positive"], "ben": ["neutral"]}))
    result = variable_reliability(CODEBOOK, codes, "severity", seed=1)
    assert result.alpha.value is None
    assert result.alpha.reason == (
        "No alpha: there are no values for this variable. None was given, or every value is "
        "missing."
    )
    assert result.alpha.interval is None
    assert (result.alpha.units, result.alpha.pairable_values) == (0, 0)
    assert result.alpha.percent_agreement is None
    assert result.coders == ()
    assert result.kappa is None
    assert result.notes == (
        "No Cohen's kappa: it compares exactly two coders, and these codes have none. Choose "
        'two with coders=["...", "..."] to compare a pair.',
    )


def test_every_value_missing_is_no_values_not_a_category() -> None:
    result = tone({"anna": [None, None], "ben": [None, None]})
    assert result.alpha.value is None
    assert result.alpha.reason.startswith("No alpha: there are no values for this variable.")
    assert result.kappa.value is None
    assert result.kappa.reason == "No kappa: no unit has a code from both coders."


def test_no_unit_coded_twice_gives_no_pairs() -> None:
    result = tone({"anna": ["positive", None], "ben": [None, "negative"]})
    assert result.alpha.value is None
    assert result.alpha.reason == (
        "No alpha: no unit has values from two or more coders, so there are no pairs of values "
        "to compare."
    )
    assert result.alpha.notes == ()
    assert result.kappa.value is None
    assert result.kappa.reason == "No kappa: no unit has a code from both coders."
    assert result.kappa.coder_counts == {}


def test_no_variation_leaves_alpha_and_kappa_undefined_never_0_or_1() -> None:
    result = tone({"anna": ["positive"] * 5, "ben": ["positive"] * 5})
    assert result.alpha.value is None
    assert result.alpha.reason == (
        'No alpha: every pairable value is "positive", so there is no variation. Alpha is '
        "undefined here (the disagreement expected by chance is zero), not 0 or 1."
    )
    assert result.kappa.value is None
    assert result.kappa.reason == (
        'No kappa: both coders gave "positive" to every unit they both coded, so the agreement '
        "expected by chance is 1 and kappa is undefined, not 0 or 1."
    )
    assert result.alpha.interval is None
    assert result.kappa.interval is None
    # The figures beside it still show what happened.
    assert result.alpha.percent_agreement == 100.0
    assert result.alpha.coder_shares["anna"] == {"positive": 1.0, "neutral": 0.0, "negative": 0.0}


def test_every_pair_agreeing_is_said_and_the_collapsed_interval_explained() -> None:
    matrix = {
        "anna": ["positive", "neutral", "negative"] * 4,
        "ben": ["positive", "neutral", "negative"] * 4,
    }
    result = tone(matrix)
    assert result.alpha.value == 1.0
    assert (result.alpha.interval.low, result.alpha.interval.high) == (1.0, 1.0)
    assert result.alpha.notes == (
        "Every pair of values agrees, so alpha is 1. Every resample gives 1 too, so the interval "
        "collapses to the point [1, 1] and understates the uncertainty: it shows only that no "
        "disagreement was seen among these 12 units, not that alpha would be 1 on others.",
    )
    assert result.kappa.value == 1.0
    assert result.kappa.notes == (
        "The two coders agree on every unit, so kappa is 1. Every resample gives 1 too, so the "
        "interval collapses to the point [1, 1] and understates the uncertainty: it shows only "
        "that no disagreement was seen among these 12 units, not that kappa would be 1 on "
        "others.",
    )


def test_every_pair_agreeing_is_said_even_without_an_interval() -> None:
    matrix = {"anna": ["positive", "negative"], "ben": ["positive", "negative"]}
    result = tone(matrix, resamples=0)
    assert result.alpha.notes == (
        "No interval: resamples is 0.",
        "Every pair of values agrees, so alpha is 1.",
    )


def test_any_other_collapsed_interval_is_said() -> None:
    result = tone({"anna": ["positive"], "ben": ["negative"]})
    assert result.alpha.value == 0.0
    assert (result.alpha.interval.low, result.alpha.interval.high) == (0.0, 0.0)
    assert result.alpha.notes == (
        "Every resample gives the same alpha, 0, so the interval collapses to a point and "
        "understates the uncertainty.",
    )
    assert result.kappa.notes == (
        "Every resample gives the same kappa, 0, so the interval collapses to a point and "
        "understates the uncertainty.",
    )


def test_a_figure_below_zero_is_reported_as_it_is() -> None:
    # Every unit holds one "positive" and one "negative": systematic disagreement.
    result = tone({"anna": ["positive", "negative"] * 5, "ben": ["negative", "positive"] * 5})
    assert result.alpha.value == (19 * 0 - 180) / (20 * 19 - 180) == -0.9
    assert result.kappa.value == (10 * 0 - 50) / (10**2 - 50) == -1.0
    # Every resample holds the same kind of unit, so the interval is a point, and says so.
    assert result.alpha.notes[0].startswith("Every resample gives the same alpha, -0.9,")


def test_the_kappa_of_an_ordinal_variable_is_said_to_be_unweighted() -> None:
    matrix = {
        "anna": ["none", "mild", "strong", "mild"],
        "ben": ["none", "strong", "strong", "mild"],
    }
    codes = read_codes(CODEBOOK, matrix_rows("severity", matrix))
    result = variable_reliability(CODEBOOK, codes, "severity", seed=1)
    assert result.type == "ordinal"
    assert result.kappa.notes[-1] == (
        "This kappa is unweighted: it treats the categories as unordered, so a near miss counts "
        "as much as a far one. Alpha uses their order."
    )
    assert all("unweighted" not in note for note in result.alpha.notes)

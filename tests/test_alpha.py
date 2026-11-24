"""Krippendorff's alpha: the worked examples of Krippendorff (2011), an exact implementation
straight from the definition as an oracle, and invariance to the order of units and the names
of coders.

Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. Annenberg School for
Communication, University of Pennsylvania (version 2011.1.25, literature updated 2013.9.13).
The data matrices below are copied from its sections A, B and C.
"""

import random
import re
from collections import Counter
from fractions import Fraction

import pytest

from helpers import matrix_rows, one_variable
from reliacode import parse_codebook
from reliacode.reliability import Codes, read_codes, variable_reliability

_ = None  # a missing value, as "." in the paper


def alpha_of(matrix: dict[str, list[object]], codes: list[str], *, ordinal: bool = False):
    codebook = parse_codebook(one_variable(codes, ordinal=ordinal))
    return variable_reliability(codebook, read_codes(codebook, matrix_rows("v", matrix)), "v")


# Krippendorff (2011), section A: binary data, two observers, no missing data.
SECTION_A = {
    "Meg": ["0", "1", "0", "0", "0", "0", "0", "0", "1", "0"],
    "Owen": ["1", "1", "1", "0", "0", "1", "0", "0", "0", "0"],
}
# Section B: nominal data, two observers, no missing data.
SECTION_B = {
    "Ben": ["a", "a", "b", "b", "d", "c", "c", "c", "e", "d", "d", "a"],
    "Gerry": ["b", "a", "b", "b", "b", "c", "c", "c", "e", "d", "d", "d"],
}
# Section C: four observers, 12 units, 7 of the 48 values missing.
SECTION_C = {
    "A": ["1", "2", "3", "3", "2", "1", "4", "1", "2", _, _, _],
    "B": ["1", "2", "3", "3", "2", "2", "4", "1", "2", "5", _, "3"],
    "C": [_, "3", "3", "3", "2", "3", "4", "2", "2", "5", "1", _],
    "D": ["1", "2", "3", "3", "2", "4", "4", "1", "2", "5", "1", _],
}


def test_section_a_binary_two_observers() -> None:
    result = alpha_of(SECTION_A, ["0", "1"])
    assert round(result.alpha.value, 3) == 0.095  # as printed in the paper
    assert result.alpha.value == pytest.approx(1 - 19 * 4 / (14 * 6), abs=1e-15)
    assert (result.alpha.units, result.alpha.pairable_values) == (10, 20)
    assert result.alpha.percent_agreement == 60.0


def test_section_b_nominal_two_observers() -> None:
    result = alpha_of(SECTION_B, ["a", "b", "c", "d", "e"])
    assert round(result.alpha.value, 3) == 0.692
    assert result.alpha.value == pytest.approx(310 / 448, abs=1e-15)
    assert (result.alpha.units, result.alpha.pairable_values) == (12, 24)
    assert result.alpha.percent_agreement == 75.0


def test_section_c_nominal_four_observers_with_missing_data() -> None:
    result = alpha_of(SECTION_C, ["1", "2", "3", "4", "5"])
    assert round(result.alpha.value, 3) == 0.743
    # (40 - 1)(7 + 10 + 8 + 4 + 3) - 344 over 40(40 - 1) - 344, as worked in the paper.
    assert result.alpha.value == pytest.approx((39 * 32 - 344) / (40 * 39 - 344), abs=1e-15)
    # Unit 12 holds one value, which cannot be paired: 11 units and 40 pairable values.
    assert (result.alpha.units, result.alpha.pairable_values) == (11, 40)
    # 43 of the 55 pairs of values within units agree.
    assert result.alpha.percent_agreement == pytest.approx(100 * 43 / 55)
    assert result.coders == ("A", "B", "C", "D")
    assert result.alpha.coder_counts["A"] == {"1": 3, "2": 3, "3": 2, "4": 1, "5": 0}
    assert result.alpha.coder_counts["C"] == {"1": 1, "2": 3, "3": 4, "4": 1, "5": 1}
    assert result.alpha.coder_shares["A"]["1"] == pytest.approx(3 / 9)
    assert sum(result.alpha.coder_shares["D"].values()) == pytest.approx(1)


def test_section_c_as_ordinal_data() -> None:
    result = alpha_of(SECTION_C, ["1", "2", "3", "4", "5"], ordinal=True)
    assert round(result.alpha.value, 3) == 0.815
    exact = alpha_by_definition(SECTION_C, ["1", "2", "3", "4", "5"], ordinal=True)
    assert result.alpha.value == pytest.approx(float(exact), abs=1e-15)
    assert result.type == "ordinal"


def test_an_unused_rank_does_not_change_ordinal_alpha() -> None:
    # The paper adds an unused rank to show that it does not matter.
    five = alpha_of(SECTION_C, ["1", "2", "3", "4", "5"], ordinal=True).alpha.value
    six = alpha_of(SECTION_C, ["1", "2", "3", "4", "unused", "5"], ordinal=True).alpha.value
    assert six == pytest.approx(five, abs=1e-15)


def test_ordinal_ranks_come_from_the_levels_not_the_order_of_the_categories() -> None:
    in_order = one_variable(["1", "2", "3", "4", "5"], ordinal=True)
    shuffled = parse_codebook(
        in_order.replace('code: "5"', 'code: "x5"')
        .replace('code: "2"', 'code: "5"')
        .replace('code: "x5"', 'code: "2"')
        .replace("level: 1", "level: 9")
        .replace("level: 4", "level: 1")
        .replace("level: 9", "level: 4")
    )
    assert shuffled.variable("v").codes == ("1", "5", "3", "4", "2")
    rows = matrix_rows("v", SECTION_C)
    result = variable_reliability(shuffled, read_codes(shuffled, rows), "v")
    assert result.alpha.value == pytest.approx(
        alpha_of(SECTION_C, ["1", "2", "3", "4", "5"], ordinal=True).alpha.value, abs=1e-15
    )
    # Counts and shares list the codes in codebook order.
    assert list(result.alpha.coder_counts["A"]) == ["1", "5", "3", "4", "2"]


def test_nominal_and_ordinal_alpha_differ_when_the_order_matters() -> None:
    near = {"A": ["0", "1", "2", "3"] * 3, "B": ["1", "2", "3", "2"] * 3}
    nominal = alpha_of(near, ["0", "1", "2", "3"]).alpha.value
    ordinal = alpha_of(near, ["0", "1", "2", "3"], ordinal=True).alpha.value
    assert ordinal > nominal


def test_coders_can_be_chosen() -> None:
    codebook = parse_codebook(one_variable(["1", "2", "3", "4", "5"]))
    codes = read_codes(codebook, matrix_rows("v", SECTION_C))
    pair = variable_reliability(codebook, codes, "v", coders=["B", "A"])
    alone = alpha_of({"A": SECTION_C["A"], "B": SECTION_C["B"]}, ["1", "2", "3", "4", "5"])
    assert pair.coders == ("A", "B")
    assert pair.alpha == alone.alpha


# The oracle: alpha straight from its definition --------------------------------------------------


def alpha_by_definition(
    matrix: dict[str, list[object]], order: list[str], *, ordinal: bool = False
) -> Fraction | None:
    """Alpha = 1 - D_o / D_e, computed exactly and slowly from the pairs of values.

    D_o is the mean difference over the ordered pairs of values within units, each unit's pairs
    weighted by 1 / (m_u - 1). D_e is the mean difference over every ordered pair of distinct
    pairable values, regardless of unit. No coincidence matrix is used.
    """
    units = [
        [coder_values[number] for coder_values in matrix.values()]
        for number in range(len(next(iter(matrix.values()))))
    ]
    units = [[value for value in unit if value is not None] for unit in units]
    units = [unit for unit in units if len(unit) >= 2]
    pooled = [value for unit in units for value in unit]
    n = len(pooled)
    if n < 2 or len(set(pooled)) < 2:
        return None
    frequency = Counter(pooled)
    rank = {code: number for number, code in enumerate(order)}

    def difference(c: object, k: object) -> Fraction:
        if c == k:
            return Fraction(0)
        if not ordinal:
            return Fraction(1)
        low, high = sorted((rank[c], rank[k]))
        between = sum(frequency[order[g]] for g in range(low, high + 1))
        return (between - Fraction(frequency[c] + frequency[k], 2)) ** 2

    observed = Fraction(0)
    for unit in units:
        within = sum(
            difference(a, b) for i, a in enumerate(unit) for j, b in enumerate(unit) if i != j
        )
        observed += within / (len(unit) - 1)
    observed /= n
    expected = sum(
        difference(a, b) for i, a in enumerate(pooled) for j, b in enumerate(pooled) if i != j
    ) / (n * (n - 1))
    return 1 - observed / expected


def invented_table(rng: random.Random, order: list[str]) -> dict[str, list[object]]:
    """Codes from 2 to 6 coders on 1 to 25 units: each coder usually gives the unit's own code,
    sometimes another, and sometimes nothing."""
    coders = [f"coder{c}" for c in range(rng.randint(2, 6))]
    units = rng.randint(1, 25)
    skill, missing = rng.random(), rng.random() * 0.5
    truth = [rng.choice(order) for _ in range(units)]
    return {
        coder: [
            None
            if rng.random() < missing
            else (truth[u] if rng.random() < skill else rng.choice(order))
            for u in range(units)
        ]
        for coder in coders
    }


@pytest.mark.parametrize("ordinal", [False, True], ids=["nominal", "ordinal"])
@pytest.mark.parametrize("seed", range(5))
def test_the_fast_alpha_equals_the_definition_on_invented_tables(seed: int, ordinal: bool) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        order = [f"c{n}" for n in range(rng.randint(2, 7))]
        matrix = invented_table(rng, order)
        exact = alpha_by_definition(matrix, order, ordinal=ordinal)
        fast = alpha_of(matrix, order, ordinal=ordinal).alpha.value
        if exact is None:
            assert fast is None
        else:
            assert fast == pytest.approx(float(exact), abs=1e-12)


def test_the_oracle_gives_the_paper_values() -> None:
    assert round(float(alpha_by_definition(SECTION_C, ["1", "2", "3", "4", "5"])), 3) == 0.743
    assert round(float(alpha_by_definition(SECTION_B, ["a", "b", "c", "d", "e"])), 3) == 0.692
    ordinal = alpha_by_definition(SECTION_C, ["1", "2", "3", "4", "5"], ordinal=True)
    assert round(float(ordinal), 3) == 0.815


# Invariance ------------------------------------------------------------------------------------


def test_alpha_does_not_depend_on_the_order_of_the_codes_or_the_coders_names() -> None:
    codebook = parse_codebook(one_variable(["1", "2", "3", "4", "5"]))
    rows = matrix_rows("v", SECTION_C)
    first = variable_reliability(codebook, read_codes(codebook, rows), "v").alpha
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    renamed = [
        {**row, "coder": {"A": "zed", "B": "x", "C": "b", "D": "a"}[row["coder"]]}
        for row in shuffled
    ]
    second = variable_reliability(codebook, read_codes(codebook, renamed), "v").alpha
    assert second.value == first.value
    assert (second.units, second.pairable_values) == (first.units, first.pairable_values)
    assert second.percent_agreement == first.percent_agreement
    assert second.coder_counts["zed"] == first.coder_counts["A"]


def test_alpha_does_not_depend_on_the_unit_ids() -> None:
    codebook = parse_codebook(one_variable(["1", "2", "3", "4", "5"]))
    rows = matrix_rows("v", SECTION_C)
    renamed = [{**row, "unit": f"w{99 - int(row['unit'])}"} for row in rows]
    first = variable_reliability(codebook, read_codes(codebook, rows), "v").alpha
    second = variable_reliability(codebook, read_codes(codebook, renamed), "v").alpha
    assert second.value == pytest.approx(first.value, abs=1e-15)


# Inputs ----------------------------------------------------------------------------------------


def test_missing_values_are_left_out_never_counted_as_a_code() -> None:
    with_gaps = alpha_of({"A": ["a", None, "b"], "B": ["a", "b", None]}, ["a", "b"])
    assert with_gaps.alpha.units == 1
    assert with_gaps.alpha.pairable_values == 2


def test_values_outside_the_codebook_are_refused_even_in_codes_built_by_hand() -> None:
    codebook = parse_codebook(one_variable(["a", "b"]))
    codes = Codes.model_validate(
        {"values": [{"unit": "1", "coder": "A", "variable": "v", "value": "c"}]}
    )
    with pytest.raises(ValueError, match=re.escape('"c" is not one of the codes of v: a, b.')):
        variable_reliability(codebook, codes, "v")


def test_bad_arguments_are_explained() -> None:
    codebook = parse_codebook(one_variable(["a", "b"]))
    codes = read_codes(codebook, matrix_rows("v", {"A": ["a"], "B": ["b"]}))
    with pytest.raises(TypeError, match="codebook should be a codebook"):
        variable_reliability("codebook.yaml", codes, "v")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="codes should be the codes to compare"):
        variable_reliability(codebook, codes.rows(), "v")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match='no variable "w"'):
        variable_reliability(codebook, codes, "w")
    with pytest.raises(
        ValueError, match=re.escape('no codes from the coder "C". The coders are: A, B.')
    ):
        variable_reliability(codebook, codes, "v", coders=["A", "C"])
    with pytest.raises(ValueError, match='"A" is named more than once'):
        variable_reliability(codebook, codes, "v", coders=["A", "A"])
    with pytest.raises(ValueError, match="coders is empty"):
        variable_reliability(codebook, codes, "v", coders=[])
    assert variable_reliability(codebook, codes, "v", coders="A").coders == ("A",)

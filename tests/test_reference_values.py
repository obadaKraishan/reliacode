"""reliacode against other implementations, on fixed invented tables.

The values were computed once by the ``krippendorff`` package (alpha) and scikit-learn's
``cohen_kappa_score`` (kappa), by ``tests/reference/make_reliability_reference.py``, and are kept
in ``tests/fixtures/reliability_reference.json`` with the versions used. reliacode depends on
neither package.
"""

import json

import pytest

from helpers import FIXTURES, matrix_rows, one_variable
from reliacode import parse_codebook
from reliacode.reliability import read_codes, variable_reliability

REFERENCE = json.loads((FIXTURES / "reliability_reference.json").read_text(encoding="utf-8"))


def test_the_reference_file_says_how_it_was_made() -> None:
    assert set(REFERENCE["made_with"]) == {"python", "krippendorff", "numpy", "scikit-learn"}
    assert len(REFERENCE["alpha"]) == 24
    assert {table["level"] for table in REFERENCE["alpha"]} == {"nominal", "ordinal"}


@pytest.mark.parametrize("number", range(24))
def test_alpha_matches_the_krippendorff_package(number: int) -> None:
    table = REFERENCE["alpha"][number]
    codebook = parse_codebook(one_variable(table["codes"], ordinal=table["level"] == "ordinal"))
    codes = read_codes(codebook, matrix_rows("v", table["matrix"]))
    alpha = variable_reliability(codebook, codes, "v", resamples=0).alpha.value
    assert alpha == pytest.approx(table["alpha"], abs=1e-9)

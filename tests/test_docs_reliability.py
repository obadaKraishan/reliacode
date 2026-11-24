"""The example in docs/reliability.md runs, and prints what the page says it prints."""

import contextlib
import io
import itertools
import re
import shutil
from pathlib import Path

import pytest

from helpers import EXAMPLES
from reliacode import load_codebook
from reliacode.reliability import CodesError, read_codes, variable_reliability

DOC = (Path(__file__).parent.parent / "docs" / "reliability.md").read_text(encoding="utf-8")
BLOCKS = re.findall(r"```(\w+)\n(.*?)```", DOC, flags=re.DOTALL)
CSV = next(body for language, body in BLOCKS if language == "csv")


def flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def example_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "examples" / "codebooks").mkdir(parents=True)
    shutil.copy(EXAMPLES / "incivility.yaml", tmp_path / "examples" / "codebooks")
    (tmp_path / "human_codes.csv").write_text(CSV, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_each_example_prints_what_the_page_shows(example_folder: Path) -> None:
    namespace: dict[str, object] = {}
    checked = 0
    for (language, body), (next_language, shown) in itertools.pairwise(BLOCKS):
        if language != "python" or next_language != "text":
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(body, "docs/reliability.md", "exec"), namespace)
        assert printed.getvalue() == shown
        checked += 1
    assert checked == 2


def test_the_figures_in_the_text_are_right(example_folder: Path) -> None:
    codebook = load_codebook("examples/codebooks/incivility.yaml")
    codes = read_codes(codebook, "human_codes.csv")
    severity = variable_reliability(codebook, codes, "severity", resamples=0)
    shares = severity.alpha.coder_shares
    assert (round(100 * shares["anna"]["none"]), round(100 * shares["ben"]["none"])) == (43, 29)
    assert "anna\ncoded 43% of the comments as `none`, while ben coded 29%" in DOC
    assert severity.alpha.units == 7
    assert "unweighted" in severity.kappa.notes[-1]


def test_the_error_shown_on_the_page_is_the_real_one(example_folder: Path) -> None:
    codebook = load_codebook("examples/codebooks/incivility.yaml")
    wrong = example_folder / "wrong.csv"
    wrong.write_text(CSV.replace("c1,ben,severity,none", "c1,ben,severity,Mild"), encoding="utf-8")
    with pytest.raises(CodesError) as error:
        read_codes(codebook, wrong)
    assert flat(str(error.value.problems[0])) in flat(DOC)


def test_the_undefined_cases_on_the_page_are_the_real_messages(example_folder: Path) -> None:
    codebook = load_codebook("examples/codebooks/incivility.yaml")
    rows = [
        {"unit": f"c{n}", "coder": coder, "variable": "severity", "value": "none"}
        for n in range(3)
        for coder in ("anna", "ben")
    ]
    result = variable_reliability(codebook, read_codes(codebook, rows), "severity", seed=1)
    assert f"`{result.alpha.reason}`" in DOC
    assert f"`{result.kappa.reason}`" in DOC

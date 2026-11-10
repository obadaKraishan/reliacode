"""The example codebooks in examples/codebooks/ load, and show the features they are meant to."""

import pytest

from helpers import EXAMPLES
from reliacode import load_codebook

EXAMPLE_FILES = sorted(EXAMPLES.glob("*.yaml"))


def test_both_examples_are_present() -> None:
    assert [path.name for path in EXAMPLE_FILES] == ["incivility.yaml", "news_frames.yaml"]


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_example_is_complete(path) -> None:
    codebook = load_codebook(path)
    assert codebook.references, "examples must cite the published source they adapt"
    assert codebook.unit_of_analysis
    for variable in codebook.variables:
        assert variable.decision_rules, f"{variable.name} has no decision rules"
        for category in variable.categories:
            assert category.examples, f"{variable.name}/{category.code} has no example"


def test_news_frames() -> None:
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    frame = codebook.variable("dominant_frame")
    assert frame.type == "nominal"
    assert frame.codes == (
        "conflict",
        "human_interest",
        "economic",
        "morality",
        "responsibility",
        "none",
    )
    assert codebook.variable("responsibility_target").codes == (
        "government",
        "non_government",
        "not_applicable",
    )


def test_incivility_has_an_ordinal_severity_scale() -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    assert len(codebook.variable("incivility_form").categories) == 6
    severity = codebook.variable("severity")
    assert severity.type == "ordinal"
    assert [(c.code, c.level) for c in severity.categories] == [
        ("none", 0),
        ("mild", 1),
        ("strong", 2),
    ]


def test_examples_have_distinct_hashes() -> None:
    hashes = {load_codebook(path).content_hash() for path in EXAMPLE_FILES}
    assert len(hashes) == len(EXAMPLE_FILES)

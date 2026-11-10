import pytest

from helpers import yaml_text
from reliacode.codebook import Codebook, Example, load_codebook, parse_codebook


def test_minimal_codebook_loads(minimal_yaml: str) -> None:
    codebook = parse_codebook(minimal_yaml)
    assert codebook.name == "tone-demo"
    assert codebook.version is None
    (tone,) = codebook.variables
    assert tone.type == "nominal"
    assert tone.codes == ("positive", "negative")
    assert tone.categories[0].examples == ()
    assert tone.decision_rules == ()


def test_full_codebook_loads() -> None:
    codebook = parse_codebook(
        yaml_text(
            """
            name: severity-demo
            version: "1.0"
            title: Severity demo
            description: A demo codebook.
            language: en
            unit_of_analysis: one comment
            instructions: Read the whole comment before coding.
            authors: [A. Researcher]
            references: ["Doe, J. (2020). A codebook. Journal, 1(1), 1-2."]
            variables:
              - name: severity
                label: Severity
                definition: How strongly uncivil the comment is.
                type: ordinal
                instructions: Code the most severe sentence.
                decision_rules:
                  - If unsure between two levels, choose the lower one.
                categories:
                  - code: none
                    label: None
                    definition: No uncivil language.
                    level: 0
                    examples:
                      - Thanks for the clear explanation.
                  - code: strong
                    definition: Insults aimed at a person.
                    level: 2
                    examples:
                      - text: You are an idiot.
                        note: A direct insult.
                  - code: mild
                    definition: Rude but not insulting.
                    level: 1
            """
        )
    )
    severity = codebook.variable("severity")
    assert severity.type == "ordinal"
    assert [c.level for c in severity.categories] == [0, 2, 1]
    assert severity.decision_rules == ("If unsure between two levels, choose the lower one.",)
    assert severity.categories[0].examples == (Example(text="Thanks for the clear explanation."),)
    assert severity.categories[1].examples[0].note == "A direct insult."
    assert codebook.authors == ("A. Researcher",)


def test_text_is_stripped_and_unicode_normalised() -> None:
    decomposed = "Café"  # "Café" written as "e" + combining accent (NFD)
    codebook = parse_codebook(
        yaml_text(
            f"""
            name: "  spaced  "
            variables:
              - name: tone
                definition: |
                  Tone of the text.
                categories:
                  - code: a
                    definition: "{decomposed}"
                  - code: b
                    definition: Other.
            """
        )
    )
    assert codebook.name == "spaced"
    assert codebook.variables[0].definition == "Tone of the text."
    assert codebook.variables[0].categories[0].definition == "Café"


def test_numeric_codes_become_text() -> None:
    codebook = parse_codebook(
        yaml_text(
            """
            name: numeric
            version: 2
            variables:
              - name: present
                definition: Whether the feature is present.
                categories:
                  - {code: 0, definition: Absent.}
                  - {code: 1, definition: Present.}
            """
        )
    )
    assert codebook.version == "2"
    assert codebook.variables[0].codes == ("0", "1")


def test_arabic_text_is_kept() -> None:
    codebook = parse_codebook(
        yaml_text(
            """
            name: arabic-demo
            language: ar
            variables:
              - name: tone
                definition: نبرة النص تجاه موضوعه الرئيسي.
                categories:
                  - {code: positive, label: إيجابي, definition: يمدح النص موضوعه.}
                  - {code: negative, label: سلبي, definition: ينتقد النص موضوعه.}
            """
        )
    )
    assert codebook.variables[0].categories[1].label == "سلبي"


def test_codebook_is_immutable(minimal_yaml: str) -> None:
    codebook = parse_codebook(minimal_yaml)
    with pytest.raises(Exception, match="frozen"):
        codebook.name = "changed"  # type: ignore[misc]


def test_variable_lookup_names_the_known_variables(minimal_yaml: str) -> None:
    codebook = parse_codebook(minimal_yaml)
    assert codebook.variable("tone").name == "tone"
    with pytest.raises(KeyError, match="Its variables are: tone"):
        codebook.variable("frame")


def test_load_codebook_reads_a_file_with_a_byte_order_mark(tmp_path, minimal_yaml: str) -> None:
    path = tmp_path / "tone.yaml"
    path.write_text("﻿" + minimal_yaml, encoding="utf-8")
    assert load_codebook(path) == parse_codebook(minimal_yaml)


def test_models_can_be_built_in_python() -> None:
    codebook = Codebook.model_validate(
        {
            "name": "py",
            "variables": [
                {
                    "name": "x",
                    "definition": "X.",
                    "categories": [
                        {"code": "a", "definition": "A."},
                        {"code": "b", "definition": "B."},
                    ],
                }
            ],
        }
    )
    assert codebook.variables[0].codes == ("a", "b")

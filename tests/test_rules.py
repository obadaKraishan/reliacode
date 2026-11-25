"""Rules across variables, given by the caller: stated in the prompt, checked in each answer, and
a broken rule makes the variables it names missing while the others keep their codes.

Every codebook, text and answer here is invented.
"""

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import yaml_text
from reliacode import FakeBackend, Ledger, Rule, code, parse_codebook
from reliacode.coding import build_unit_prompt

CODEBOOK = parse_codebook(
    yaml_text(
        """
        name: topic-demo
        variables:
          - name: primary_topic
            definition: The main topic of the article.
            categories:
              - {code: transport, definition: Buses and trains.}
              - {code: schools, definition: Schools and pupils.}
              - {code: parks, definition: Parks and gardens.}
          - name: secondary_topic
            definition: A second topic, if the article has one.
            categories:
              - {code: none, definition: No second topic.}
              - {code: transport, definition: Buses and trains.}
              - {code: schools, definition: Schools and pupils.}
              - {code: parks, definition: Parks and gardens.}
          - name: tone
            definition: The tone toward the main topic.
            categories:
              - {code: positive, definition: Praise.}
              - {code: negative, definition: Criticism.}
        """
    )
)
NOT_THE_SAME = Rule(
    text="The secondary topic is never the same as the primary topic.",
    variables=["primary_topic", "secondary_topic"],
    check=lambda codes: codes["secondary_topic"] != codes["primary_topic"],
)
TEXT = "The new tram line will stop outside two schools."


def answer(primary: str, secondary: str, tone: str = "positive") -> str:
    return json.dumps({"primary_topic": primary, "secondary_topic": secondary, "tone": tone})


def run_with(script: list[str], rules: list[Rule] | None = None, **settings: object):
    backend = FakeBackend(script)
    run = code(
        CODEBOOK,
        [TEXT],
        backend=backend,
        recipe="per_unit",
        rules=[NOT_THE_SAME] if rules is None else rules,
        **settings,  # type: ignore[arg-type]
    )
    return run, backend


# The rule itself ---------------------------------------------------------------------------------


def test_a_rule_keeps_its_sentence_and_variables() -> None:
    rule = Rule(
        text="  Severity is none\n if and only if form is none. ",
        variables=["a", "b"],
        check=lambda codes: True,
    )
    assert rule.text == "Severity is none if and only if form is none."
    assert rule.variables == ("a", "b")
    assert "check" not in rule.model_dump()
    assert "check" not in repr(rule)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"text": "  ", "variables": ["a"]}, "give the rule as a sentence"),
        ({"text": "A rule.", "variables": []}, "name the variables the rule is about"),
        ({"text": "A rule.", "variables": ["a", "a"]}, '"a" is named more than once'),
        ({"text": "A rule.", "variables": ["a"], "check": "not a function"}, "callable"),
        ({"text": "A rule.", "variables": "a"}, "tuple"),
    ],
)
def test_a_wrong_rule_is_explained(fields: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Rule(**{"check": lambda codes: True, **fields})


# The rules in the prompt -------------------------------------------------------------------------


def test_the_rules_are_stated_in_the_prompt_after_the_variables() -> None:
    other = Rule(
        text="A text with no second topic is coded none.",
        variables=["secondary_topic"],
        check=lambda codes: True,
    )
    prompt = build_unit_prompt(CODEBOOK, rules=[NOT_THE_SAME, other])
    section = (
        "\n\nRules across variables (the codes must follow every one):\n"
        "- The secondary topic is never the same as the primary topic.\n"
        "- A text with no second topic is coded none.\n\n"
        "Answer with only a JSON object with one field for each variable, in this form:\n"
    )
    assert section in prompt.system
    assert prompt.system.index("Variable: tone") < prompt.system.index("Rules across variables")
    assert prompt.rules == {
        NOT_THE_SAME.text: ("primary_topic", "secondary_topic"),
        other.text: ("secondary_topic",),
    }


def test_without_rules_there_is_no_rules_section() -> None:
    prompt = build_unit_prompt(CODEBOOK)
    assert "Rules across variables" not in prompt.system
    assert prompt.rules == {} == build_unit_prompt(CODEBOOK, rules=[]).rules
    assert build_unit_prompt(CODEBOOK, rules=[]).hash == prompt.hash


def test_the_hash_covers_the_sentences_and_the_variables_but_not_the_check() -> None:
    base = build_unit_prompt(CODEBOOK, rules=[NOT_THE_SAME]).hash
    assert base != build_unit_prompt(CODEBOOK).hash
    other_check = NOT_THE_SAME.model_copy(update={"check": lambda codes: False})
    assert build_unit_prompt(CODEBOOK, rules=[other_check]).hash == base
    other_variables = Rule(
        text=NOT_THE_SAME.text,
        variables=["secondary_topic", "primary_topic"],
        check=NOT_THE_SAME.check,
    )
    assert build_unit_prompt(CODEBOOK, rules=[other_variables]).hash != base
    other_text = NOT_THE_SAME.model_copy(update={"text": "The topics differ."})
    assert build_unit_prompt(CODEBOOK, rules=[other_text]).hash != base


# Rules that cannot be used are refused before any call -------------------------------------------


@pytest.mark.parametrize(
    ("settings", "error", "match"),
    [
        ({"rules": [NOT_THE_SAME]}, ValueError, 'use recipe="per_unit" with them'),
        ({"recipe": "per_unit", "rules": NOT_THE_SAME}, TypeError, "rules should be a list"),
        ({"recipe": "per_unit", "rules": "no"}, TypeError, "rules should be a list"),
        ({"recipe": "per_unit", "rules": [NOT_THE_SAME, "x"]}, TypeError, "Rule 2 should be"),
        (
            {"recipe": "per_unit", "rules": [NOT_THE_SAME, NOT_THE_SAME]},
            ValueError,
            "Rule 2 has the same sentence as rule 1",
        ),
        (
            {"recipe": "per_unit", "rules": [NOT_THE_SAME], "variables": ["primary_topic", "tone"]},
            ValueError,
            'names "secondary_topic", which is not among the variables coded',
        ),
    ],
)
def test_rules_that_cannot_be_used_are_refused_before_any_call(settings, error, match) -> None:
    backend = FakeBackend([answer("transport", "schools")])
    with pytest.raises(error, match=match):
        code(CODEBOOK, [TEXT], backend=backend, **settings)
    assert backend.calls == []


def test_an_empty_list_of_rules_is_fine_with_either_recipe() -> None:
    backend = FakeBackend(['{"code": "schools"}', '{"code": "none"}', '{"code": "positive"}'])
    assert code(CODEBOOK, [TEXT], backend=backend, rules=[]).missing() == ()


# The rules in the answer -------------------------------------------------------------------------


def test_an_answer_that_follows_every_rule_is_kept() -> None:
    run, backend = run_with([answer("transport", "schools")])
    assert run.table() == [
        {"unit": 0, "primary_topic": "transport", "secondary_topic": "schools", "tone": "positive"}
    ]
    assert len(backend.calls) == 1


def test_a_broken_rule_is_asked_again_with_its_sentence() -> None:
    run, backend = run_with([answer("transport", "transport"), answer("transport", "schools")])
    assert run.missing() == ()
    assert backend.calls[1][-1]["content"] == (
        "That answer cannot be used as it is:\n"
        "- The codes break this rule: The secondary topic is never the same as the primary "
        "topic.\n"
        "Answer again with only a JSON object with one field for each variable (primary_topic, "
        "secondary_topic, tone), each holding one of that variable's codes. Write nothing else."
    )


def test_a_rule_still_broken_makes_its_variables_missing_and_keeps_the_others(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    run, _ = run_with([answer("parks", "parks", "negative")] * 2, ledger=ledger)
    primary, secondary, tone = run.results
    assert (primary.code, secondary.code, tone.code) == (None, None, "negative")
    reason = (
        "The last of the 2 answers could not be used for this variable. The codes break this "
        "rule: The secondary topic is never the same as the primary topic."
    )
    assert primary.missing_reason == secondary.missing_reason == reason
    assert tone.missing_reason is None
    assert [entry.outcome for entry in ledger.entries()] == ["asked_again", "missing"]


def test_a_rule_is_checked_only_when_all_its_variables_have_a_valid_code() -> None:
    seen: list[dict[str, str]] = []

    def check(codes: dict[str, str]) -> bool:
        seen.append(dict(codes))
        return True

    rule = Rule(text="Any rule.", variables=["primary_topic", "secondary_topic"], check=check)
    run, backend = run_with([answer("harbor", "schools"), answer("parks", "schools")], [rule])
    assert seen == [{"primary_topic": "parks", "secondary_topic": "schools"}]
    retry = backend.calls[1][-1]["content"]
    assert "break this rule" not in retry
    assert '- The code for "primary_topic" is not one of its codes' in retry
    assert run.missing() == ()


def test_the_check_gets_only_the_codes_of_the_variables_it_names() -> None:
    seen: list[dict[str, str]] = []

    def check(codes: dict[str, str]) -> bool:
        seen.append(codes)
        codes["tone"] = "changed"  # a check cannot change the codes it is given
        return True

    rule = Rule(text="Any rule.", variables=["tone"], check=check)
    run, _ = run_with([answer("transport", "none", "positive")], [rule])
    assert seen == [{"tone": "changed"}]
    assert run.table()[0]["tone"] == "positive"


def test_two_broken_rules_on_one_variable_are_both_given() -> None:
    no_parks = Rule(
        text="Parks are never a secondary topic.",
        variables=["secondary_topic"],
        check=lambda codes: codes["secondary_topic"] != "parks",
    )
    run, backend = run_with([answer("parks", "parks")] * 2, [NOT_THE_SAME, no_parks])
    retry_lines = backend.calls[1][-1]["content"].splitlines()[1:-1]
    assert retry_lines == [
        "- The codes break this rule: The secondary topic is never the same as the primary topic.",
        "- The codes break this rule: Parks are never a secondary topic.",
    ]
    secondary = run.results[1]
    assert secondary.missing_reason is not None
    assert secondary.missing_reason.endswith(
        "The codes break this rule: The secondary topic is never the same as the primary topic. "
        "The codes break this rule: Parks are never a secondary topic."
    )
    assert run.results[0].code is None and run.results[2].code == "positive"


def test_a_broken_rule_is_logged_by_its_number(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="reliacode")
    run_with([answer("schools", "schools"), answer("schools", "none")])
    assert caplog.messages == ["Text 0: answer 1 of 2 cannot be used as it is: rule 1 is broken."]


def test_a_check_that_fails_stops_the_run_after_the_call_is_recorded(tmp_path: Path) -> None:
    rule = Rule(
        text="A rule.", variables=["tone"], check=lambda codes: codes["primary_topic"] == ""
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(
        ValueError, match=r'The check of rule 1 \("A rule."\) failed with KeyError'
    ) as caught:
        run_with([answer("transport", "none")], [rule], ledger=ledger)
    assert '{"tone": "..."}' in str(caught.value)
    assert isinstance(caught.value.__cause__, KeyError)
    assert [entry.outcome for entry in ledger.entries()] == ["missing"]


def test_a_check_must_return_true_or_false() -> None:
    rule = Rule(text="A rule.", variables=["tone"], check=lambda codes: "yes")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match=r"should return True or False, but it returned"):
        run_with([answer("transport", "none")], [rule])


def test_the_severity_rule_of_the_incivility_codebook() -> None:
    from helpers import EXAMPLES
    from reliacode import load_codebook

    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    rule = Rule(
        text="Severity is none if and only if incivility_form is none.",
        variables=["incivility_form", "severity"],
        check=lambda codes: (codes["severity"] == "none") == (codes["incivility_form"] == "none"),
    )
    script = [
        json.dumps({"incivility_form": "none", "severity": "mild"}),
        json.dumps({"incivility_form": "aspersion", "severity": "mild"}),
    ]
    run = code(
        codebook,
        ["This plan is the silliest idea of the year."],
        backend=FakeBackend(script),
        recipe="per_unit",
        rules=[rule],
    )
    assert run.table() == [{"unit": 0, "incivility_form": "aspersion", "severity": "mild"}]


def test_a_run_with_rules_survives_a_json_round_trip() -> None:
    from reliacode import CodingRun
    from reliacode.coding import UnitPrompt

    run, _ = run_with([answer("transport", "schools")])
    again = CodingRun.model_validate_json(run.model_dump_json())
    assert again == run
    prompt = again.prompts[0]
    assert isinstance(prompt, UnitPrompt)
    assert prompt.rules == {NOT_THE_SAME.text: ("primary_topic", "secondary_topic")}
    assert "check" not in run.model_dump_json()  # a rule's check is never saved

"""Invalid codebooks: every problem is reported in plain language, with its line."""

import pytest

from helpers import MINIMAL, yaml_text
from reliacode import CodebookError, Problem, load_codebook, parse_codebook


def problems_of(text: str) -> tuple[Problem, ...]:
    with pytest.raises(CodebookError) as info:
        parse_codebook(text)
    return info.value.problems


def only_problem(text: str) -> Problem:
    problems = problems_of(text)
    assert len(problems) == 1, [str(p) for p in problems]
    return problems[0]


def variable_with(categories: str, extra: str = "") -> str:
    """A one-variable codebook; ``categories`` and ``extra`` are YAML lines under the variable."""
    head = yaml_text(
        """
        name: demo
        variables:
          - name: tone
            definition: The tone of the text.
        """
    )
    indent = " " * 4
    body = "".join(f"{indent}{line}\n" for line in yaml_text(extra).splitlines())
    body += f"{indent}categories:\n"
    body += "".join(f"{indent}  {line}\n" for line in yaml_text(categories).splitlines())
    return head + body


TWO_CATEGORIES = """
- {code: positive, definition: Praises the subject.}
- {code: negative, definition: Criticises the subject.}
"""


# --------------------------------------------------------------------- the error object


def test_codebook_error_is_a_value_error_listing_numbered_problems() -> None:
    with pytest.raises(ValueError) as info:
        parse_codebook("name: demo\n", source="demo.yaml")
    error = info.value
    assert isinstance(error, CodebookError)
    assert error.source == "demo.yaml"
    assert str(error).startswith("The codebook in demo.yaml has 1 problem:\n  1. ")


def test_problem_text_puts_line_and_location_first() -> None:
    assert str(Problem("Fix it.", 'variable "tone"', 7)) == 'Line 7, variable "tone": Fix it.'
    assert str(Problem("Fix it.")) == "Fix it."


def test_all_problems_are_reported_together_in_file_order() -> None:
    text = yaml_text(
        """
        name: demo
        variables:
          - name: tone
            definition: The tone.
            categories:
              - {code: positive}
              - {code: negative, definition: ""}
          - name: frame
            categories:
              - {code: a, definition: A.}
              - {code: b, definition: B.}
        """
    )
    with pytest.raises(CodebookError) as info:
        parse_codebook(text)
    lines = [p.line for p in info.value.problems]
    assert lines == [6, 7, 8]
    assert "has 3 problems" in str(info.value)


# --------------------------------------------------------------------- file-level problems


def test_yaml_syntax_error_is_reported_with_its_line() -> None:
    problem = only_problem("name: demo\nvariables:\n  - name: tone\n   label: x\n")
    assert problem.line == 4  # the misindented "label:" line
    assert problem.message.startswith("This is not valid YAML")


def test_duplicate_key_is_reported() -> None:
    problem = only_problem(
        variable_with(
            """
            - code: positive
              definition: Praises the subject.
              definition: Supports the subject.
            - {code: negative, definition: Criticises the subject.}
            """
        )
    )
    assert '"definition" appears twice' in problem.message
    assert problem.line == 8


def test_empty_file() -> None:
    assert "The file is empty" in only_problem("# nothing yet\n").message


def test_top_level_list() -> None:
    problem = only_problem("- name: tone\n")
    assert "but this file holds a list" in problem.message


def test_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_codebook(tmp_path / "missing.yaml")


def test_file_that_is_not_utf8(tmp_path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes(MINIMAL.replace("criticises", "critiqué").encode("latin-1"))
    with pytest.raises(CodebookError) as info:
        load_codebook(path)
    assert info.value.source == str(path)
    assert "not saved as UTF-8" in info.value.problems[0].message


# --------------------------------------------------------------------- missing and unknown fields


def test_missing_category_definition_names_variable_category_and_line() -> None:
    problem = only_problem(
        variable_with(
            """
            - code: positive
              label: Positive
            - {code: negative, definition: Criticises the subject.}
            """
        )
    )
    assert problem.location == 'variable "tone" > category "positive"'
    assert problem.line == 6
    assert 'The required field "definition" is missing' in problem.message
    assert "tells coders when to choose it" in problem.message


def test_missing_top_level_fields() -> None:
    problems = problems_of("title: Only a title\n")
    messages = [p.message for p in problems]
    assert any('"name" is missing' in m for m in messages)
    assert any('"variables" is missing' in m for m in messages)
    assert all(p.location == "" and p.line is None for p in problems)


def test_misspelled_field_suggests_the_right_one_and_hides_the_missing_error() -> None:
    text = yaml_text(
        """
        name: demo
        variables:
          - name: tone
            defintion: The tone of the text.
            categories:
              - {code: positive, definition: Praises the subject.}
              - {code: negative, definition: Criticises the subject.}
        """
    )
    problem = only_problem(text)
    assert problem.message == '"defintion" is not a recognized field. Did you mean "definition"?'
    assert problem.line == 4


def test_unknown_field_without_a_close_match_lists_the_allowed_fields() -> None:
    problem = only_problem(variable_with(TWO_CATEGORIES, extra="colour: blue"))
    assert '"colour" is not a recognized field' in problem.message
    assert "The fields a variable can have are: name, label, definition" in problem.message


def test_decision_rules_at_the_top_level_point_to_instructions() -> None:
    text = "decision_rules: [Code the headline only.]\n" + variable_with(TWO_CATEGORIES)
    problem = only_problem(text)
    assert 'use "instructions"' in problem.message
    assert problem.line == 1


# --------------------------------------------------------------------- wrong shapes


def test_categories_written_as_one_line_of_text() -> None:
    text = yaml_text(
        """
        name: demo
        variables:
          - name: tone
            definition: The tone.
            categories: positive, negative
        """
    )
    problem = only_problem(text)
    assert problem.message.startswith('"categories" should be a list, but it is the text')
    assert 'starting with "- "' in problem.message
    assert problem.line == 5


def test_category_written_as_plain_text() -> None:
    problem = only_problem(
        variable_with(
            """
            - positive
            - {code: negative, definition: Criticises the subject.}
            """
        )
    )
    assert problem.location == 'variable "tone" > category 1'
    assert problem.message.startswith("This category should be a block of fields")
    assert '"- code: ..."' in problem.message


def test_example_with_the_wrong_shape() -> None:
    problem = only_problem(
        variable_with(
            """
            - code: positive
              definition: Praises the subject.
              examples: [Good news., [a list]]
            - {code: negative, definition: Criticises the subject.}
            """
        )
    )
    assert problem.location == 'variable "tone" > category "positive" > example 2'
    assert "should be a block of fields, but it is a list" in problem.message


def test_text_field_given_a_block_of_fields() -> None:
    problem = only_problem(variable_with(TWO_CATEGORIES, extra="label: {en: Tone}"))
    assert problem.message == '"label" should be text, but it is a block of fields.'


def test_decision_rules_given_as_text_instead_of_a_list() -> None:
    problem = only_problem(variable_with(TWO_CATEGORIES, extra="decision_rules: Code the lead."))
    assert problem.message.startswith('"decision_rules" should be a list')


# --------------------------------------------------------------------- values


def test_fewer_than_two_categories() -> None:
    problem = only_problem(variable_with("- {code: positive, definition: Praises.}"))
    assert problem.location == 'variable "tone"'
    assert '"categories" needs at least 2 entries, but it has 1' in problem.message


def test_no_variables() -> None:
    problem = only_problem("name: demo\nvariables: []\n")
    assert '"variables" needs at least 1 entry, but it has 0' in problem.message
    assert problem.line == 2


def test_yes_and_no_codes_get_a_quoting_hint() -> None:
    problems = problems_of(
        variable_with(
            """
            - {code: yes, definition: Present.}
            - {code: no, definition: Absent.}
            """
        )
    )
    assert len(problems) == 2
    assert "was read as the value true, not as text" in problems[0].message
    assert 'Put the value in quotes, for example "no"' in problems[1].message


def test_empty_definition() -> None:
    problems = problems_of(
        variable_with(
            """
            - {code: positive, definition: "   "}
            - code: negative
              definition:
            """
        )
    )
    assert [p.message.split(".")[0] for p in problems] == [
        '"definition" is empty',
        '"definition" is empty',
    ]


def test_empty_decision_rule_is_named_by_position() -> None:
    problem = only_problem(
        variable_with(TWO_CATEGORIES, extra='decision_rules: [Code the lead., ""]')
    )
    assert problem.location == 'variable "tone" > decision rule 2'
    assert problem.message.startswith("This decision rule is empty")


def test_unknown_variable_type() -> None:
    problem = only_problem(variable_with(TWO_CATEGORIES, extra="type: interval"))
    assert '"type" must be "nominal" or "ordinal", but it is the text "interval"' in problem.message


def test_variable_name_that_cannot_be_a_column() -> None:
    text = variable_with(TWO_CATEGORIES).replace("name: tone", "name: dominant frame")
    problem = only_problem(text)
    assert problem.message.startswith('"dominant frame" cannot be used as a variable name')


def test_quoted_level_gets_a_remove_quotes_hint() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: low, definition: Low., level: "1"}
            - {code: high, definition: High., level: 2}
            """,
            extra="type: ordinal",
        )
    )
    assert "should be a whole number" in problem.message
    assert "Remove the quotes" in problem.message


def test_level_that_is_not_a_number() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: low, definition: Low., level: 1.5}
            - {code: high, definition: High., level: 2}
            """,
            extra="type: ordinal",
        )
    )
    assert "but it is the number 1.5." in problem.message
    assert "Remove the quotes" not in problem.message


# --------------------------------------------------------------------- cross-field rules


def test_duplicate_codes() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: positive, definition: Praises.}
            - {code: positive, definition: Supports.}
            """
        )
    )
    assert problem.location == 'variable "tone"'
    assert problem.line == 3
    assert 'The code "positive" is used by more than one category' in problem.message


def test_duplicate_variable_names() -> None:
    text = variable_with(TWO_CATEGORIES)
    text += text.split("variables:\n", 1)[1]
    problem = only_problem(text)
    assert 'The variable name "tone" is used more than once' in problem.message
    assert problem.location == ""
    assert problem.line is None


def test_ordinal_variable_without_levels() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: none, definition: No incivility., level: 0}
            - {code: mild, definition: Mild incivility.}
            - {code: strong, definition: Strong incivility.}
            """,
            extra="type: ordinal",
        )
    )
    assert 'every category needs a "level"' in problem.message
    assert 'Missing for: "mild", "strong".' in problem.message


def test_ordinal_variable_with_repeated_levels() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: none, definition: No incivility., level: 0}
            - {code: mild, definition: Mild incivility., level: 1}
            - {code: strong, definition: Strong incivility., level: 1}
            """,
            extra="type: ordinal",
        )
    )
    assert 'Level 1 is given to more than one category ("mild", "strong")' in problem.message


def test_level_on_a_nominal_variable() -> None:
    problem = only_problem(
        variable_with(
            """
            - {code: positive, definition: Praises., level: 1}
            - {code: negative, definition: Criticises.}
            """
        )
    )
    assert '"positive" has a "level"' in problem.message
    assert 'add "type: ordinal"' in problem.message


def test_several_rule_problems_in_one_variable_are_listed_separately() -> None:
    problems = problems_of(
        variable_with(
            """
            - {code: low, definition: Low.}
            - {code: low, definition: Also low., level: 1}
            """,
            extra="type: ordinal",
        )
    )
    assert len(problems) == 2
    assert all(p.location == 'variable "tone"' for p in problems)
    assert "used by more than one category" in problems[0].message
    assert 'Missing for: "low"' in problems[1].message

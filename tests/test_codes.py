"""Reading the codes to compare: a CSV file in long form, rows, and the codes of a run."""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import AGREEMENT, MINIMAL, matrix_rows
from reliacode import FakeBackend, code, parse_codebook
from reliacode.reliability import CodedValue, Codes, CodesError, read_codes

CODEBOOK = parse_codebook(AGREEMENT)


def write(tmp_path: Path, text: str, name: str = "codes.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def problems(error: pytest.ExceptionInfo[CodesError]) -> list[str]:
    return [str(problem) for problem in error.value.problems]


# A CSV file ------------------------------------------------------------------------------------


def test_a_csv_file_in_long_form_is_read_value_by_value(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "unit,coder,variable,value\n"
        "a1,anna,tone,positive\n"
        "a1,ben,tone,neutral\n"
        "a2,anna,tone,\n"
        "a2,ben,severity,mild\n",
    )
    codes = read_codes(CODEBOOK, path)
    assert codes.values == (
        CodedValue(unit="a1", coder="anna", variable="tone", value="positive"),
        CodedValue(unit="a1", coder="ben", variable="tone", value="neutral"),
        CodedValue(unit="a2", coder="anna", variable="tone", value=None),
        CodedValue(unit="a2", coder="ben", variable="severity", value="mild"),
    )
    assert codes.coders == ("anna", "ben")
    assert codes.units == ("a1", "a2")
    assert codes.variables == ("severity", "tone")


def test_column_order_extra_columns_spaces_quotes_blank_lines_and_a_bom_are_fine(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path,
        "﻿note, value ,variable,coder,unit\r\n"
        '"a long note, with a comma",  negative ,tone, anna ,7\r\n'
        "\r\n"
        '"two\nlines",positive,tone,ben,7\r\n',
    )
    codes = read_codes(CODEBOOK, path)
    assert [(v.unit, v.coder, v.value) for v in codes.values] == [
        ("7", "anna", "negative"),
        ("7", "ben", "positive"),
    ]


def test_a_short_line_is_read_with_its_last_cells_empty(tmp_path: Path) -> None:
    codes = read_codes(CODEBOOK, write(tmp_path, "unit,coder,variable,value\na1,anna,tone\n"))
    assert codes.values[0].value is None


def test_an_unknown_code_names_the_line_unit_coder_and_variable(tmp_path: Path) -> None:
    path = write(
        tmp_path, "unit,coder,variable,value\na1,anna,tone,positive\na2,anna,tone,postive\n"
    )
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error) == [
        'Line 3, unit "a2", coder "anna", variable "tone": "postive" is not one of the codes of '
        'tone: positive, neutral, negative. Did you mean "positive"?'
    ]
    assert str(error.value).startswith(f"The codes in {path} have 1 problem:\n  1. Line 3,")
    assert error.value.source == str(path)


def test_a_code_that_differs_only_in_case_is_refused_with_a_hint(tmp_path: Path) -> None:
    path = write(tmp_path, "unit,coder,variable,value\na1,anna,tone,Positive\n")
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error)[0].endswith(
        'Did you mean "positive"? Codes must match exactly, including upper and lower case.'
    )


@pytest.mark.parametrize("word", ["NA", "n/a", "NaN", "missing", "-", "None"])
def test_a_word_for_missing_is_refused_and_an_empty_cell_suggested(
    tmp_path: Path, word: str
) -> None:
    path = write(tmp_path, f"unit,coder,variable,value\na1,anna,tone,{word}\n")
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error)[0].endswith("For a missing value, leave the value empty.")


def test_an_unknown_variable_is_named_with_the_codebook_variables(tmp_path: Path) -> None:
    path = write(tmp_path, "unit,coder,variable,value\na1,anna,tones,positive\n")
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error) == [
        'Line 2, unit "a1", coder "anna", variable "tones": The codebook has no variable '
        '"tones". Its variables are: tone, severity. Did you mean "tone"?'
    ]


def test_two_values_for_one_unit_coder_and_variable_are_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "unit,coder,variable,value\na1,anna,tone,positive\na1,ben,tone,neutral\n"
        "a1,anna,tone,positive\n",
    )
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error) == [
        'Line 4, unit "a1", coder "anna", variable "tone": This unit, coder and variable '
        "already have a value (line 2). Each coder gives one value per unit and variable: keep "
        "one and remove the other."
    ]


def test_every_problem_is_listed_and_a_long_list_is_cut_short(tmp_path: Path) -> None:
    lines = "".join(f"u{n},anna,tone,good\n" for n in range(25))
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, write(tmp_path, "unit,coder,variable,value\n" + lines))
    assert len(error.value.problems) == 25
    text = str(error.value)
    assert "have 25 problems:" in text
    assert "  20. Line 21," in text
    assert "  21." not in text
    assert text.endswith("  ... and 5 more.")


def test_an_empty_unit_or_coder_is_refused_and_problems_come_in_file_order(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path,
        "unit,coder,variable,value\na1,anna,tone,good\n,anna,tone,positive\na2,, ,positive\n",
    )
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert problems(error) == [
        'Line 2, unit "a1", coder "anna", variable "tone": "good" is not one of the codes of '
        "tone: positive, neutral, negative.",
        'Line 3: The "unit" is empty. Every code needs the unit that was coded, the coder who '
        "coded it and the variable.",
        'Line 4: The "coder" and "variable" are empty. Every code needs the unit that was coded, '
        "the coder who coded it and the variable.",
    ]


def test_a_file_without_the_four_columns_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, write(tmp_path, "unit,rater,variable,code\na1,anna,tone,positive\n"))
    assert problems(error) == [
        "Line 1: The first line should name the columns unit, coder, variable and value, but "
        '"coder" and "value" are missing. The first line is: unit,rater,variable,code.'
    ]


@pytest.mark.parametrize("text", ["", "\n\n", " , \n"])
def test_an_empty_file_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, write(tmp_path, text))
    assert problems(error)[0].startswith("The file is empty.")


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "unit,coder,variable,value\nné,anna,tone,positive\n", encoding="latin-1")
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, path)
    assert "not saved as UTF-8" in problems(error)[0]


def test_a_missing_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_codes(CODEBOOK, tmp_path / "nowhere.csv")


# Rows ------------------------------------------------------------------------------------------


def test_rows_such_as_a_data_frames_records_are_read() -> None:
    rows = [
        {"unit": 1, "coder": "anna", "variable": "tone", "value": "positive", "note": "x"},
        {"unit": 1, "coder": 2, "variable": "tone", "value": " neutral "},
        {"unit": 2, "coder": "anna", "variable": "tone", "value": math.nan},
        {"unit": 2, "coder": 2, "variable": "tone", "value": None},
        {"unit": 3, "coder": "anna", "variable": "tone", "value": ""},
    ]
    codes = read_codes(CODEBOOK, rows)
    assert [(v.unit, v.coder, v.value) for v in codes.values] == [
        ("1", "anna", "positive"),
        ("1", "2", "neutral"),
        ("2", "anna", None),
        ("2", "2", None),
        ("3", "anna", None),
    ]


def test_whole_number_values_are_read_as_codes() -> None:
    codebook = parse_codebook(
        MINIMAL.replace("code: positive", 'code: "1"').replace("code: negative", 'code: "0"')
    )
    codes = read_codes(codebook, [{"unit": "a", "coder": "anna", "variable": "tone", "value": 1}])
    assert codes.values[0].value == "1"


def test_rows_with_problems_name_the_row() -> None:
    rows = [
        {"unit": "a", "coder": "anna", "variable": "tone", "value": "positive"},
        ["a", "ben", "tone", "positive"],
        {"unit": "b", "coder": "anna", "variable": "tone"},
        {"unit": 1.5, "coder": "anna", "variable": "tone", "value": "positive"},
        {"unit": "c", "coder": True, "variable": "tone", "value": "positive"},
        {"unit": "d", "coder": "anna", "variable": "tone", "value": 0.5},
        {"unit": "e", "coder": "anna", "variable": "tone", "value": "good"},
        {"unit": "", "coder": "anna", "variable": "tone", "value": "positive"},
    ]
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, rows)
    assert problems(error) == [
        "Row 2: This row is a list, but each row should have the fields unit, coder, variable "
        "and value.",
        'Row 3: This row has no "value".',
        "Row 4: The unit should be text or a whole number, but it is the number 1.5.",
        "Row 5: The coder should be text or a whole number, but it is the value true.",
        "Row 6: The value should be one of the variable's codes, or empty when it is missing, "
        "but it is the number 0.5.",
        'Row 7, unit "e", coder "anna", variable "tone": "good" is not one of the codes of tone: '
        "positive, neutral, negative.",
        'Row 8: The "unit" is empty. Every code needs the unit that was coded, the coder who '
        "coded it and the variable.",
    ]
    assert error.value.source is None
    assert str(error.value).startswith("The codes have 7 problems:")


@pytest.mark.parametrize("source", [{"unit": "a"}, 42])
def test_a_source_that_is_not_a_file_or_rows_is_a_type_error(source: object) -> None:
    with pytest.raises(TypeError, match="source should be the path of a CSV file"):
        read_codes(CODEBOOK, source)  # type: ignore[arg-type]


def test_a_matrix_helper_gives_one_row_per_coder_and_unit() -> None:
    rows = matrix_rows("tone", {"A": ["positive", None], "B": ["neutral", "negative"]})
    codes = read_codes(CODEBOOK, rows)
    assert codes.units == ("1", "2")
    assert len(codes.values) == 4


# A coding run ----------------------------------------------------------------------------------


def tone_run(*answers: str, texts: object = None) -> object:
    backend = FakeBackend(list(answers))
    texts = texts if texts is not None else [f"Invented text {n}." for n in range(len(answers))]
    return code(CODEBOOK, texts, backend=backend, variables="tone", attempts=1)


def test_a_run_becomes_the_coder_model_with_its_text_ids_as_units() -> None:
    run = tone_run('{"code": "positive"}', "not json", '{"code": "negative"}')
    codes = read_codes(CODEBOOK, run=run)
    assert codes.values == (
        CodedValue(unit="0", coder="model", variable="tone", value="positive"),
        CodedValue(unit="1", coder="model", variable="tone", value=None),
        CodedValue(unit="2", coder="model", variable="tone", value="negative"),
    )


def test_human_codes_and_a_run_are_read_together_on_the_same_units(tmp_path: Path) -> None:
    run = tone_run('{"code": "positive"}', '{"code": "neutral"}', texts={"a1": "x", "a2": "y"})
    path = write(tmp_path, "unit,coder,variable,value\na1,anna,tone,positive\na2,anna,tone,\n")
    codes = read_codes(CODEBOOK, path, run=run)
    assert codes.coders == ("anna", "model")
    assert codes.units == ("a1", "a2")
    assert len(codes.values) == 4


def test_several_runs_are_named_by_their_coders() -> None:
    first = tone_run('{"code": "positive"}')
    second = tone_run('{"code": "negative"}')
    codes = read_codes(CODEBOOK, run={"qwen": first, " gemma ": second})
    assert [(v.coder, v.value) for v in codes.values] == [
        ("qwen", "positive"),
        ("gemma", "negative"),
    ]


def test_a_run_coded_with_another_codebook_is_refused() -> None:
    other = parse_codebook(AGREEMENT.replace("agreement-demo", "agreement-other"))
    with pytest.raises(CodesError) as error:
        read_codes(other, run=tone_run('{"code": "positive"}'))
    message = problems(error)[0]
    assert message.startswith('The run "model" was coded with another codebook (sha256:')
    assert f"than the one given ({other.content_hash()})" in message


def test_a_run_may_not_share_a_name_with_a_human_coder() -> None:
    rows = [{"unit": "5", "coder": "model", "variable": "tone", "value": "positive"}]
    with pytest.raises(CodesError) as error:
        read_codes(CODEBOOK, rows, run=tone_run('{"code": "positive"}'))
    assert problems(error) == [
        'The coder name "model" is used by the codes and by a run. Give the run another name, '
        'for example run={"model_run": run}.'
    ]


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ("run.json", "run should be what reliacode.code() returned"),
        ({"qwen": "run.json"}, 'The run "qwen" should be what reliacode.code() returned'),
        ({1: None}, "The names of runs should be text, but one is the number 1."),
        ({" ": None}, "The names of runs should be text"),
    ],
)
def test_a_run_that_is_not_a_coding_run_is_a_type_error(run: object, message: str) -> None:
    with pytest.raises(TypeError, match=message.replace("(", r"\(").replace(")", r"\)")):
        read_codes(CODEBOOK, run=run)  # type: ignore[arg-type]


def test_codes_need_a_codebook_and_something_to_read() -> None:
    with pytest.raises(TypeError, match="codebook should be a codebook"):
        read_codes("codebook.yaml", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Give the codes to compare"):
        read_codes(CODEBOOK)


# The record ------------------------------------------------------------------------------------


def test_codes_are_frozen_and_hold_one_value_per_unit_coder_and_variable() -> None:
    value = CodedValue(unit="a", coder="anna", variable="tone", value="positive")
    codes = Codes(values=(value,))
    with pytest.raises(ValidationError):
        codes.values = ()  # type: ignore[misc]
    with pytest.raises(ValidationError, match="more than one value"):
        Codes(values=(value, value.model_copy(update={"value": "negative"})))
    with pytest.raises(ValidationError):
        CodedValue(unit="a", coder="anna", variable="tone", value="x", note="extra")  # type: ignore[call-arg]


def test_codes_give_rows_for_a_data_frame() -> None:
    codes = read_codes(CODEBOOK, matrix_rows("tone", {"A": ["positive", None]}))
    assert codes.rows() == [
        {"unit": "1", "coder": "A", "variable": "tone", "value": "positive"},
        {"unit": "2", "coder": "A", "variable": "tone", "value": None},
    ]

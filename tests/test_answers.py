"""Answer checking: an answer is one of the codes, exactly, or it is explained and not used."""

import pytest

from reliacode.coding import check_answer

CODES = ("conflict", "economic", "none")


@pytest.mark.parametrize(
    "answer",
    [
        '{"code": "economic"}',
        '  {"code": "economic"}\n',
        '{"code": " economic "}',
        '{"code": "economic", "reason": "The article is about the budget."}',
        '```json\n{"code": "economic"}\n```',
        '```\n{"code": "economic"}\n```',
        '<think>The article talks about costs, so...</think>\n{"code": "economic"}',
        '<THINK>\nmulti\nline\n</THINK>```json\n{"code": "economic"}```',
    ],
    ids=[
        "plain",
        "spaces",
        "spaces-in-code",
        "extra-field",
        "json-fence",
        "bare-fence",
        "thinking",
        "thinking-and-fence",
    ],
)
def test_valid_answers_give_the_code(answer: str) -> None:
    assert check_answer(answer, CODES) == ("economic", None)


def test_a_whole_number_is_read_as_a_numeric_code() -> None:
    assert check_answer('{"code": 2}', ("0", "1", "2")) == ("2", None)


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ("", "The answer is empty."),
        ("   \n", "The answer is empty."),
        ("<think>hmm</think>", "The answer is empty."),
        ("economic", 'The answer is not a JSON object: "economic".'),
        ("<think>still thinking when the token limit", "The answer is not a JSON object"),
        ('{"code": "economic"', "The answer is not a JSON object"),
        ('["economic"]', 'The answer is not a JSON object with a "code" field: "["economic"]".'),
        ('"economic"', 'not a JSON object with a "code" field'),
        ('{"frame": "economic"}', 'The answer has no "code" field: "{"frame": "economic"}".'),
        ('{"code": null}', 'The "code" in the answer is not text: "null".'),
        ('{"code": true}', 'The "code" in the answer is not text: "true".'),
        ('{"code": ["economic"]}', "is not text"),
        ('{"code": "Economic"}', '"Economic" is not one of the codes.'),
        ('{"code": "economy"}', '"economy" is not one of the codes.'),
        ('{"code": ""}', '"" is not one of the codes.'),
        ('{"code": 1}', '"1" is not one of the codes.'),
    ],
)
def test_an_answer_that_cannot_be_used_is_explained(answer: str, problem: str) -> None:
    code, explained = check_answer(answer, CODES)
    assert code is None
    assert explained is not None
    assert problem in explained


def test_a_long_bad_answer_is_quoted_briefly() -> None:
    _, problem = check_answer("I think the answer is " + "very " * 50 + "economic", CODES)
    assert problem is not None
    assert problem.endswith('...".')
    assert len(problem) < 120


def test_case_is_never_folded_and_near_misses_are_never_corrected() -> None:
    for near_miss in ("CONFLICT", "Conflict", "conflicts", "con flict"):
        assert check_answer(f'{{"code": "{near_miss}"}}', CODES)[0] is None

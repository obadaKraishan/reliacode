"""code(): one checked code per variable per text, with a full record of how each was obtained."""

import json
import logging
import secrets
from datetime import timezone

import httpx
import pytest

import reliacode
from helpers import EXAMPLES, MINIMAL, FakeServer, completion
from reliacode import (
    BackendError,
    CodingResult,
    CodingRun,
    FakeBackend,
    OpenAICompatibleBackend,
    code,
    load_codebook,
    parse_codebook,
)
from reliacode.coding import build_prompt

TONE = parse_codebook(MINIMAL)
TEXTS = [
    "The volunteers cleaned the whole beach in one morning. Wonderful work.",
    "The repairs on the bridge are late again, and the costs keep rising.",
]
POSITIVE, NEGATIVE = '{"code": "positive"}', '{"code": "negative"}'


def by_keyword(messages: list[dict[str, str]]) -> str:
    """A scripted coder: positive when the text praises, negative otherwise."""
    text = messages[1]["content"]
    return POSITIVE if "Wonderful" in text else NEGATIVE


# The run and its records -------------------------------------------------------------------------


def test_every_text_gets_a_checked_code() -> None:
    run = code(TONE, TEXTS, backend=FakeBackend(by_keyword))
    assert isinstance(run, CodingRun)
    assert run.table() == [
        {"unit": 0, "tone": "positive"},
        {"unit": 1, "tone": "negative"},
    ]
    assert run.missing() == ()


def test_every_result_records_what_the_issue_asks_for() -> None:
    # The model name, endpoint, parameters, codebook content hash, prompt hash and timestamp.
    run = code(TONE, TEXTS, backend=FakeBackend(by_keyword, model="scripted-coder"))
    prompt = build_prompt(TONE, "tone")
    for result in run.results:
        assert isinstance(result, CodingResult)
        assert result.model == "scripted-coder"
        assert result.endpoint == "fake"
        assert result.parameters.model == "scripted-coder"
        assert result.codebook_hash == TONE.content_hash()
        assert result.prompt_hash == prompt.hash
        assert result.created_at.tzinfo == timezone.utc
        assert result.answers == (result.responses[0].text,)
        assert not result.is_missing


def test_the_run_records_the_codebook_the_prompts_and_the_time() -> None:
    run = code(TONE, TEXTS, backend=FakeBackend(by_keyword))
    assert run.codebook_name == "tone-demo"
    assert run.codebook_version is None
    assert run.codebook_hash == TONE.content_hash()
    assert run.prompts == (build_prompt(TONE, "tone"),)
    assert run.started_at <= run.finished_at
    assert run.started_at.tzinfo == timezone.utc
    assert run.reliacode_version == reliacode.__version__


def test_each_request_is_the_prompt_and_the_text_with_the_schema() -> None:
    backend = FakeBackend(by_keyword)
    code(TONE, TEXTS, backend=backend)
    prompt = build_prompt(TONE, "tone")
    assert backend.calls == [prompt.messages(text) for text in TEXTS]
    assert backend.json_schemas == [prompt.json_schema, prompt.json_schema]


def test_texts_are_coded_one_variable_at_a_time_in_order() -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    answers = ['{"code": "none"}', '{"code": "none"}', '{"code": "lying"}', '{"code": "strong"}']
    texts = {"c1": "Thanks for the clear summary.", "c2": "The mayor is lying about the funds."}
    run = code(codebook, texts, backend=FakeBackend(answers))
    assert [(r.unit, r.variable) for r in run.results] == [
        ("c1", "incivility_form"),
        ("c1", "severity"),
        ("c2", "incivility_form"),
        ("c2", "severity"),
    ]
    assert run.table() == [
        {"unit": "c1", "incivility_form": "none", "severity": "none"},
        {"unit": "c2", "incivility_form": "lying", "severity": "strong"},
    ]
    assert len({r.prompt_hash for r in run.results}) == 2


def test_the_run_survives_a_json_round_trip() -> None:
    run = code(TONE, TEXTS, backend=FakeBackend(by_keyword))
    assert CodingRun.model_validate_json(run.model_dump_json()) == run


# Answers that cannot be used ---------------------------------------------------------------------


def test_an_invalid_answer_is_asked_again_with_the_reason() -> None:
    backend = FakeBackend(['{"code": "Positive"}', POSITIVE])
    run = code(TONE, TEXTS[:1], backend=backend)
    result = run.results[0]
    assert result.code == "positive"
    assert result.answers == ('{"code": "Positive"}', POSITIVE)
    assert len(result.responses) == 2
    retry = backend.calls[1]
    assert retry[:2] == backend.calls[0]
    assert retry[2] == {"role": "assistant", "content": '{"code": "Positive"}'}
    assert retry[3] == {
        "role": "user",
        "content": 'That answer cannot be used: "Positive" is not one of the codes. Answer '
        'again with only a JSON object of the form {"code": "..."}, where the code is one of: '
        "positive, negative.",
    }


def test_an_answer_outside_the_codes_never_enters_the_results() -> None:
    backend = FakeBackend(lambda messages: '{"code": "mixed"}')
    run = code(TONE, TEXTS, backend=backend)
    assert [row["tone"] for row in run.table()] == [None, None]
    assert len(run.missing()) == 2
    result = run.missing()[0]
    assert result.is_missing
    assert result.answers == ('{"code": "mixed"}', '{"code": "mixed"}')
    assert result.missing_reason == (
        'None of the 2 answers could be used. The last one: "mixed" is not one of the codes.'
    )
    assert len(backend.calls) == 4


def test_with_one_attempt_there_is_no_second_try() -> None:
    backend = FakeBackend(["not json"])
    result = code(TONE, TEXTS[:1], backend=backend, attempts=1).results[0]
    assert len(backend.calls) == 1
    assert result.missing_reason == (
        'The answer could not be used. The answer is not a JSON object: "not json".'
    )


def test_more_attempts_can_be_allowed() -> None:
    backend = FakeBackend(["", "maybe positive", POSITIVE])
    result = code(TONE, TEXTS[:1], backend=backend, attempts=3).results[0]
    assert result.code == "positive"
    assert result.answers == ("", "maybe positive", POSITIVE)


def test_invalid_answers_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="reliacode")
    code(TONE, {"a7": TEXTS[0]}, backend=FakeBackend(["nope", POSITIVE]))
    assert caplog.messages == [
        'The text "a7", variable tone: answer 1 of 2 cannot be used: the answer is not a JSON '
        "object."
    ]


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ("Probably positive, given {mark}.", "the answer is not a JSON object"),
        ('["{mark}"]', 'the answer is not a JSON object with a "code" field'),
        ('{{"label": "{mark}"}}', 'the answer has no "code" field'),
        ('{{"code": ["{mark}"]}}', 'the "code" in the answer is not text'),
        ('{{"code": "{mark}"}}', "the code in the answer is not one of the codes"),
    ],
)
def test_no_word_of_an_unusable_answer_is_logged(
    answer: str, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, at every level
    mark = secrets.token_hex(12)
    run = code(TONE, ["An invented text."], backend=FakeBackend([answer.format(mark=mark)] * 2))
    assert [record.levelname for record in caplog.records] == ["INFO", "INFO"]
    assert all(kind in message for message in caplog.messages)
    logged = [f"{record.getMessage()} {record.args!r}" for record in caplog.records]
    assert all(mark not in line for line in logged)
    result = run.results[0]
    assert mark in result.answers[0]  # the answer stays in the results, to check by hand
    assert result.missing_reason is not None


# Choosing variables ------------------------------------------------------------------------------


def test_variables_can_be_chosen_by_name() -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    run = code(codebook, TEXTS[:1], backend=FakeBackend(['{"code": "mild"}']), variables="severity")
    assert run.table() == [{"unit": 0, "severity": "mild"}]
    assert [p.variable for p in run.prompts] == ["severity"]
    run = code(
        codebook,
        TEXTS[:1],
        backend=FakeBackend(['{"code": "none"}', '{"code": "none"}']),
        variables=["severity", "incivility_form"],
    )
    assert list(run.table()[0]) == ["unit", "severity", "incivility_form"]


@pytest.mark.parametrize(
    ("variables", "error", "match"),
    [
        ("frame", KeyError, 'no variable "frame"'),
        ([], ValueError, "variables is empty"),
        (["tone", "tone"], ValueError, 'The variable "tone" is named more than once'),
    ],
)
def test_wrong_variables_are_explained(variables, error, match) -> None:
    with pytest.raises(error, match=match):
        code(TONE, TEXTS, backend=FakeBackend([]), variables=variables)


# Wrong input -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texts", "error", "match"),
    [
        ("A single text.", TypeError, r"To code one text, write \[text\]"),
        ([], ValueError, "nothing to code"),
        ({}, ValueError, "nothing to code"),
        (42, TypeError, "but it is the number 42"),
        (["Fine.", None], TypeError, "Text 1 should be text, but it is empty. Remove missing"),
        (["Fine.", float("nan")], TypeError, "Text 1 should be text, but it is the number nan"),
        ({"x9": "   "}, ValueError, 'The text "x9" is empty'),
        ({(1, 2): "Fine."}, TypeError, "Text ids should be text or whole numbers"),
        ({True: "Fine."}, TypeError, "Text ids should be text or whole numbers"),
    ],
)
def test_wrong_texts_are_explained(texts, error, match) -> None:
    backend = FakeBackend(lambda messages: POSITIVE)
    with pytest.raises(error, match=match):
        code(TONE, texts, backend=backend)
    assert backend.calls == []


def test_texts_can_come_from_any_iterable() -> None:
    run = code(TONE, (text for text in TEXTS), backend=FakeBackend(by_keyword))
    assert [row["unit"] for row in run.table()] == [0, 1]


@pytest.mark.parametrize("attempts", [0, -1, "2", True, 1.5])
def test_attempts_must_be_a_whole_number_of_at_least_one(attempts) -> None:
    with pytest.raises(ValueError, match="attempts should be a whole number"):
        code(TONE, TEXTS, backend=FakeBackend([]), attempts=attempts)


def test_the_codebook_and_the_backend_are_checked() -> None:
    with pytest.raises(TypeError, match="load_codebook"):
        code(MINIMAL, TEXTS, backend=FakeBackend([]))
    with pytest.raises(TypeError, match="should be a reliacode backend"):
        code(TONE, TEXTS, backend="http://localhost:11434/v1")


def test_a_backend_must_return_a_response() -> None:
    class Wrong:
        def complete(self, messages, *, json_schema=None):
            return POSITIVE

    with pytest.raises(TypeError, match="should return a reliacode Response"):
        code(TONE, TEXTS, backend=Wrong())


def test_a_backend_error_stops_the_run() -> None:
    stop = BackendError("No server.", reason="no_server", endpoint="fake")
    backend = FakeBackend([POSITIVE, stop, POSITIVE])
    with pytest.raises(BackendError, match=r"No server\."):
        code(TONE, TEXTS, backend=backend)
    assert len(backend.calls) == 2


# End to end through the HTTP backend -------------------------------------------------------------


def test_code_runs_end_to_end_through_the_http_backend() -> None:
    server = FakeServer(
        httpx.Response(200, json=completion(POSITIVE)),
        httpx.Response(200, json=completion("Probably negative.")),
        httpx.Response(200, json=completion(NEGATIVE, usage={"prompt_tokens": 90})),
    )
    backend = OpenAICompatibleBackend(
        "http://localhost:11434/v1", "tiny-model", seed=1, transport=server.transport
    )
    run = code(TONE, TEXTS, backend=backend)
    assert run.table() == [{"unit": 0, "tone": "positive"}, {"unit": 1, "tone": "negative"}]
    body = server.body(0)
    assert body["response_format"]["json_schema"]["schema"]["properties"]["code"]["enum"] == [
        "positive",
        "negative",
    ]
    assert [m["role"] for m in server.body(2)["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    first, second = run.results
    assert first.model == "tiny-model-served"
    assert first.endpoint == "http://localhost:11434/v1/chat/completions"
    assert first.parameters.seed == 1
    assert first.responses[0].prompt_tokens == 42
    assert second.answers == ("Probably negative.", NEGATIVE)
    assert [r.prompt_tokens for r in second.responses] == [42, 90]
    record = json.loads(run.model_dump_json())
    assert record["results"][1]["responses"][1]["parameters"]["structured_output"] is True

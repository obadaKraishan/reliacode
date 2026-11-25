"""One call per unit: every variable of a text in one call, each code checked against its
variable's codes, one more ask for a broken answer, and one result per text and variable.

Every codebook, text and answer here is invented.
"""

import json
import logging
import secrets
from pathlib import Path

import httpx
import pytest

from helpers import FakeServer, completion, yaml_text
from reliacode import (
    BackendError,
    Budget,
    BudgetError,
    CodingRun,
    FakeBackend,
    Ledger,
    OpenAICompatibleBackend,
    code,
    codebook_reliability,
    parse_codebook,
    read_codes,
)
from reliacode.coding import (
    UNIT_INSTRUCTION,
    UNIT_PROMPT_VERSION,
    UNIT_RETRY_TEMPLATE,
    USER_TEMPLATE,
    UnitPrompt,
    build_prompt,
    build_unit_prompt,
    unit_answer_schema,
)

CODEBOOK = parse_codebook(
    yaml_text(
        """
        name: council-news
        version: "1.0"
        language: en
        variables:
          - name: tone
            definition: The overall tone of the text toward its main subject.
            categories:
              - code: positive
                definition: The text mostly praises or supports its subject.
              - code: negative
                definition: The text mostly criticizes or opposes its subject.
          - name: topic
            label: Main topic
            definition: The main topic of the text.
            decision_rules:
              - Code the topic of the first paragraph.
            categories:
              - code: transport
                definition: Buses, roads, bridges, ferries or trains.
                examples:
                  - The ferry now runs every hour.
              - code: schools
                definition: Schools, teachers or pupils.
        """
    )
)
TEXTS = {
    "n1": "The new bus line finally reaches the hospital, and riders are pleased.",
    "n2": "Parents are angry that the school library closes early.",
}
BOTH = {"n1": ("positive", "transport"), "n2": ("negative", "schools")}


def answer(**codes: object) -> str:
    return json.dumps(codes)


def by_text_answer(unit: str) -> str:
    tone, topic = BOTH[unit]
    return answer(tone=tone, topic=topic)


def by_text(messages: list[dict[str, str]]) -> str:
    """A scripted coder that answers each invented text correctly in one object."""
    tone, topic = BOTH["n1"] if "bus line" in messages[1]["content"] else BOTH["n2"]
    return answer(tone=tone, topic=topic)


GOLDEN_SYSTEM = """\
You are a careful coder in a quantitative content analysis. Code every variable below for the \
text you are given, following the codebook below exactly.

Codebook: council-news, version 1.0
Language of the texts: en

Variable: tone
Definition: The overall tone of the text toward its main subject.

Categories (choose exactly one):
- positive: The text mostly praises or supports its subject.
- negative: The text mostly criticizes or opposes its subject.

Variable: topic (Main topic)
Definition: The main topic of the text.
Decision rules:
- Code the topic of the first paragraph.

Categories (choose exactly one):
- transport: Buses, roads, bridges, ferries or trains.
  Example: "The ferry now runs every hour."
- schools: Schools, teachers or pupils.

Answer with only a JSON object with one field for each variable, in this form:
{"tone": "...", "topic": "..."}
The codes for tone are: positive, negative.
The codes for topic are: transport, schools.
Write nothing else."""
#: Pinned hash of GOLDEN_SYSTEM's prompt. If this test fails, the unit prompt changed: bump
#: UNIT_PROMPT_VERSION and add a CHANGELOG entry.
GOLDEN_HASH = "sha256:beb994353f2160a0ddbd5902682afda299ffbffda5011433a66d5eae36de5be1"


# The prompt --------------------------------------------------------------------------------------


def test_the_unit_prompt_is_pinned() -> None:
    prompt = build_unit_prompt(CODEBOOK)
    assert prompt.system == GOLDEN_SYSTEM
    assert prompt.hash == GOLDEN_HASH
    assert (prompt.recipe, prompt.recipe_version, UNIT_PROMPT_VERSION) == ("per_unit", 1, 1)
    assert prompt.variables == ("tone", "topic")
    assert prompt.user_template == USER_TEMPLATE
    assert prompt.retry_template == UNIT_RETRY_TEMPLATE
    assert prompt.system.startswith(UNIT_INSTRUCTION)


def test_the_schema_has_one_field_per_variable_limited_to_its_codes() -> None:
    schema = build_unit_prompt(CODEBOOK).json_schema
    assert schema == {
        "type": "object",
        "properties": {
            "tone": {"type": "string", "enum": ["positive", "negative"]},
            "topic": {"type": "string", "enum": ["transport", "schools"]},
        },
        "required": ["tone", "topic"],
        "additionalProperties": False,
    }
    assert schema == unit_answer_schema(CODEBOOK.variables)


def test_the_unit_prompt_holds_each_variable_as_the_default_prompt_does() -> None:
    unit = build_unit_prompt(CODEBOOK).system
    for variable in CODEBOOK.variables:
        alone = build_prompt(CODEBOOK, variable.name).system
        block = alone.split("\n\nVariable: ", 1)[1].rsplit("\n\nAnswer with only", 1)[0]
        assert f"\n\nVariable: {block}\n\n" in unit


def test_chosen_variables_are_asked_in_the_order_given() -> None:
    prompt = build_unit_prompt(CODEBOOK, ["topic", "tone"])
    assert prompt.variables == ("topic", "tone")
    assert '{"topic": "...", "tone": "..."}' in prompt.system
    assert prompt.system.index("Variable: topic") < prompt.system.index("Variable: tone")
    assert list(prompt.json_schema["properties"]) == ["topic", "tone"]
    only = build_unit_prompt(CODEBOOK, "tone")
    assert only.variables == ("tone",) and "Variable: topic" not in only.system
    assert len({prompt.hash, only.hash, build_unit_prompt(CODEBOOK).hash}) == 3


def test_the_hash_changes_with_the_codebook() -> None:
    other = CODEBOOK.model_copy(update={"version": "1.1"})
    assert build_unit_prompt(other).hash != build_unit_prompt(CODEBOOK).hash


# The run and its records -------------------------------------------------------------------------


def test_one_call_per_text_gives_one_result_per_text_and_variable() -> None:
    backend = FakeBackend(by_text)
    run = code(CODEBOOK, TEXTS, backend=backend, recipe="per_unit")
    assert isinstance(run, CodingRun)
    assert run.recipe == "per_unit"
    assert len(backend.calls) == 2
    assert run.table() == [
        {"unit": "n1", "tone": "positive", "topic": "transport"},
        {"unit": "n2", "tone": "negative", "topic": "schools"},
    ]
    assert [(r.unit, r.variable) for r in run.results] == [
        ("n1", "tone"),
        ("n1", "topic"),
        ("n2", "tone"),
        ("n2", "topic"),
    ]
    prompt = build_unit_prompt(CODEBOOK)
    assert run.prompts == (prompt,)
    assert all(result.prompt_hash == prompt.hash for result in run.results)
    assert all(result.codebook_hash == CODEBOOK.content_hash() for result in run.results)
    first, second = run.results[:2]
    assert first.answers == second.answers and first.responses == second.responses
    assert run.missing() == ()


def test_each_request_is_the_unit_prompt_and_the_text_with_the_schema() -> None:
    backend = FakeBackend(by_text)
    code(CODEBOOK, {"n1": TEXTS["n1"]}, backend=backend, recipe="per_unit")
    prompt = build_unit_prompt(CODEBOOK)
    assert backend.calls == [prompt.messages(TEXTS["n1"])]
    assert backend.json_schemas == [prompt.json_schema]
    assert backend.calls[0][1]["content"] == f"Text to code:\n\n{TEXTS['n1']}"


def test_the_default_recipe_is_still_one_call_per_variable() -> None:
    backend = FakeBackend([answer(code="positive"), answer(code="transport")])
    run = code(CODEBOOK, {"n1": TEXTS["n1"]}, backend=backend)
    assert run.recipe == "per_variable"
    assert len(backend.calls) == 2
    assert run.prompts == (build_prompt(CODEBOOK, "tone"), build_prompt(CODEBOOK, "topic"))


def test_a_per_unit_run_survives_a_json_round_trip() -> None:
    run = code(CODEBOOK, TEXTS, backend=FakeBackend(by_text), recipe="per_unit")
    again = CodingRun.model_validate_json(run.model_dump_json())
    assert again == run
    assert isinstance(again.prompts[0], UnitPrompt)


def test_variables_can_be_chosen() -> None:
    backend = FakeBackend([answer(topic="schools")])
    run = code(
        CODEBOOK, ["Pupils planted trees."], backend=backend, variables="topic", recipe="per_unit"
    )
    assert run.table() == [{"unit": 0, "topic": "schools"}]
    assert run.prompts[0].variables == ("topic",)


@pytest.mark.parametrize("recipe", ["per-unit", "unit", "", None, 1])
def test_an_unknown_recipe_is_explained(recipe: object) -> None:
    with pytest.raises(ValueError, match='recipe should be "per_variable"') as caught:
        code(CODEBOOK, TEXTS, backend=FakeBackend(by_text), recipe=recipe)  # type: ignore[arg-type]
    assert '"per_unit"' in str(caught.value)


# The checks and the second ask -------------------------------------------------------------------


def test_an_unknown_code_is_asked_again_naming_the_variable_and_its_codes() -> None:
    backend = FakeBackend(
        [answer(tone="Positive", topic="transport"), answer(tone="positive", topic="transport")]
    )
    run = code(CODEBOOK, {"n1": TEXTS["n1"]}, backend=backend, recipe="per_unit")
    assert run.table() == [{"unit": "n1", "tone": "positive", "topic": "transport"}]
    second = backend.calls[1]
    assert second[:2] == build_unit_prompt(CODEBOOK).messages(TEXTS["n1"])
    assert second[2] == {"role": "assistant", "content": answer(tone="Positive", topic="transport")}
    assert second[3] == {
        "role": "user",
        "content": (
            "That answer cannot be used as it is:\n"
            '- The code for "tone" is not one of its codes: positive, negative.\n'
            "Answer again with only a JSON object with one field for each variable (tone, "
            "topic), each holding one of that variable's codes. Write nothing else."
        ),
    }
    assert run.results[0].answers == (
        answer(tone="Positive", topic="transport"),
        answer(tone="positive", topic="transport"),
    )


@pytest.mark.parametrize(
    ("first", "problems"),
    [
        ("", ["The answer is empty."]),
        ("tone: positive, topic: transport", ["The answer is not a JSON object."]),
        ('["positive", "transport"]', ["The answer is not a JSON object."]),
        (answer(tone="positive"), ['The answer has no field for "topic".']),
        (
            answer(tone=["positive"], topic=2),
            [
                'The value for "tone" is not text.',
                'The code for "topic" is not one of its codes: transport, schools.',
            ],
        ),
        (
            answer(Tone="positive", Topic="transport"),
            ['The answer has no field for "tone".', 'The answer has no field for "topic".'],
        ),
    ],
)
def test_each_problem_is_named_in_the_second_ask(first: str, problems: list[str]) -> None:
    backend = FakeBackend([first, answer(tone="positive", topic="transport")])
    run = code(CODEBOOK, ["The ferry is late."], backend=backend, recipe="per_unit")
    assert run.missing() == ()
    retry = backend.calls[1][-1]["content"]
    assert retry.splitlines()[1:-1] == [f"- {problem}" for problem in problems]


def test_after_the_last_answer_an_unusable_variable_is_missing_and_the_others_kept() -> None:
    backend = FakeBackend([answer(tone="glad", topic="transport")] * 2)
    run = code(CODEBOOK, {"n1": TEXTS["n1"]}, backend=backend, recipe="per_unit")
    tone, topic = run.results
    assert (tone.code, topic.code) == (None, "transport")
    assert tone.missing_reason == (
        'The last of the 2 answers could not be used for this variable. The code for "tone" '
        "is not one of its codes: positive, negative."
    )
    assert topic.missing_reason is None
    assert run.missing() == (tone,)


def test_an_answer_that_is_not_json_leaves_every_variable_missing() -> None:
    run = code(
        CODEBOOK,
        ["Invented."],
        backend=FakeBackend(["no idea", "still no idea"]),
        recipe="per_unit",
    )
    assert [r.code for r in run.results] == [None, None]
    assert {r.missing_reason for r in run.results} == {
        "The last of the 2 answers could not be used for this variable. The answer is not a JSON "
        "object."
    }


def test_the_last_answer_decides_every_code() -> None:
    backend = FakeBackend(
        [answer(tone="positive", topic="harbor"), answer(tone="happy", topic="schools")]
    )
    run = code(CODEBOOK, ["Invented."], backend=backend, recipe="per_unit")
    tone, topic = run.results
    assert (tone.code, topic.code) == (None, "schools")  # tone was fine in the first answer only
    assert "The last of the 2 answers" in (tone.missing_reason or "")


def test_with_one_attempt_there_is_no_second_ask() -> None:
    backend = FakeBackend([answer(tone="positive")])
    run = code(CODEBOOK, ["Invented."], backend=backend, recipe="per_unit", attempts=1)
    assert len(backend.calls) == 1
    assert run.results[1].missing_reason == (
        'The answer could not be used for this variable. The answer has no field for "topic".'
    )
    assert run.results[0].code == "positive"


def test_more_attempts_can_be_allowed() -> None:
    script = ["", "nope", answer(tone="negative", topic="schools")]
    run = code(CODEBOOK, ["Invented."], backend=FakeBackend(script), recipe="per_unit", attempts=3)
    assert run.table() == [{"unit": 0, "tone": "negative", "topic": "schools"}]
    assert len(run.results[0].answers) == 3


def test_the_same_leniency_as_one_call_per_variable() -> None:
    codebook = parse_codebook(
        yaml_text(
            """
            name: scale-demo
            variables:
              - name: stars
                definition: The rating the review gives.
                type: ordinal
                categories:
                  - {code: "1", definition: One star., level: 1}
                  - {code: "2", definition: Two stars., level: 2}
              - name: tone
                definition: The tone of the review.
                categories:
                  - {code: positive, definition: Praise.}
                  - {code: negative, definition: Complaint.}
            """
        )
    )
    reply = (
        "<think>The reviewer is happy.</think>\n```json\n"
        '{"stars": 2, "tone": " positive ", "note": "extra"}\n```'
    )
    run = code(codebook, ["Invented."], backend=FakeBackend([reply]), recipe="per_unit")
    assert run.table() == [{"unit": 0, "stars": "2", "tone": "positive"}]


# Privacy: neither the text nor the answer is quoted or logged -----------------------------------


def test_no_word_of_the_text_or_the_answer_is_quoted_or_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    text_mark, answer_mark = secrets.token_hex(12), secrets.token_hex(12)
    text = f"A private note that names {text_mark}."
    unusable = [
        f"I would say {answer_mark}.",
        answer(tone=answer_mark, topic=[answer_mark]),
    ]
    backend = FakeBackend(unusable)
    run = code(CODEBOOK, [text], backend=backend, recipe="per_unit")
    retry = backend.calls[1][-1]["content"]
    for mark in (text_mark, answer_mark):
        assert mark not in retry
        assert all(mark not in (result.missing_reason or "") for result in run.results)
        assert all(mark not in f"{r.getMessage()} {r.args!r}" for r in caplog.records)
    assert [record.levelname for record in caplog.records] == ["INFO", "INFO"]
    assert caplog.messages == [
        "Text 0: answer 1 of 2 cannot be used as it is: the answer is not a JSON object.",
        "Text 0: answer 2 of 2 cannot be used as it is: the code for tone is not one of its "
        "codes; the value for topic is not text.",
    ]
    assert answer_mark in run.results[0].answers[0]  # kept in the results, to check by hand


# The ledger and the budget -----------------------------------------------------------------------


def test_the_ledger_keeps_one_line_per_call(tmp_path: Path) -> None:
    backend = FakeBackend(
        [answer(tone="positive"), by_text_answer("n1"), by_text_answer("n2")], tokens=(300, 12)
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    code(CODEBOOK, TEXTS, backend=backend, ledger=ledger, recipe="per_unit")
    entries = ledger.entries()
    assert [(e.variable, e.outcome, e.ask) for e in entries] == [
        ("tone+topic", "asked_again", 1),
        ("tone+topic", "answered", 2),
        ("tone+topic", "answered", 1),
    ]
    prompt_hash = build_unit_prompt(CODEBOOK).hash
    assert {entry.prompt_hash for entry in entries} == {prompt_hash}
    summary = ledger.summary()
    assert summary.total.calls == 3 and summary.total.texts == 2
    assert list(summary.by_variable) == ["tone+topic"]
    assert summary.prompt_hashes == (prompt_hash,)


def test_a_line_is_missing_when_a_variable_is_still_missing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    backend = FakeBackend([answer(tone="positive")] * 2)
    code(CODEBOOK, ["Invented."], backend=backend, ledger=ledger, recipe="per_unit")
    assert [e.outcome for e in ledger.entries()] == ["asked_again", "missing"]


def test_a_budget_stop_keeps_the_texts_already_finished(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", budget=Budget(call_limit=2))
    backend = FakeBackend([by_text_answer("n1"), answer(tone="positive"), by_text_answer("n2")])
    with pytest.raises(BudgetError) as caught:
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger, recipe="per_unit")
    run = caught.value.run
    assert isinstance(run, CodingRun) and run.recipe == "per_unit"
    assert run.table() == [{"unit": "n1", "tone": "positive", "topic": "transport"}]
    assert len(backend.calls) == 2


def test_a_failed_call_is_recorded_with_every_variable_named(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    failure = BackendError("no server", reason="no_server")
    with pytest.raises(BackendError):
        code(
            CODEBOOK,
            ["Invented."],
            backend=FakeBackend([failure]),
            ledger=ledger,
            recipe="per_unit",
        )
    (entry,) = ledger.entries()
    assert (entry.outcome, entry.variable, entry.error_reason) == (
        "failed",
        "tone+topic",
        "no_server",
    )


# The rest of reliacode works on a per-unit run as it is -----------------------------------------


def test_reliability_reads_a_per_unit_run() -> None:
    run = code(CODEBOOK, TEXTS, backend=FakeBackend(by_text), recipe="per_unit")
    human = [
        {"unit": unit, "coder": "anna", "variable": variable, "value": value}
        for unit, (tone, topic) in BOTH.items()
        for variable, value in (("tone", tone), ("topic", topic))
    ]
    codes = read_codes(CODEBOOK, human, run=run)
    assert codes.coders == ("anna", "model")
    result = codebook_reliability(CODEBOOK, codes, resamples=0)
    assert [v.alpha.percent_agreement for v in result.variables] == [100.0, 100.0]


def test_per_unit_runs_end_to_end_through_the_http_backend() -> None:
    server = FakeServer(
        httpx.Response(200, json=completion(answer(tone="negative", topic="schools")))
    )
    backend = OpenAICompatibleBackend(
        "http://localhost:8000/v1",
        "gpt-oss-120b",
        transport=server.transport,
        temperature=None,
        extra_body={"reasoning_effort": "low"},
    )
    run = code(CODEBOOK, [TEXTS["n2"]], backend=backend, recipe="per_unit")
    body = server.body()
    assert (
        body["response_format"]["json_schema"]["schema"] == build_unit_prompt(CODEBOOK).json_schema
    )
    assert "temperature" not in body and body["reasoning_effort"] == "low"
    assert run.table() == [{"unit": 0, "tone": "negative", "topic": "schools"}]
    parameters = run.results[0].parameters
    assert parameters.structured_output is True
    assert parameters.temperature is None
    assert parameters.extra_body == {"reasoning_effort": "low"}


def test_the_record_says_when_no_schema_was_sent() -> None:
    server = FakeServer(
        httpx.Response(200, json=completion(answer(tone="negative", topic="schools")))
    )
    backend = OpenAICompatibleBackend(
        "http://localhost:8000/v1", "m", transport=server.transport, structured_output=False
    )
    run = code(CODEBOOK, [TEXTS["n2"]], backend=backend, recipe="per_unit")
    assert "response_format" not in server.body()
    assert run.results[0].parameters.structured_output is False

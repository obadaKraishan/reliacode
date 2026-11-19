"""code() with a ledger: every call is recorded, second asks and failures included, and the budget
stops a run before a limit is passed, keeping the codes already finished.

Every model name, price and sentence in these tests is invented.
"""

import json
import logging
import secrets
from pathlib import Path

import httpx
import pytest

from helpers import FakeServer, TickingClock, completion, yaml_text
from reliacode import (
    Budget,
    BudgetError,
    FakeBackend,
    Ledger,
    ModelPrice,
    OpenAICompatibleBackend,
    PriceTable,
    code,
    parse_codebook,
    read_ledger,
    summarize,
)
from reliacode.backends import BackendError
from reliacode.ledger import hash_text

CODEBOOK = parse_codebook(
    yaml_text(
        """
        name: council-demo
        version: "1.0"
        variables:
          - name: tone
            definition: The overall tone of the text toward its main subject.
            categories:
              - code: positive
                definition: The text mostly praises or supports its subject.
              - code: negative
                definition: The text mostly criticizes or opposes its subject.
          - name: topic
            definition: The main topic of the text.
            categories:
              - code: transport
                definition: Buses, roads, bridges, ferries or trains.
              - code: schools
                definition: Schools, teachers or pupils.
        """
    )
)
TEXTS = [
    "The new bus line finally reaches the hospital.",
    "Parents praised the school's longer library hours.",
    "The ferry timetable changed again without notice.",
]
PRICES = PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={
        "fake-model": ModelPrice(input_per_million=1000, output_per_million=1000),
        "example-local-model": ModelPrice(input_per_million=0, output_per_million=0),
    },
)


def answer(value: str) -> str:
    return json.dumps({"code": value})


def ledger_at(tmp_path: Path, **settings: object) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", clock=TickingClock(), **settings)  # type: ignore[arg-type]


def lines(tmp_path: Path) -> list[dict[str, object]]:
    text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


# Every call is recorded --------------------------------------------------------------------------


def test_every_call_is_recorded_with_second_asks(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            answer("positive"),  # text 0, tone
            "transport",  # text 0, topic: not JSON, so asked again
            answer("transport"),
            answer("positive"),  # text 1, tone
            answer("schools"),  # text 1, topic
            answer("neutral"),  # text 2, tone: not a code, asked again
            answer("unsure"),  # still not a code: missing
            answer("transport"),  # text 2, topic
        ],
        tokens=(90, 10),
    )
    ledger = ledger_at(tmp_path)
    run = code(CODEBOOK, TEXTS, backend=backend, ledger=ledger)
    recorded = read_ledger(tmp_path / "ledger.jsonl")
    assert len(recorded) == len(backend.calls) == 8
    assert [e.outcome for e in recorded] == [
        "answered",
        "asked_again",
        "answered",
        "answered",
        "answered",
        "asked_again",
        "missing",
        "answered",
    ]
    assert [e.ask for e in recorded] == [1, 1, 2, 1, 1, 1, 2, 1]
    variables = ["tone", "topic", "topic", "tone", "topic", "tone", "tone", "topic"]
    assert [e.variable for e in recorded] == variables
    assert [e.text_hash for e in recorded[:3]] == [hash_text(TEXTS[0])] * 3
    assert {e.text_length for e in recorded[5:]} == {len(TEXTS[2])}
    prompt_hashes = {prompt.variable: prompt.hash for prompt in run.prompts}
    assert all(e.prompt_hash == prompt_hashes[e.variable] for e in recorded)
    assert {e.codebook_hash for e in recorded} == {run.codebook_hash}
    assert all(e.model_requested == "fake-model" for e in recorded)
    assert all((e.prompt_tokens, e.completion_tokens) == (90, 10) for e in recorded)
    assert [e.at for e in recorded] == sorted({e.at for e in recorded})  # the ledger's clock
    assert ledger.calls == 8
    assert run.table()[2] == {"unit": 2, "tone": None, "topic": "transport"}


def test_a_failed_call_is_recorded_before_the_run_stops(tmp_path: Path) -> None:
    down = BackendError("The server is down.", reason="no_server", endpoint="fake")
    backend = FakeBackend([answer("positive"), answer("transport"), down])
    with pytest.raises(BackendError):
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger_at(tmp_path))
    recorded = read_ledger(tmp_path / "ledger.jsonl")
    assert [e.outcome for e in recorded] == ["answered", "answered", "failed"]
    failed = recorded[-1]
    assert (failed.error_reason, failed.variable, failed.ask) == ("no_server", "tone", 1)
    assert failed.text_hash == hash_text(TEXTS[1])
    assert failed.model_requested == "fake-model"


def test_a_backend_that_returns_no_response_is_recorded_as_failed(tmp_path: Path) -> None:
    class Broken:
        def complete(self, messages: object, *, json_schema: object = None) -> object:
            return "positive"

    with pytest.raises(TypeError, match="should return a reliacode Response"):
        code(CODEBOOK, TEXTS, backend=Broken(), ledger=ledger_at(tmp_path))  # type: ignore[arg-type]
    (failed,) = read_ledger(tmp_path / "ledger.jsonl")
    assert (failed.outcome, failed.error_reason, failed.model_requested) == (
        "failed",
        "TypeError",
        None,
    )


def test_token_counts_come_from_the_servers_usage_field(tmp_path: Path) -> None:
    no_usage = completion(answer("positive"))
    del no_usage["usage"]
    server = FakeServer(
        httpx.Response(200, json=completion(answer("negative"))),
        httpx.Response(200, json=no_usage),
    )
    backend = OpenAICompatibleBackend(
        "http://localhost:11434/v1", "example-local-model", transport=server.transport
    )
    ledger = ledger_at(tmp_path, prices=PRICES)
    code(CODEBOOK, TEXTS[:2], backend=backend, ledger=ledger, variables="tone")
    counted, uncounted = ledger.entries()
    assert (counted.prompt_tokens, counted.completion_tokens) == (42, 6)
    assert counted.token_source == "server"
    assert counted.model_returned == "tiny-model-served"
    assert (counted.response_id, counted.attempts) == ("chatcmpl-test-1", 1)
    assert counted.parameters is not None
    assert counted.parameters.endpoint == "http://localhost:11434/v1/chat/completions"
    assert (uncounted.prompt_tokens, uncounted.token_source) == (None, None)
    total = ledger.summary().total
    assert (total.input_tokens, total.calls_without_tokens) == (42, 1)
    assert (total.cost, total.calls_without_cost) == (0, 1)  # a local model at price 0


def test_costs_can_be_worked_out_again_from_the_file(tmp_path: Path) -> None:
    tokens = [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50), (600, 60)]
    backend = FakeBackend(
        [answer("positive"), answer("transport")] * 3, tokens=tokens, model="fake-model"
    )
    code(CODEBOOK, TEXTS, backend=backend, ledger=ledger_at(tmp_path, prices=PRICES))
    by_hand = sum(p * 1000 + c * 1000 for p, c in tokens) / 1_000_000
    again = summarize(read_ledger(tmp_path / "ledger.jsonl"), PRICES)
    assert again.total.cost == pytest.approx(by_hand)
    assert again.total.input_tokens == 2100
    assert again.by_variable["topic"].output_tokens == 20 + 40 + 60


def test_without_a_ledger_nothing_is_recorded(tmp_path: Path) -> None:
    backend = FakeBackend([answer("positive"), answer("transport")])
    run = code(CODEBOOK, TEXTS[:1], backend=backend)
    assert run.table() == [{"unit": 0, "tone": "positive", "topic": "transport"}]
    assert list(tmp_path.iterdir()) == []


def test_the_ledger_must_be_a_ledger(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"ledger should be a reliacode Ledger.*the text"):
        code(CODEBOOK, TEXTS, backend=FakeBackend([]), ledger="ledger.jsonl")  # type: ignore[arg-type]


# The budget stops a run --------------------------------------------------------------------------


def test_a_run_with_a_token_cap_stops_before_the_cap_is_crossed(tmp_path: Path) -> None:
    backend = FakeBackend(lambda messages: answer("positive"), tokens=(90, 10))
    ledger = ledger_at(tmp_path, budget=Budget(token_limit=450))
    texts = [f"Invented sentence number {n}." for n in range(10)]
    with pytest.raises(BudgetError) as caught:
        code(CODEBOOK, texts, backend=backend, ledger=ledger, variables="tone")
    used = sum(e.prompt_tokens + e.completion_tokens for e in ledger.entries())  # type: ignore[operator]
    assert used == 400 <= 450  # the cap was never crossed
    assert len(backend.calls) == ledger.calls == 4  # the fifth call was never made
    assert "The next call could pass the token limit of 450" in str(caught.value)


def test_the_codes_finished_before_the_stop_come_with_the_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        [answer("positive"), answer("transport"), answer("negative"), answer("schools")],
        tokens=(90, 10),
    )
    ledger = ledger_at(tmp_path, budget=Budget(call_limit=3))
    with pytest.raises(BudgetError) as caught:
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger)
    run = caught.value.run
    assert run is not None
    assert run.table() == [
        {"unit": 0, "tone": "positive", "topic": "transport"},
        {"unit": 1, "tone": "negative"},
    ]
    assert run.codebook_hash == CODEBOOK.content_hash()
    assert len(run.prompts) == 2
    assert run.finished_at >= run.started_at


def test_a_result_still_being_coded_when_the_budget_stops_is_left_out(tmp_path: Path) -> None:
    backend = FakeBackend([answer("positive"), "not json", answer("transport")], tokens=(90, 10))
    ledger = ledger_at(tmp_path, budget=Budget(call_limit=2))
    with pytest.raises(BudgetError, match="allows 2 calls") as caught:
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger)
    assert caught.value.run.table() == [{"unit": 0, "tone": "positive"}]
    assert [e.outcome for e in ledger.entries()] == ["answered", "asked_again"]


def test_a_run_with_a_cost_cap_stops_before_the_cap_is_crossed(tmp_path: Path) -> None:
    # At the invented price of 1,000 USD per million tokens, each call of 100 tokens costs 0.10.
    backend = FakeBackend(lambda messages: answer("positive"), tokens=(90, 10))
    ledger = ledger_at(tmp_path, prices=PRICES, budget=Budget(cost_limit=0.25))
    with pytest.raises(BudgetError) as caught:
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger, variables="tone")
    assert ledger.summary().total.cost == pytest.approx(0.20)
    assert caught.value.limit == "cost"
    assert len(caught.value.run.results) == 2


def test_a_cost_cap_for_a_model_with_no_price_is_refused_before_any_call(tmp_path: Path) -> None:
    backend = FakeBackend([answer("positive")], model="example-other-model", tokens=(90, 10))
    ledger = ledger_at(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    with pytest.raises(BudgetError, match='no price for the model "example-other-model"') as caught:
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger)
    assert backend.calls == []
    assert caught.value.run is None
    assert ledger.calls == 0


def test_a_token_cap_with_a_backend_that_reports_no_counts_stops_after_one_call(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(lambda messages: answer("positive"))  # no token counts
    ledger = ledger_at(tmp_path, budget=Budget(token_limit=10_000))
    with pytest.raises(BudgetError, match=r"cannot be kept.*FakeBackend reports token counts"):
        code(CODEBOOK, TEXTS, backend=backend, ledger=ledger)
    assert len(backend.calls) == 1


def test_the_budget_carries_over_to_the_next_run_on_the_same_file(tmp_path: Path) -> None:
    budget = Budget(call_limit=8)
    backend = FakeBackend(lambda messages: answer("positive"), tokens=(90, 10))
    first = ledger_at(tmp_path, budget=budget)
    code(CODEBOOK, TEXTS, backend=backend, ledger=first, variables="tone")  # 3 calls
    again = ledger_at(tmp_path, budget=budget)
    with pytest.raises(BudgetError, match="already has 8") as caught:
        code(CODEBOOK, TEXTS * 3, backend=backend, ledger=again, variables="tone")
    assert len(caught.value.run.results) == 5  # the 5 calls left of the 8 allowed


# Privacy, end to end -----------------------------------------------------------------------------


KEY_ENV = "RELIACODE_TEST_API_KEY"


def test_no_text_answer_or_key_reaches_the_ledger_from_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    key = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv(KEY_ENV, key)
    text_mark, answer_mark = secrets.token_hex(12), secrets.token_hex(12)
    texts = [f"A private message that mentions {text_mark} by name."]
    usable = json.dumps({"code": "positive", "note": answer_mark})
    unusable = f"I am not sure about {answer_mark}."
    echo = {"error": {"message": f"Cannot read {texts[0]} with key {key}"}}
    server = FakeServer(
        httpx.Response(200, json=completion(usable)),  # tone
        httpx.Response(200, json=completion(unusable)),  # topic: asked again
        httpx.Response(400, json=echo),  # then the server fails, echoing the text and key
    )
    backend = OpenAICompatibleBackend(
        "https://api.example.org/v1", "fake-model", api_key_env=KEY_ENV, transport=server.transport
    )
    ledger = ledger_at(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    with pytest.raises(BackendError):
        code(CODEBOOK, texts, backend=backend, ledger=ledger)
    assert [e.outcome for e in ledger.entries()] == ["answered", "asked_again", "failed"]
    raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    summary = ledger.summary()
    shown = [raw, repr(ledger), repr(ledger.entries()), summary.model_dump_json(), repr(summary)]
    for mark in (text_mark, answer_mark, key):
        assert all(mark not in part for part in shown)
    logged = [f"{r.getMessage()} {r.args!r}" for r in caplog.records]
    assert all(text_mark not in line and key not in line for line in logged)
    ledger_logs = [r for r in caplog.records if r.name == "reliacode.ledger"]
    assert ledger_logs == []  # the ledger itself writes no log line


def test_a_budget_stop_shows_no_text(tmp_path: Path) -> None:
    mark = secrets.token_hex(12)
    backend = FakeBackend(lambda messages: answer("positive"), tokens=(90, 10))
    ledger = ledger_at(tmp_path, budget=Budget(call_limit=1))
    with pytest.raises(BudgetError) as caught:
        code(CODEBOOK, [f"Secret {mark}.", "Other."], backend=backend, ledger=ledger)
    shown = f"{caught.value} {caught.value!r} {vars(caught.value)}"
    assert mark not in shown.replace(repr(caught.value.run), "")
    assert mark not in str(caught.value)

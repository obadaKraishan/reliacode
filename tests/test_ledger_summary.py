"""The summary: totals in all, by model and by variable, that say what they could not count, with
the times, the hashes and the prices they come from. Exported as CSV and JSON.

Every model name and price in these tests is invented.
"""

import csv
import json
from datetime import timedelta
from pathlib import Path

import pytest

from helpers import CODEBOOK_HASH, NOON, PROMPT_HASH, response
from reliacode.backends import BackendError
from reliacode.ledger import (
    UNKNOWN_MODEL,
    Ledger,
    LedgerEntry,
    LedgerSummary,
    ModelPrice,
    PriceTable,
    Totals,
    summarize,
)

OTHER_CODEBOOK = "sha256:" + "e" * 64
PRICES = PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={
        "example-hosted-model": ModelPrice(input_per_million=1000, output_per_million=2000),
        "example-local-model": ModelPrice(input_per_million=0, output_per_million=0),
    },
)


def line(
    minute: int,
    *,
    text: str = "Text one.",
    variable: str = "tone",
    model: str = "example-hosted-model",
    served_as: str | None = None,
    tokens: tuple[int | None, int | None] = (100, 10),
    outcome: str = "answered",
    attempts: int = 1,
    codebook_hash: str = CODEBOOK_HASH,
) -> LedgerEntry:
    reply = response(
        model=model, prompt_tokens=tokens[0], completion_tokens=tokens[1], attempts=attempts
    )
    if served_as is not None:
        reply = reply.model_copy(update={"model": served_as})
    return LedgerEntry.from_response(
        reply,
        outcome=outcome,  # type: ignore[arg-type]
        variable=variable,
        codebook_hash=codebook_hash,
        prompt_hash=PROMPT_HASH,
        text=text,
        at=NOON + timedelta(minutes=minute),
    )


def failed(minute: int, *, text: str = "Text two.", variable: str = "topic") -> LedgerEntry:
    return LedgerEntry.from_failure(
        BackendError("Timed out.", reason="timeout"),
        model="example-hosted-model",
        variable=variable,
        codebook_hash=CODEBOOK_HASH,
        prompt_hash=PROMPT_HASH,
        text=text,
        at=NOON + timedelta(minutes=minute),
    )


#: An invented run: two texts, two variables, two models, a second ask, a failure, an answer
#: with no token counts and a model with no price.
LINES = [
    line(0, text="Text one.", variable="tone", attempts=2, served_as="example-hosted-model-0613"),
    line(1, text="Text one.", variable="topic", outcome="asked_again", tokens=(100, 30)),
    line(2, text="Text one.", variable="topic", tokens=(150, 10)),
    line(3, text="Text two.", variable="tone", model="example-local-model", tokens=(400, 5)),
    failed(4),
    line(5, text="Text two.", variable="topic", outcome="missing", tokens=(None, None)),
    line(6, text="Text two.", variable="tone", model="example-other-model", tokens=(80, 2)),
]


def test_the_totals_add_up_every_call() -> None:
    total = summarize(LINES, PRICES).total
    assert (total.calls, total.texts) == (7, 2)
    assert (total.answered, total.asked_again, total.missing, total.failed) == (4, 1, 1, 1)
    assert total.attempts == 7  # 2 + 1 + 1 + 1 + 1 + 1; the failed call's tries are not known
    assert (total.input_tokens, total.output_tokens) == (830, 57)
    assert total.calls_without_tokens == 2  # the failed call and the answer with no counts
    # hosted: (100+100+150) input at 1000/M and (10+30+10) output at 2000/M; local: 0
    assert total.cost == pytest.approx((350 * 1000 + 50 * 2000) / 1_000_000)
    assert total.calls_without_cost == 3  # two without counts and one model without a price
    assert total.latency_seconds == pytest.approx(6 * 0.25)
    assert total.calls_per_text == 3.5


def test_missing_counts_never_become_zero_and_averages_are_not_guessed() -> None:
    total = summarize(LINES, PRICES).total
    assert total.input_tokens_per_text is None  # two calls could not be counted
    assert total.output_tokens_per_text is None
    assert total.cost_per_text is None
    nothing_counted = summarize([line(0, tokens=(None, None))], PRICES).total
    assert (nothing_counted.input_tokens, nothing_counted.output_tokens) == (None, None)
    assert nothing_counted.cost is None
    assert nothing_counted.calls_without_tokens == 1


def test_averages_are_given_when_every_call_is_counted() -> None:
    complete = [
        line(0, text="A.", tokens=(100, 10)),
        line(1, text="A.", tokens=(120, 10), variable="topic"),
        line(2, text="B.", tokens=(140, 10)),
        line(3, text="B.", tokens=(160, 10), variable="topic"),
    ]
    total = summarize(complete, PRICES).total
    assert total.input_tokens_per_text == 260
    assert total.output_tokens_per_text == 20
    assert total.cost_per_text == pytest.approx((520 * 1000 + 40 * 2000) / 1_000_000 / 2)
    assert total.calls_per_text == 2


def test_the_totals_by_model_use_the_name_asked_for() -> None:
    summary = summarize(LINES, PRICES)
    assert list(summary.by_model) == [
        "example-hosted-model",
        "example-local-model",
        "example-other-model",
    ]
    hosted = summary.by_model["example-hosted-model"]
    assert (hosted.calls, hosted.failed, hosted.calls_without_tokens) == (5, 1, 2)
    assert (hosted.input_tokens, hosted.output_tokens) == (350, 50)
    local = summary.by_model["example-local-model"]
    assert (local.cost, local.input_tokens, local.cost_per_text) == (0, 400, 0)
    other = summary.by_model["example-other-model"]
    assert (other.input_tokens, other.cost, other.calls_without_cost) == (80, None, 1)
    assert summary.served_as == {
        "example-hosted-model": ("example-hosted-model-0613", "example-hosted-model"),
        "example-local-model": ("example-local-model",),
        "example-other-model": ("example-other-model",),
    }


def test_the_totals_by_variable() -> None:
    summary = summarize(LINES, PRICES)
    assert list(summary.by_variable) == ["tone", "topic"]
    tone, topic = summary.by_variable["tone"], summary.by_variable["topic"]
    assert (tone.calls, tone.texts, tone.input_tokens) == (3, 2, 580)
    assert (topic.calls, topic.asked_again, topic.missing, topic.failed) == (4, 1, 1, 1)
    assert topic.input_tokens == 250


def test_a_call_with_no_model_is_summed_under_unknown() -> None:
    no_model = LedgerEntry.from_failure(
        BackendError("No server.", reason="no_server"),
        variable="tone",
        codebook_hash=CODEBOOK_HASH,
        prompt_hash=PROMPT_HASH,
        text="Text.",
        at=NOON,
    )
    summary = summarize([no_model])
    assert list(summary.by_model) == [UNKNOWN_MODEL]
    assert summary.served_as == {}


def test_the_times_hashes_and_prices_are_kept() -> None:
    lines = [*LINES, line(9, codebook_hash=OTHER_CODEBOOK)]
    summary = summarize(reversed(lines), PRICES)
    assert summary.first_at == NOON
    assert summary.last_at == NOON + timedelta(minutes=9)
    assert summary.codebook_hashes == (OTHER_CODEBOOK, CODEBOOK_HASH)  # in the order seen
    assert summary.prompt_hashes == (PROMPT_HASH,)
    assert summary.prices == PRICES


def test_without_prices_there_are_tokens_and_no_costs() -> None:
    total = summarize(LINES).total
    assert total.input_tokens == 830
    assert total.cost is None
    assert total.calls_without_cost == 7
    assert summarize(LINES).prices is None


def test_costs_follow_the_price_table_given() -> None:
    corrected = PRICES.model_copy(
        update={
            "prices": {
                **PRICES.prices,
                "example-other-model": ModelPrice(input_per_million=0, output_per_million=0),
            }
        }
    )
    before = summarize(LINES, PRICES).total
    after = summarize(LINES, corrected).total
    assert (before.calls_without_cost, after.calls_without_cost) == (3, 2)
    assert after.cost == pytest.approx(before.cost)


def test_an_empty_ledger_has_nothing_to_add_up() -> None:
    summary = summarize([], PRICES)
    total = summary.total
    assert (total.calls, total.texts, total.attempts) == (0, 0, 0)
    assert (total.input_tokens, total.cost, total.latency_seconds) == (None, None, None)
    assert total.calls_per_text is None
    assert (summary.first_at, summary.last_at, summary.codebook_hashes) == (None, None, ())


def test_summarize_takes_ledger_lines_and_a_price_table() -> None:
    with pytest.raises(TypeError, match="ledger lines"):
        summarize([{"outcome": "answered"}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="PriceTable"):
        summarize(LINES, {"currency": "USD"})  # type: ignore[arg-type]


def test_the_ledger_summarizes_its_whole_file_with_its_prices(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path)
    for entry in LINES[:3]:
        first.record(entry)
    second = Ledger(path, prices=PRICES)
    for entry in LINES[3:]:
        second.record(entry)
    assert second.summary() == summarize(LINES, PRICES)


# Exports -----------------------------------------------------------------------------------------


def test_the_rows_give_one_row_per_group() -> None:
    rows = summarize(LINES, PRICES).rows()
    assert [(row["group"], row["name"]) for row in rows] == [
        ("all", "all calls"),
        ("model", "example-hosted-model"),
        ("model", "example-local-model"),
        ("model", "example-other-model"),
        ("variable", "tone"),
        ("variable", "topic"),
    ]
    assert rows[0]["calls"] == 7
    assert rows[0]["currency"] == "USD"
    assert summarize(LINES).rows()[0]["currency"] is None


def test_the_summary_is_written_as_csv(tmp_path: Path) -> None:
    summarize(LINES, PRICES).to_csv(tmp_path / "summary.csv")
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == ["group", "name", *Totals.model_fields, "currency"]
    assert len(rows) == 6
    assert (rows[0]["calls"], rows[0]["input_tokens"], rows[0]["currency"]) == ("7", "830", "USD")
    assert rows[0]["input_tokens_per_text"] == ""  # not known, so left empty
    assert rows[3]["cost"] == ""


def test_the_summary_is_written_as_json_and_read_back(tmp_path: Path) -> None:
    summary = summarize(LINES, PRICES)
    summary.to_json(tmp_path / "summary.json")
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert json.loads(text)["total"]["calls"] == 7
    assert json.loads(text)["prices"]["read_on"] == "2026-11-19"
    assert LedgerSummary.model_validate_json(text) == summary

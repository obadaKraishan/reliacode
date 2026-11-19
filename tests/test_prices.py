"""Prices come from the user, never from the package, and a missing count or price stays missing.

Every model name and price in these tests is invented.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import CODEBOOK_HASH, PROMPT_HASH, response
from reliacode.backends import BackendError
from reliacode.ledger import Ledger, LedgerEntry, ModelPrice, PriceTable, read_ledger

TEXT = "The ferry now runs every hour on weekends."
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}
PRICES = PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={
        "example-hosted-model": ModelPrice(input_per_million=0.50, output_per_million=1.50),
        "example-local-model": ModelPrice(input_per_million=0, output_per_million=0),
    },
)


def call(
    model: str = "example-hosted-model",
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 10,
    served_as: str | None = None,
) -> LedgerEntry:
    reply = response(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    if served_as is not None:
        reply = reply.model_copy(update={"model": served_as})
    return LedgerEntry.from_response(reply, outcome="answered", text=TEXT, **WORK)


# The cost of a call ------------------------------------------------------------------------------


def test_a_call_costs_its_tokens_at_the_prices_per_million() -> None:
    assert PRICES.cost(call()) == pytest.approx((1000 * 0.50 + 10 * 1.50) / 1_000_000)
    assert PRICES.cost(call(prompt_tokens=1_000_000, completion_tokens=0)) == pytest.approx(0.50)
    assert PRICES.cost(call(prompt_tokens=0, completion_tokens=2_000_000)) == pytest.approx(3.00)


def test_a_local_model_at_price_zero_costs_zero_and_its_tokens_still_count() -> None:
    local = call("example-local-model", prompt_tokens=900, completion_tokens=7)
    assert PRICES.cost(local) == 0
    assert (local.prompt_tokens, local.completion_tokens) == (900, 7)
    assert local.counted


def test_a_model_with_no_price_has_tokens_and_no_cost() -> None:
    unpriced = call("example-other-model")
    assert PRICES.price_of(unpriced) is None
    assert PRICES.cost(unpriced) is None
    assert unpriced.prompt_tokens == 1000


@pytest.mark.parametrize(("prompt", "completion"), [(None, None), (1000, None), (None, 10)])
def test_a_call_without_both_counts_has_no_cost_never_zero(
    prompt: int | None, completion: int | None
) -> None:
    assert PRICES.cost(call(prompt_tokens=prompt, completion_tokens=completion)) is None


def test_a_failed_call_has_no_cost() -> None:
    error = BackendError("Timed out.", reason="timeout")
    failed = LedgerEntry.from_failure(error, model="example-hosted-model", text=TEXT, **WORK)
    assert PRICES.cost(failed) is None


def test_the_price_is_found_under_the_name_asked_for_then_the_name_returned() -> None:
    served = call("example-hosted-model", served_as="example-hosted-model-0613")
    assert PRICES.price_of(served) == PRICES.prices["example-hosted-model"]
    renamed = PriceTable(
        currency="USD",
        read_on="2026-11-19",
        prices={"example-hosted-model-0613": ModelPrice(input_per_million=2, output_per_million=8)},
    )
    assert renamed.price_of(served) == ModelPrice(input_per_million=2, output_per_million=8)
    both = PriceTable(
        currency="USD",
        read_on="2026-11-19",
        prices={**renamed.prices, **PRICES.prices},
    )
    assert both.price_of(served) == PRICES.prices["example-hosted-model"]  # asked-for name wins


def test_a_line_with_no_model_name_has_no_price() -> None:
    error = BackendError("No server.", reason="no_server")
    failed = LedgerEntry.from_failure(error, text=TEXT, **WORK)
    assert PRICES.price(None) is None
    assert PRICES.price_of(failed) is None


def test_costs_can_be_worked_out_again_from_the_file_and_the_table(tmp_path: Path) -> None:
    calls = [
        call(),
        call("example-local-model"),
        call("example-other-model"),
        call(prompt_tokens=None, completion_tokens=None),
    ]
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for entry in calls:
        ledger.record(entry)
    again = read_ledger(tmp_path / "ledger.jsonl")
    assert [PRICES.cost(entry) for entry in again] == [PRICES.cost(entry) for entry in calls]
    assert [PRICES.cost(entry) is None for entry in again] == [False, False, True, True]


# The table ---------------------------------------------------------------------------------------


def test_the_table_keeps_the_currency_and_the_date_the_prices_were_read() -> None:
    assert PRICES.currency == "USD"
    assert PRICES.read_on == date(2026, 11, 19)


def test_the_table_can_be_written_as_plain_data_or_read_from_json(tmp_path: Path) -> None:
    data = {
        "currency": "EUR",
        "read_on": "2026-11-18",
        "prices": {"example-hosted-model": {"input_per_million": 1, "output_per_million": 4}},
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    table = PriceTable.model_validate_json(path.read_text(encoding="utf-8"))
    assert table == PriceTable.model_validate(data)
    assert table.prices["example-hosted-model"].output_per_million == 4
    assert PriceTable.model_validate_json(table.model_dump_json()) == table


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"currency": "  "}, "give the currency"),
        ({"read_on": "last week"}, "read_on"),
        ({"prices": {}}, "lists no model"),
        ({"prices": {" ": {"input_per_million": 1, "output_per_million": 1}}}, "empty name"),
        ({"prices": {"m": {"input_per_million": -1, "output_per_million": 1}}}, "greater than"),
        ({"prices": {"m": {"input_per_million": 1, "output_per_million": "nan"}}}, "finite"),
        ({"prices": {"m": {"input_per_million": float("inf"), "output_per_million": 1}}}, "finite"),
        ({"prices": {"m": {"input_per_million": 1}}}, "output_per_million"),
        ({"extra": 1}, "Extra inputs"),
    ],
)
def test_a_table_that_cannot_be_used_is_refused(changes: dict[str, object], message: str) -> None:
    data = {**PRICES.model_dump(mode="json"), **changes}
    with pytest.raises(ValidationError, match=message):
        PriceTable.model_validate(data)


def test_the_currency_is_trimmed() -> None:
    table = PRICES.model_copy(update={"currency": " USD "})
    assert PriceTable.model_validate(table.model_dump()).currency == "USD"


def test_the_table_is_frozen() -> None:
    with pytest.raises(ValidationError):
        PRICES.currency = "EUR"  # type: ignore[misc]

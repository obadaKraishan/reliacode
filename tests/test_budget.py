"""The budget: a limit on cost, tokens or calls, checked before each call. The call that would
pass a limit is refused in plain words, and a limit that cannot be kept is refused too.

Every model name and price in these tests is invented.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import CODEBOOK_HASH, PROMPT_HASH, TickingClock, response
from reliacode.backends import BackendError, RequestParameters
from reliacode.ledger import Budget, BudgetError, Ledger, LedgerEntry, ModelPrice, PriceTable

TEXT = "The school board met for three hours on Tuesday."
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}
PRICES = PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={
        "example-hosted-model": ModelPrice(input_per_million=1000, output_per_million=1000),
        "example-local-model": ModelPrice(input_per_million=0, output_per_million=0),
    },
)


def open_ledger(path: Path, **settings: object) -> Ledger:
    return Ledger(path / "ledger.jsonl", clock=TickingClock(), **settings)  # type: ignore[arg-type]


def call(
    ledger: Ledger,
    prompt_tokens: int | None = 90,
    completion_tokens: int | None = 10,
    model: str = "example-hosted-model",
    **fields: object,
) -> None:
    """Check the budget, then record one answered call, as code() does."""
    ledger.check()
    reply = response(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, **fields
    )
    entry = LedgerEntry.from_response(
        reply, outcome="answered", text=TEXT, at=ledger.clock(), **WORK
    )
    ledger.record(entry)


def failed_call(ledger: Ledger) -> None:
    ledger.check()
    error = BackendError("No server.", reason="no_server")
    entry = LedgerEntry.from_failure(
        error, model="example-hosted-model", text=TEXT, at=ledger.clock(), **WORK
    )
    ledger.record(entry)


def used_tokens(ledger: Ledger) -> int:
    return sum((e.prompt_tokens or 0) + (e.completion_tokens or 0) for e in ledger.entries())


# The budget --------------------------------------------------------------------------------------


def test_a_budget_needs_a_limit() -> None:
    with pytest.raises(ValidationError, match="needs at least one limit"):
        Budget()


@pytest.mark.parametrize(
    "limits",
    [{"cost_limit": 0}, {"cost_limit": -1}, {"token_limit": 0}, {"call_limit": -5}],
)
def test_a_limit_must_be_more_than_zero(limits: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Budget(**limits)


def test_a_cost_limit_must_be_a_number() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Budget(cost_limit=float("inf"))


def test_the_limits_can_be_combined() -> None:
    budget = Budget(cost_limit=2.5, token_limit=10_000, call_limit=100)
    assert (budget.cost_limit, budget.token_limit, budget.call_limit) == (2.5, 10_000, 100)


def test_a_cost_limit_needs_prices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has a cost limit, but the ledger has no prices"):
        open_ledger(tmp_path, budget=Budget(cost_limit=1))


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"prices": {"m": 1}}, "PriceTable"), ({"budget": {"call_limit": 3}}, "Budget")],
)
def test_prices_and_budget_must_be_the_right_kind(
    tmp_path: Path, settings: dict[str, object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        open_ledger(tmp_path, **settings)


def test_without_a_budget_no_call_is_refused(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES)
    for _ in range(50):
        call(ledger, 10_000, 10_000)
    ledger.check()
    assert ledger.calls == 50


# The call limit ----------------------------------------------------------------------------------


def test_the_call_that_would_pass_the_call_limit_is_refused(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(call_limit=3))
    for _ in range(3):
        call(ledger)
    with pytest.raises(BudgetError) as caught:
        call(ledger)
    error = caught.value
    assert str(error) == (
        f"The budget allows 3 calls, and the ledger {ledger.path} already has 3, so reliacode "
        "stopped before the next call. To make more calls, raise call_limit in the Budget."
    )
    assert (error.limit, error.allowed, error.used, error.run) == ("calls", 3, 3, None)
    assert ledger.calls == 3  # the refused call was not made and not recorded


def test_failed_calls_count_toward_the_call_limit(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(call_limit=2))
    failed_call(ledger)
    call(ledger)
    with pytest.raises(BudgetError, match="allows 2 calls"):
        ledger.check()


def test_the_budget_covers_the_calls_already_in_the_file(tmp_path: Path) -> None:
    first = open_ledger(tmp_path)
    for _ in range(4):
        call(first)
    again = open_ledger(tmp_path, budget=Budget(call_limit=5))
    call(again)
    with pytest.raises(BudgetError, match="already has 5"):
        call(again)


# The token limit ---------------------------------------------------------------------------------


def test_the_call_that_could_pass_the_token_limit_is_refused(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=450))
    for _ in range(4):
        call(ledger, 90, 10)  # 100 tokens each
    with pytest.raises(BudgetError) as caught:
        call(ledger, 90, 10)
    assert str(caught.value) == (
        f"The next call could pass the token limit of 450, so reliacode stopped before it. The "
        f"calls in the ledger {ledger.path} have used 400 tokens (input and output together), "
        "and the largest one used 100. To go on, raise token_limit in the Budget."
    )
    assert (caught.value.limit, caught.value.allowed, caught.value.used) == ("tokens", 450, 400)
    assert used_tokens(ledger) == 400 <= 450  # the limit was never passed


def test_the_token_limit_can_be_reached_but_not_passed(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=400))
    for _ in range(4):
        call(ledger, 90, 10)
    assert used_tokens(ledger) == 400
    with pytest.raises(BudgetError, match="have used 400 tokens"):
        ledger.check()


def test_the_largest_call_so_far_sets_the_size_of_the_next(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=800))
    call(ledger, 90, 10)
    call(ledger, 290, 10)  # 300 tokens: the largest so far
    call(ledger, 90, 10)
    assert used_tokens(ledger) == 500
    call(ledger, 90, 10)  # 500 + 300 <= 800
    with pytest.raises(BudgetError, match=r"have used 600 tokens.*largest one used 300"):
        call(ledger)  # 600 + 300 > 800


def test_the_first_call_is_made_and_a_larger_one_is_then_stopped(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=50))
    call(ledger, 90, 10)  # the first call's size cannot be known, so it is made
    with pytest.raises(BudgetError, match="have used 100 tokens"):
        call(ledger)


def test_a_call_without_token_counts_makes_a_token_limit_impossible_to_keep(
    tmp_path: Path,
) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=10_000))
    call(ledger)
    call(ledger, None, None)
    with pytest.raises(BudgetError) as caught:
        call(ledger)
    assert str(caught.value) == (
        "The token limit of 10,000 cannot be kept: the server reported no token counts for call "
        f'2 in the ledger {ledger.path} (model "example-hosted-model"), so reliacode cannot '
        "tell what the calls use. Use a call limit instead, as in Budget(call_limit=1000), or a "
        "server that reports token counts."
    )
    assert (caught.value.limit, caught.value.used) == ("tokens", 100)


def test_the_message_says_how_to_give_fake_backend_token_counts(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=10_000))
    fake = RequestParameters(endpoint="fake", model="fake-model")
    call(ledger, None, None, model="fake-model", parameters=fake)
    with pytest.raises(BudgetError, match=r"FakeBackend reports token counts only when it is"):
        ledger.check()


def test_a_failed_call_does_not_make_a_token_limit_impossible_to_keep(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=1000))
    failed_call(ledger)
    call(ledger)
    ledger.check()


def test_a_call_with_one_count_cannot_be_counted(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, budget=Budget(token_limit=1000))
    call(ledger, 90, None)
    with pytest.raises(BudgetError, match="cannot be kept"):
        ledger.check()


# The cost limit ----------------------------------------------------------------------------------


def test_the_call_that_could_pass_the_cost_limit_is_refused(tmp_path: Path) -> None:
    # At the invented price of 1,000 USD per million tokens, a call of 100 tokens costs 0.10 USD.
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=0.35))
    for _ in range(3):
        call(ledger, 90, 10)
    with pytest.raises(BudgetError) as caught:
        call(ledger, 90, 10)
    assert str(caught.value) == (
        "The next call could pass the cost limit of 0.35 USD, so reliacode stopped before it. "
        f"The calls in the ledger {ledger.path} have cost 0.30 USD so far, and the largest one "
        "cost 0.10 USD. To go on, raise cost_limit in the Budget."
    )
    assert caught.value.limit == "cost"
    assert caught.value.allowed == 0.35
    assert caught.value.used == pytest.approx(0.30)


def test_small_costs_are_written_with_the_decimals_they_need(tmp_path: Path) -> None:
    cheap = PriceTable(
        currency="EUR",
        read_on="2026-11-19",
        prices={"example-hosted-model": ModelPrice(input_per_million=0.4, output_per_million=1.6)},
    )
    ledger = open_ledger(tmp_path, prices=cheap, budget=Budget(cost_limit=0.0001))
    call(ledger, 90, 10)  # 0.000052 EUR
    with pytest.raises(BudgetError) as caught:
        call(ledger, 90, 10)
    assert "cost limit of 0.0001 EUR" in str(caught.value)
    assert "have cost 0.000052 EUR so far, and the largest one cost 0.000052 EUR" in str(
        caught.value
    )


def test_a_local_model_at_price_zero_never_reaches_a_cost_limit(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=0.01))
    for _ in range(20):
        call(ledger, 5000, 500, model="example-local-model")
    ledger.check()


def test_a_call_to_a_model_with_no_price_makes_a_cost_limit_impossible_to_keep(
    tmp_path: Path,
) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    call(ledger, model="example-other-model")
    with pytest.raises(BudgetError) as caught:
        call(ledger)
    assert str(caught.value) == (
        "The cost limit of 5.00 USD cannot be kept: the price table has no price for the model "
        f'"example-other-model" (call 1 in the ledger {ledger.path}), so the cost of its calls '
        "is not known. Add the model to the price table (write 0 for a local model that costs "
        "nothing), or use a token or call limit instead."
    )


def test_a_call_without_token_counts_makes_a_cost_limit_impossible_to_keep(
    tmp_path: Path,
) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    call(ledger, None, None)
    with pytest.raises(BudgetError, match=r"The cost limit of 5\.00 USD cannot be kept: the serv"):
        ledger.check()


def test_a_cost_limit_with_an_unpriced_model_is_refused_at_the_start(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    with pytest.raises(BudgetError) as caught:
        ledger.check_price("example-other-model")
    assert str(caught.value) == (
        "The cost limit of 5.00 USD cannot be kept: the price table has no price for the model "
        '"example-other-model", so the cost of its calls is not known. Add the model to the '
        "price table (write 0 for a local model that costs nothing), or use a token or call "
        "limit instead."
    )
    assert ledger.calls == 0


def test_the_start_check_passes_a_priced_or_unknown_model_or_no_cost_limit(tmp_path: Path) -> None:
    ledger = open_ledger(tmp_path, prices=PRICES, budget=Budget(cost_limit=5))
    ledger.check_price("example-hosted-model")
    ledger.check_price(None)
    open_ledger(tmp_path, prices=PRICES, budget=Budget(call_limit=5)).check_price("other")
    open_ledger(tmp_path).check_price("other")


def test_every_limit_is_checked(tmp_path: Path) -> None:
    ledger = open_ledger(
        tmp_path, prices=PRICES, budget=Budget(cost_limit=100, token_limit=250, call_limit=10)
    )
    call(ledger)
    call(ledger)
    with pytest.raises(BudgetError, match="token limit of 250"):
        call(ledger)

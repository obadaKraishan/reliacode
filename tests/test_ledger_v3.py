"""Ledger version 3: the cache counts and the batch ids are recorded, lines of versions 1 and 2
are still read, and the price table prices cache writes, cache reads, batches and long prompts
exactly. All model names, texts and prices are invented unless a test says where they are from."""

import json
from pathlib import Path

import httpx
import pytest

from helpers import (
    CODEBOOK_HASH,
    EXAMPLES,
    FIXTURES,
    NOON,
    PROMPT_HASH,
    FakeServer,
    completion,
    response,
)
from reliacode import (
    Budget,
    BudgetError,
    FakeBackend,
    Ledger,
    LedgerEntry,
    ModelPrice,
    OpenAICompatibleBackend,
    PriceTable,
    RequestParameters,
    code,
    load_codebook,
    read_ledger,
    summarize,
)
from reliacode.backends import BackendError
from reliacode.ledger import LEDGER_VERSION, Totals

WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}
V1_FILE = FIXTURES / "ledger_v1.jsonl"
V2_FILE = FIXTURES / "ledger_v2.jsonl"
NEW_FIELDS = ["cache_write_tokens", "cache_write_1h_tokens", "cache_read_tokens", "batch_id"]
NEW_FIELDS.append("custom_id")


def cached(cache: str | None = "1h", **fields: object):
    """An answer from a backend that cached its system prompt, with invented counts."""
    parameters = RequestParameters(
        endpoint="https://api.anthropic.com/v1/messages",
        model="example-cached-model",
        max_tokens=4000,
        structured_output=True,
        api_version="2026-10-21",
        cache=cache,
    )
    values: dict[str, object] = {
        "model": "example-cached-model",
        "parameters": parameters,
        "prompt_tokens": 2_000,
        "completion_tokens": 100,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
    }
    values.update(fields)
    return response(**values)


def line(answer, **extra: object) -> LedgerEntry:
    return LedgerEntry.from_response(answer, outcome="answered", text="Invented.", **WORK, **extra)


PRICE = ModelPrice(
    input_per_million=1.0,
    output_per_million=5.0,
    cache_write_5m_per_million=1.25,
    cache_write_1h_per_million=2.0,
    cache_read_per_million=0.1,
    batch_discount=0.5,
)


# The version and its fields -------------------------------------------------------------------


def test_the_new_fields_come_last_in_a_line() -> None:
    assert LEDGER_VERSION == 3
    fields = list(LedgerEntry.model_fields)
    assert fields[-6:] == ["reasoning_tokens", *NEW_FIELDS]
    assert fields.index("text_length") < fields.index("reasoning_tokens")


def test_a_line_keeps_the_cache_counts_and_the_batch_ids() -> None:
    answer = cached(
        cache_write_tokens=1_800,
        cache_write_1h_tokens=1_800,
        batch_id="msgbatch_test1",
        custom_id="a1-t1",
    )
    entry = line(answer)
    assert (entry.cache_write_tokens, entry.cache_write_1h_tokens, entry.cache_read_tokens) == (
        1_800,
        1_800,
        0,
    )
    assert (entry.batch_id, entry.custom_id) == ("msgbatch_test1", "a1-t1")
    assert entry.latency_seconds is None  # a request in a batch has no latency of its own
    assert entry.parameters is not None and entry.parameters.cache == "1h"
    assert line(cached()).latency_seconds == 0.25


def test_a_failed_request_in_a_batch_names_it() -> None:
    error = BackendError("expired", reason="batch_expired")
    entry = LedgerEntry.from_failure(
        error, text="Invented.", batch_id="msgbatch_test1", custom_id="a1-t4", **WORK
    )
    assert (entry.outcome, entry.error_reason) == ("failed", "batch_expired")
    assert (entry.batch_id, entry.custom_id) == ("msgbatch_test1", "a1-t4")


def test_cache_counts_need_a_source_and_a_failed_line_has_none() -> None:
    data = line(cached(cache_read_tokens=5)).model_dump(mode="json")
    data.update(prompt_tokens=None, completion_tokens=None, token_source=None)
    with pytest.raises(ValueError, match="token counts need token_source"):
        LedgerEntry.model_validate(data)
    failed = LedgerEntry.from_failure(RuntimeError("x"), text="Invented.", **WORK)
    data = {**failed.model_dump(mode="json"), "cache_read_tokens": 3, "token_source": "server"}
    with pytest.raises(ValueError, match="a failed call got no answer"):
        LedgerEntry.model_validate(data)


# Versions 1 and 2 are still read --------------------------------------------------------------


def test_a_ledger_written_with_version_2_is_still_read() -> None:
    # Written by reliacode at ledger version 2 (tests/reference/make_ledger_v2.py).
    raw = [json.loads(text) for text in V2_FILE.read_text(encoding="utf-8").splitlines()]
    assert [entry["schema_version"] for entry in raw] == [2, 2, 2, 2, 2]
    assert all(name not in entry for entry in raw for name in NEW_FIELDS)
    entries = read_ledger(V2_FILE)
    assert [e.outcome for e in entries] == [
        "answered",
        "asked_again",
        "answered",
        "answered",
        "failed",
    ]
    assert [e.reasoning_tokens for e in entries] == [32, 32, 32, 32, None]
    assert entries[0].parameters is not None
    assert entries[0].parameters.extra_body == {"reasoning_effort": "low"}
    assert entries[0].parameters.temperature is None
    assert entries[0].parameters.cache is None and entries[0].parameters.api_version is None
    assert all(getattr(e, name) is None for e in entries for name in NEW_FIELDS)
    table = PriceTable(
        currency="USD",
        read_on="2026-11-25",
        prices={"example-reasoning-model": ModelPrice(input_per_million=1, output_per_million=2)},
    )
    # Priced exactly as before: 610 input and 40 output tokens a call, four calls counted.
    assert summarize(entries, table).total.cost == pytest.approx(4 * (610 * 1 + 40 * 2) / 1e6)


def test_lines_of_all_three_versions_can_share_one_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        V1_FILE.read_text(encoding="utf-8") + V2_FILE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    ledger = Ledger(path)
    assert ledger.calls == 9
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    backend = FakeBackend(['{"code": "conflict"}', '{"code": "government"}'], tokens=(50, 5))
    code(codebook, ["An invented fourth text."], backend=backend, ledger=ledger)
    versions = [entry.schema_version for entry in read_ledger(path)]
    assert versions == [1] * 4 + [2] * 5 + [3, 3]


def test_the_version_2_fixture_was_made_with_the_old_source() -> None:
    script = (FIXTURES.parent / "reference" / "make_ledger_v2.py").read_text(encoding="utf-8")
    assert "git archive c69b4e8 src" in script
    assert "assert LEDGER_VERSION == 2" in script


# Prices ---------------------------------------------------------------------------------------


def test_a_call_without_cache_counts_costs_what_it_cost_before() -> None:
    entry = line(response(prompt_tokens=1_000, completion_tokens=100))
    plain = ModelPrice(input_per_million=1.0, output_per_million=5.0)
    assert plain.cost(entry) == pytest.approx((1_000 * 1.0 + 100 * 5.0) / 1e6)
    assert PRICE.cost(entry) == plain.cost(entry)


@pytest.mark.parametrize(
    ("cache", "breakdown", "expected"),
    [
        ("1h", None, 200 * 1.0 + 1_800 * 2.0 + 100 * 5.0),
        ("5m", None, 200 * 1.0 + 1_800 * 1.25 + 100 * 5.0),
        ("1h", 1_800, 200 * 1.0 + 1_800 * 2.0 + 100 * 5.0),
        ("1h", 800, 200 * 1.0 + 1_000 * 1.25 + 800 * 2.0 + 100 * 5.0),
        ("5m", 0, 200 * 1.0 + 1_800 * 1.25 + 100 * 5.0),
    ],
)
def test_cache_writes_are_priced_by_their_time_to_live(cache, breakdown, expected) -> None:
    entry = line(cached(cache, cache_write_tokens=1_800, cache_write_1h_tokens=breakdown))
    assert PRICE.cost(entry) == pytest.approx(expected / 1e6)


def test_cache_reads_are_priced_at_the_read_price_or_else_as_plain_input() -> None:
    entry = line(cached(cache_read_tokens=1_800))
    assert PRICE.cost(entry) == pytest.approx((200 * 1.0 + 1_800 * 0.1 + 100 * 5.0) / 1e6)
    no_read_price = PRICE.model_copy(update={"cache_read_per_million": None})
    assert no_read_price.cost(entry) == pytest.approx((2_000 * 1.0 + 100 * 5.0) / 1e6)


def test_a_cost_that_cannot_be_known_is_none_never_a_guess() -> None:
    no_write_price = PRICE.model_copy(update={"cache_write_1h_per_million": None})
    writes = line(cached("1h", cache_write_tokens=1_800))
    assert no_write_price.cost(writes) is None
    # Caching was asked for, and the server did not say what it did.
    assert PRICE.cost(line(cached("5m", cache_write_tokens=None))) is None
    # Tokens written, but to which cache is not known.
    assert PRICE.cost(line(cached(None, cache_write_tokens=1_800))) is None
    # Counts that do not add up.
    assert PRICE.cost(line(cached(cache_read_tokens=2_500))) is None
    assert PRICE.cost(line(cached(cache_write_tokens=1_000, cache_write_1h_tokens=1_200))) is None


def test_a_request_in_a_batch_gets_the_batch_discount() -> None:
    alone = line(cached(cache_read_tokens=1_800))
    batched = line(cached(cache_read_tokens=1_800, batch_id="msgbatch_x", custom_id="a1-t1"))
    assert PRICE.cost(batched) == pytest.approx(PRICE.cost(alone) * 0.5)
    no_discount = PRICE.model_copy(update={"batch_discount": None})
    assert no_discount.cost(batched) == pytest.approx(PRICE.cost(alone))


def test_a_long_prompt_pays_the_long_prompt_prices_for_all_its_tokens() -> None:
    long = ModelPrice(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_write_1h_per_million=10.0,
        cache_read_per_million=0.5,
    )
    price = PRICE.model_copy(update={"long_prompt_above": 2_000, "long_prompt_price": long})
    at_the_limit = line(cached(cache_read_tokens=1_800))
    assert price.cost(at_the_limit) == PRICE.cost(at_the_limit)
    above = line(cached(prompt_tokens=2_001, cache_read_tokens=1_800))
    assert price.cost(above) == pytest.approx((201 * 5.0 + 1_800 * 0.5 + 100 * 25.0) / 1e6)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"long_prompt_above": 100_000}, "give long_prompt_above and long_prompt_price together"),
        ({"batch_discount": 1.0}, "less than 1"),
        ({"batch_discount": -0.1}, "greater than or equal to 0"),
        ({"cache_read_per_million": -1}, "greater than or equal to 0"),
        ({"cache_write_5m_per_million": float("inf")}, "finite"),
    ],
)
def test_prices_that_cannot_be_used_are_refused(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelPrice(input_per_million=1, output_per_million=2, **changes)
    nested = ModelPrice(
        input_per_million=1,
        output_per_million=2,
        long_prompt_above=10,
        long_prompt_price=ModelPrice(input_per_million=2, output_per_million=4),
    )
    with pytest.raises(ValueError, match="cannot have long-prompt prices of its own"):
        ModelPrice(
            input_per_million=1,
            output_per_million=2,
            long_prompt_above=5,
            long_prompt_price=nested,
        )


def test_a_worked_example_at_haiku_5_5s_prices_as_read_on_2026_11_25() -> None:
    # From Anthropic's pricing page, read 2026-11-25, for prompts up to 100,000 tokens: input
    # $0.10, 5m cache writes $0.125, 1h cache writes $0.20, cache hits $0.01, output $0.50 per
    # million tokens; batches at 50 percent; prompts over 100,000 tokens at $0.50, $0.625, $1,
    # $0.05 and $2.50. The model's name here is the one the page gives.
    haiku = ModelPrice(
        input_per_million=0.10,
        output_per_million=0.50,
        cache_write_5m_per_million=0.125,
        cache_write_1h_per_million=0.20,
        cache_read_per_million=0.01,
        batch_discount=0.5,
        long_prompt_above=100_000,
        long_prompt_price=ModelPrice(
            input_per_million=0.50,
            output_per_million=2.50,
            cache_write_5m_per_million=0.625,
            cache_write_1h_per_million=1.0,
            cache_read_per_million=0.05,
        ),
    )
    # One request in a batch: 16,900 tokens read from the cache, 500 plain, 570 output.
    hit = line(
        cached(
            prompt_tokens=17_400,
            completion_tokens=570,
            cache_read_tokens=16_900,
            batch_id="msgbatch_x",
            custom_id="a1-t1",
        )
    )
    assert haiku.cost(hit) == pytest.approx((500 * 0.10 + 16_900 * 0.01 + 570 * 0.50) / 1e6 * 0.5)
    # The same request writing the shared prompt to the one-hour cache instead.
    miss = line(
        cached(
            prompt_tokens=17_400,
            completion_tokens=570,
            cache_write_tokens=16_900,
            cache_write_1h_tokens=16_900,
            batch_id="msgbatch_x",
            custom_id="a1-t1",
        )
    )
    assert haiku.cost(miss) == pytest.approx((500 * 0.10 + 16_900 * 0.20 + 570 * 0.50) / 1e6 * 0.5)


# The other backend's cached tokens, and the summary --------------------------------------------


def test_the_openai_compatible_backend_counts_cached_input_tokens() -> None:
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 10,
        "total_tokens": 1_010,
        "prompt_tokens_details": {"cached_tokens": 900},
    }
    server = FakeServer(httpx.Response(200, json=completion(usage=usage)))
    backend = OpenAICompatibleBackend("http://localhost:8000/v1", "m", transport=server.transport)
    answer = backend.complete([{"role": "user", "content": "An invented text."}])
    assert answer.cache_read_tokens == 900
    assert answer.prompt_tokens == 1_000  # the cached tokens are part of the input
    assert answer.cache_write_tokens is None
    entry = line(answer)
    price = ModelPrice(input_per_million=1.0, output_per_million=2.0, cache_read_per_million=0.5)
    assert price.cost(entry) == pytest.approx((100 * 1.0 + 900 * 0.5 + 10 * 2.0) / 1e6)
    plain = ModelPrice(input_per_million=1.0, output_per_million=2.0)
    assert plain.cost(entry) == pytest.approx((1_000 * 1.0 + 10 * 2.0) / 1e6)  # as before
    no_details = completion(usage={"prompt_tokens": 5, "completion_tokens": 1})
    server = FakeServer(httpx.Response(200, json=no_details))
    backend = OpenAICompatibleBackend("http://localhost:8000/v1", "m", transport=server.transport)
    assert backend.complete([{"role": "user", "content": "x"}]).cache_read_tokens is None


def test_the_summary_adds_up_the_cache_counts(tmp_path: Path) -> None:
    lines = [
        line(cached(cache_write_tokens=1_800, cache_write_1h_tokens=1_800)),
        line(cached(cache_read_tokens=1_800)),
        line(response()),
    ]
    table = PriceTable(currency="USD", read_on="2026-11-25", prices={"example-cached-model": PRICE})
    summary = summarize(lines, table)
    assert summary.total.cache_write_tokens == 1_800
    assert summary.total.cache_read_tokens == 1_800
    assert summary.by_model["example-model"].cache_write_tokens is None
    assert list(Totals.model_fields)[-2:] == ["cache_write_tokens", "cache_read_tokens"]
    summary.to_csv(tmp_path / "summary.csv")
    header = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith("cache_write_tokens,cache_read_tokens,currency")


def test_a_cost_limit_counts_cached_calls_at_their_cache_prices(tmp_path: Path) -> None:
    table = PriceTable(currency="USD", read_on="2026-11-25", prices={"example-cached-model": PRICE})
    ledger = Ledger(tmp_path / "ledger.jsonl", prices=table, budget=Budget(cost_limit=0.01))
    ledger.record(line(cached(cache_write_tokens=1_800, cache_write_1h_tokens=1_800)))
    write_cost = (200 * 1.0 + 1_800 * 2.0 + 100 * 5.0) / 1e6
    assert ledger.summary().total.cost == pytest.approx(write_cost)
    ledger.record(line(cached(cache_read_tokens=1_800)))
    read_cost = (200 * 1.0 + 1_800 * 0.1 + 100 * 5.0) / 1e6
    assert ledger.summary().total.cost == pytest.approx(write_cost + read_cost)
    ledger.check()  # about 0.0052 so far, and the largest call 0.0043: the next one fits
    ledger.record(line(cached(cache_write_tokens=1_800, cache_write_1h_tokens=1_800)))
    with pytest.raises(BudgetError, match="could pass the cost limit"):
        ledger.check()


def test_the_time_of_a_line_is_kept_in_utc() -> None:
    assert line(cached()).at == NOON


# The worst case of a batch request, and the reserve ---------------------------------------------


def priced_ledger(tmp_path: Path, budget: Budget | None = None) -> Ledger:
    table = PriceTable(currency="USD", read_on="2026-11-25", prices={"example-cached-model": PRICE})
    return Ledger(tmp_path / "ledger.jsonl", prices=table, budget=budget)


def test_the_worst_case_is_the_largest_call_written_to_the_cache_at_the_batch_price(
    tmp_path: Path,
) -> None:
    ledger = priced_ledger(tmp_path)
    assert ledger._worst_case("example-cached-model", "1h") == (None, None)  # nothing counted
    ledger.record(line(cached(cache_read_tokens=1_800)))
    ledger.record(line(cached(prompt_tokens=1_500, completion_tokens=300)))
    tokens, cost = ledger._worst_case("example-cached-model", "1h")
    assert tokens == 2_000 + 300  # the largest input and the largest output seen
    assert cost == pytest.approx((2_000 * 2.0 + 300 * 5.0) / 1e6 * 0.5)
    _, five_minutes = ledger._worst_case("example-cached-model", "5m")
    assert five_minutes == pytest.approx((2_000 * 1.25 + 300 * 5.0) / 1e6 * 0.5)
    _, uncached = ledger._worst_case("example-cached-model", None)
    assert uncached == pytest.approx((2_000 * 1.0 + 300 * 5.0) / 1e6 * 0.5)
    assert ledger._worst_case("a-model-with-no-price", "1h") == (2_300, None)


def test_a_reserve_is_kept_back_from_the_budget_until_it_is_released(tmp_path: Path) -> None:
    ledger = priced_ledger(tmp_path, Budget(cost_limit=0.01, call_limit=10))
    ledger.record(line(cached(cache_read_tokens=1_800)))  # 0.00088
    ledger._reserve(4, 9_000, 0.0085)
    with pytest.raises(BudgetError, match=r"already sent may cost 0\.0085 USD more"):
        ledger._reserve(1, 1, 0.002)
    with pytest.raises(BudgetError, match="already sent may make 4 more"):
        ledger._reserve(6, 1, 0.0)
    with pytest.raises(BudgetError, match="could pass the cost limit"):
        ledger.check()  # a call made now would count the reserve too
    ledger._release(4, 9_000, 0.0085)
    ledger._reserve(1, 1, 0.002)
    ledger._reserve(100, None, 1.0, refuse=False)  # batches already sent are kept, not checked
    assert ledger._reserved_calls == 101
    ledger._release(500, 10**9, 50.0)
    assert (ledger._reserved_calls, ledger._reserved_tokens, ledger._reserved_cost) == (0, 0, 0)


def test_a_batch_whose_worst_case_is_not_known_cannot_be_kept_in_a_cost_or_token_limit(
    tmp_path: Path,
) -> None:
    ledger = priced_ledger(tmp_path, Budget(cost_limit=1.0))
    with pytest.raises(BudgetError, match="cannot be kept for a batch") as caught:
        ledger._reserve(3, None, None)
    assert "Code a few texts with code() first" in str(caught.value)
    tokens = Ledger(tmp_path / "tokens.jsonl", budget=Budget(token_limit=5_000))
    with pytest.raises(BudgetError, match="token limit of 5,000 cannot be kept"):
        tokens._reserve(3, None, None)
    tokens.record(line(cached()))
    with pytest.raises(BudgetError, match="could pass the token limit"):
        tokens._reserve(3, 3 * 2_100, None)
    calls_only = Ledger(tmp_path / "calls.jsonl", budget=Budget(call_limit=5))
    calls_only._reserve(5, None, None)  # a call limit needs no size
    with pytest.raises(BudgetError, match="allows 5 calls"):
        calls_only.check()

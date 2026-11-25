"""code_in_batches sends a run's requests through the Message Batches API and keeps the same
records as code(). A state file makes a restart safe: nothing is sent twice by itself. Every
answer comes from a mock of the API; no request leaves the test, and every text is invented."""

import json
from pathlib import Path

import httpx
import pytest

import reliacode.batches as batches
from helpers import EXAMPLES, FakeBatchAPI, FakeClock, FakeServer, anthropic_message
from reliacode import (
    AnthropicBackend,
    BackendError,
    BatchRun,
    Budget,
    BudgetError,
    FakeBackend,
    Ledger,
    ModelPrice,
    PriceTable,
    Rule,
    code,
    code_in_batches,
    load_codebook,
)

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
KEY = "test-batch-key-4f9a2c"
NEWS = load_codebook(EXAMPLES / "news_frames.yaml")
INCIVILITY = load_codebook(EXAMPLES / "incivility.yaml")
TEXTS = {
    "n1": "The council raised the bus fare to pay for new buses.",
    "n2": "A storm closed the harbor, and the ferry company blamed the forecast.",
}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)


def news_answer(custom_id: str, params: dict) -> str | dict:
    """Invented answers: the first ask for the second text's second variable cannot be used."""
    system = params["system"][0]["text"]
    if "Variable: dominant_frame" in system:
        return '{"code": "economic"}'
    if custom_id == "a1-t4":
        return "not json"
    return '{"code": "government"}'


def backend_for(api: FakeBatchAPI | FakeServer, clock: FakeClock | None = None, **settings):
    clock = clock or FakeClock()
    return AnthropicBackend(
        "claude-test-model",
        api_key_env=KEY_ENV,
        transport=api.transport,
        sleep=clock.sleep,
        clock=clock,
        **settings,
    )


def run_batches(api: FakeBatchAPI, tmp_path: Path, **settings) -> BatchRun:
    lines: list[str] = []
    settings.setdefault("report", lines.append)
    settings.setdefault("poll_every", 30)
    codebook = settings.pop("codebook", NEWS)
    texts = settings.pop("texts", TEXTS)
    backend = settings.pop("backend", None) or backend_for(api, cache="1h")
    return code_in_batches(
        codebook, texts, backend=backend, state=tmp_path / "state.json", **settings
    )


# Sending, reading and recording ---------------------------------------------------------------


def test_a_run_goes_as_a_batch_then_a_batch_of_second_asks(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    clock = FakeClock()
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = run_batches(api, tmp_path, ledger=ledger, backend=backend_for(api, clock, cache="1h"))
    created = api.created()
    assert [[r["custom_id"] for r in body["requests"]] for body in created] == [
        ["a1-t1", "a1-t2", "a1-t3", "a1-t4"],
        ["a2-t4"],
    ]
    assert result.run.table() == [
        {"unit": "n1", "dominant_frame": "economic", "responsibility_target": "government"},
        {"unit": "n2", "dominant_frame": "economic", "responsibility_target": "government"},
    ]
    assert result.submitted == 5
    assert result.failed == ()
    assert [(b.batch_id, b.ask, b.requests) for b in result.batches] == [
        ("msgbatch_test1", 1, 4),
        ("msgbatch_test2", 2, 1),
    ]
    assert result.batches[0].succeeded == 4
    assert result.batches[0].seconds == pytest.approx(270.530865)
    assert clock.sleeps == [30, 30]  # one look while each batch was in progress
    lines = ledger.entries()
    assert [(line.custom_id, line.outcome, line.ask) for line in lines] == [
        ("a1-t4", "asked_again", 1),
        ("a1-t3", "answered", 1),
        ("a1-t2", "answered", 1),
        ("a1-t1", "answered", 1),
        ("a2-t4", "answered", 2),
    ]  # in the order the results came, which is not the order sent
    assert {line.batch_id for line in lines} == {"msgbatch_test1", "msgbatch_test2"}
    assert all(line.latency_seconds is None for line in lines)
    assert all(line.parameters.endpoint.endswith("/v1/messages/batches") for line in lines)


def test_each_request_in_a_batch_is_the_request_code_would_send(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path)
    sync = FakeServer(
        *[
            httpx.Response(200, json=anthropic_message(text))
            for text in [
                '{"code": "economic"}',
                '{"code": "government"}',
                '{"code": "economic"}',
                "not json",
                '{"code": "government"}',
            ]
        ]
    )
    code(NEWS, TEXTS, backend=backend_for(sync, cache="1h"))
    sent = [r["params"] for body in api.created() for r in body["requests"]]
    order = [0, 1, 2, 3, 4]
    assert sent == [json.loads(sync.requests[n].content) for n in order]


def test_the_records_are_the_same_as_on_the_synchronous_path(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    batched = run_batches(api, tmp_path).run
    script = [
        '{"code": "economic"}',
        '{"code": "government"}',
        '{"code": "economic"}',
        "not json",
        '{"code": "government"}',
    ]
    plain = code(NEWS, TEXTS, backend=FakeBackend(script))
    assert batched.table() == plain.table()
    assert batched.prompts == plain.prompts
    assert batched.codebook_hash == plain.codebook_hash
    for one, other in zip(batched.results, plain.results, strict=True):
        assert (one.unit, one.variable, one.code) == (other.unit, other.variable, other.code)
        assert one.answers == other.answers
        assert one.missing_reason == other.missing_reason
        assert one.prompt_hash == other.prompt_hash
    last = batched.results[-1].responses[-1]
    assert (last.batch_id, last.custom_id) == ("msgbatch_test2", "a2-t4")
    assert last.parameters.cache == "1h"
    assert last.latency_seconds == pytest.approx(270.530865)
    assert last.created_at.isoformat() == "2026-11-25T10:04:30.654321+00:00"


def test_one_call_per_unit_with_a_rule_goes_by_batch_too(tmp_path: Path) -> None:
    rule = Rule(
        text="Severity is none if and only if incivility_form is none.",
        variables=["incivility_form", "severity"],
        check=lambda codes: (codes["severity"] == "none") == (codes["incivility_form"] == "none"),
    )

    def answer(custom_id: str, params: dict) -> str:
        if custom_id == "a1-t1":
            return '{"incivility_form": "none", "severity": "strong"}'  # breaks the rule
        return '{"incivility_form": "none", "severity": "none"}'

    api = FakeBatchAPI(answer)
    texts = ["Thanks for the clear notes.", "The meeting moved to Friday."]
    result = run_batches(
        api, tmp_path, codebook=INCIVILITY, texts=texts, recipe="per_unit", rules=[rule]
    )
    assert [[r["custom_id"] for r in body["requests"]] for body in api.created()] == [
        ["a1-t1", "a1-t2"],
        ["a2-t1"],
    ]
    second = api.created()[1]["requests"][0]["params"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "The codes break this rule" in second[-1]["content"]
    assert result.run.table() == [
        {"unit": 0, "incivility_form": "none", "severity": "none"},
        {"unit": 1, "incivility_form": "none", "severity": "none"},
    ]
    assert result.run.recipe == "per_unit"
    assert result.submitted == 3


# Starting again never pays twice ---------------------------------------------------------------


def test_a_run_started_again_sends_nothing_and_adds_no_line(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first = run_batches(api, tmp_path, ledger=ledger)
    before = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    again = run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert again.submitted == 0
    assert len(api.created()) == 2  # no new batch
    assert (tmp_path / "ledger.jsonl").read_text(encoding="utf-8") == before
    assert again.run.table() == first.run.table()
    # The records are complete again: the answers were read back from the API.
    assert [r.answers for r in again.run.results] == [r.answers for r in first.run.results]


def test_a_run_stopped_while_waiting_sends_only_what_was_never_sent(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer, polls=3)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    broken = backend_for(api, cache="1h")
    looks = {"count": 0}
    real_get = broken._get_batch

    def stops(batch_id: str) -> dict:
        looks["count"] += 1
        if looks["count"] == 2:
            raise BackendError("the network went down", reason="connection_lost")
        return real_get(batch_id)

    broken._get_batch = stops  # type: ignore[method-assign]
    with pytest.raises(BackendError, match="network went down"):
        run_batches(api, tmp_path, ledger=ledger, backend=broken)
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert [b["batch_id"] for b in state["batches"]] == ["msgbatch_test1"]
    assert ledger.entries() == ()
    result = run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert result.submitted == 1  # only the second ask, which was never sent
    assert [[r["custom_id"] for r in body["requests"]] for body in api.created()] == [
        ["a1-t1", "a1-t2", "a1-t3", "a1-t4"],
        ["a2-t4"],
    ]
    assert len(Ledger(tmp_path / "ledger.jsonl").entries()) == 5


def test_lines_already_in_the_ledger_are_not_added_again(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    # As if the program stopped after recording the lines but before saving the state file.
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    for entry in state["batches"]:
        entry["recorded"] = []
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert len(Ledger(tmp_path / "ledger.jsonl").entries()) == 5


# Requests that got no answer -----------------------------------------------------------------


def expiring(expired: set[str]):
    def answer(custom_id: str, params: dict) -> str | dict:
        if custom_id in expired:
            return {"type": "expired"}
        return news_answer(custom_id, params)

    return answer


def test_failed_requests_are_listed_and_sent_again_only_when_asked(tmp_path: Path) -> None:
    expired = {"a1-t2"}
    errored = {"type": "errored", "error": {"type": "error", "error": {}}}
    errored["error"]["error"] = {"type": "invalid_request_error", "message": "private words"}

    def answer(custom_id: str, params: dict) -> str | dict:
        if custom_id == "a1-t3":
            return errored
        return expiring(expired)(custom_id, params)

    api = FakeBatchAPI(answer)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first = run_batches(api, tmp_path, ledger=ledger)
    assert [(f.custom_id, f.unit, f.variable, f.reason, f.error_type) for f in first.failed] == [
        ("a1-t2", "n1", "responsibility_target", "expired", None),
        ("a1-t3", "n2", "dominant_frame", "errored", "invalid_request_error"),
    ]
    assert [(r.unit, r.variable) for r in first.run.results] == [
        ("n1", "dominant_frame"),
        ("n2", "responsibility_target"),
    ]
    failed_lines = [line for line in ledger.entries() if line.outcome == "failed"]
    assert sorted((line.custom_id, line.error_reason) for line in failed_lines) == [
        ("a1-t2", "batch_expired"),
        ("a1-t3", "batch_errored"),
    ]
    assert "private words" not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    # Started again without asking: nothing is sent, and the same requests are listed.
    again = run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    assert again.submitted == 0
    assert [f.custom_id for f in again.failed] == ["a1-t2", "a1-t3"]
    # Asked to: only those requests are sent, in a new batch.
    api.answer = news_answer  # this time, every request gets an answer
    resent = run_batches(
        api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"), resubmit_failed=True
    )
    assert [r["custom_id"] for r in api.created()[-1]["requests"]] == ["a1-t2", "a1-t3"]
    assert resent.failed == ()
    assert len(resent.run.results) == 4


def test_a_result_that_is_not_a_message_or_is_missing_is_listed(tmp_path: Path) -> None:
    def answer(custom_id: str, params: dict) -> str | dict:
        if custom_id == "a1-t1":
            return {"type": "succeeded", "message": {"id": "msg_1"}}  # no content
        return '{"code": "government"}' if custom_id != "a1-t2" else '{"code": "economic"}'

    api = FakeBatchAPI(answer)
    real = api._result

    def drop_one(request: dict) -> dict:
        if request["custom_id"] == "a1-t3":
            return {"custom_id": "a1-t3", "result": {"type": "canceled"}}
        return real(request)

    api._result = drop_one  # type: ignore[method-assign]
    result = run_batches(api, tmp_path, texts={"x": "An invented text."}, variables=None)
    assert [(f.custom_id, f.reason) for f in result.failed] == [("a1-t1", "unexpected_answer")]


# A batch whose sending was not confirmed -------------------------------------------------------


def test_a_lost_answer_while_sending_stops_until_you_say_what_happened(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    api.create_failures.append(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendError, match="cannot tell whether Anthropic created it") as caught:
        run_batches(api, tmp_path)
    assert caught.value.reason == "batch_unconfirmed"
    assert len(api.created()) == 1  # tried once, never again by itself
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["unconfirmed"]["custom_ids"] == ["a1-t1", "a1-t2", "a1-t3", "a1-t4"]
    with pytest.raises(BackendError, match="adopt_batch"):
        run_batches(api, tmp_path)
    assert len(api.created()) == 1
    result = run_batches(api, tmp_path, resubmit_unconfirmed=True)
    assert result.submitted == 5
    assert len(api.created()) == 3


def test_a_batch_found_in_the_console_can_be_adopted(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    api.create_failures.append(httpx.RemoteProtocolError("cut"))
    with pytest.raises(BackendError):
        run_batches(api, tmp_path)
    sent = api.created()[0]["requests"]
    api.add("msgbatch_found1", sent)  # it had been created after all
    result = run_batches(api, tmp_path, adopt_batch="msgbatch_found1")
    assert result.submitted == 1  # only the second ask
    assert result.batches[0].batch_id == "msgbatch_found1"
    assert len(result.run.results) == 4


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"adopt_batch": "msgbatch_1"}, "nothing to act on"),
        ({"resubmit_unconfirmed": True}, "nothing to act on"),
        ({"adopt_batch": "msgbatch_1", "resubmit_unconfirmed": True}, "not both"),
    ],
)
def test_the_choices_after_a_lost_answer_are_checked(tmp_path: Path, settings, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        run_batches(FakeBatchAPI(news_answer), tmp_path, **settings)


def test_a_batch_refused_outright_leaves_nothing_unconfirmed(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    refusal = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    api.create_failures.append(httpx.Response(400, json=refusal))
    with pytest.raises(BackendError) as caught:
        run_batches(api, tmp_path)
    assert caught.value.reason == "bad_request"
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["unconfirmed"] is None and state["batches"] == []
    assert run_batches(api, tmp_path).submitted == 5


def test_an_adopted_batch_with_requests_of_another_run_is_refused(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    api.create_failures.append(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendError):
        run_batches(api, tmp_path)
    params = api.created()[0]["requests"][0]["params"]
    api.add("msgbatch_other", [{"custom_id": "someone-else", "params": params}])
    with pytest.raises(ValueError, match="not one of this run's"):
        run_batches(api, tmp_path, adopt_batch="msgbatch_other")


# The estimate and the budget ------------------------------------------------------------------

PRICES = PriceTable(
    currency="USD",
    read_on="2026-11-25",
    prices={
        "claude-test-model": ModelPrice(
            input_per_million=1.0,
            output_per_million=5.0,
            cache_write_5m_per_million=1.25,
            cache_write_1h_per_million=2.0,
            cache_read_per_million=0.1,
            batch_discount=0.5,
        )
    },
)


def pilot(tmp_path: Path, budget: Budget | None) -> Ledger:
    """A ledger with one counted call, as after a small pilot with code()."""
    ledger = Ledger(tmp_path / "ledger.jsonl", prices=PRICES, budget=budget)
    server = FakeServer(httpx.Response(200, json=anthropic_message('{"code": "economic"}')))
    code(
        NEWS,
        ["A pilot text."],
        backend=backend_for(server),
        variables="dominant_frame",
        ledger=ledger,
    )
    return ledger


def test_the_worst_case_is_reported_and_kept_back_until_the_results_come(tmp_path: Path) -> None:
    ledger = pilot(tmp_path, Budget(cost_limit=1.0))
    lines: list[str] = []
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path, ledger=ledger, report=lines.append)
    # The pilot call used 50 input and 7 output tokens: each request is taken to write all 50
    # to the one-hour cache, at the batch price.
    each = (50 * 2.0 + 7 * 5.0) / 1e6 * 0.5
    assert lines[0] == (
        f"Sending 4 requests for ask 1 in 1 batch. Worst case: {4 * each:.8f}".rstrip("0")
        + " USD for 228 tokens (each request taken as large as the largest call in the ledger so "
        "far, 50 input and 7 output tokens, with all of its input written to the 1h cache and "
        "none read, at the batch price)."
    )
    assert lines[1] == "Batch msgbatch_test1 sent: 4 requests (ask 1)."
    assert lines[2].startswith("Batch msgbatch_test1 ended after 4 min 31 s: 4 succeeded")
    assert (ledger._reserved_calls, ledger._reserved_tokens, ledger._reserved_cost) == (0, 0, 0)


def test_a_round_that_could_pass_the_budget_is_not_sent(tmp_path: Path) -> None:
    ledger = pilot(tmp_path, Budget(cost_limit=0.0002))
    api = FakeBatchAPI(news_answer)
    with pytest.raises(BudgetError, match="This batch could pass the cost limit") as caught:
        run_batches(api, tmp_path, ledger=ledger)
    assert api.created() == []
    assert caught.value.run is not None and caught.value.run.results == ()


def test_without_a_counted_call_a_cost_limit_cannot_be_kept_for_a_batch(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", prices=PRICES, budget=Budget(cost_limit=1.0))
    api = FakeBatchAPI(news_answer)
    with pytest.raises(BudgetError, match="cannot be kept for a batch"):
        run_batches(api, tmp_path, ledger=ledger)
    assert api.created() == []
    lines: list[str] = []
    (tmp_path / "free").mkdir()
    run_batches(api, tmp_path / "free", ledger=None, report=lines.append)
    assert lines[0].endswith("No cost estimate: the run has no ledger.")
    (tmp_path / "unpriced").mkdir()
    lines.clear()
    run_batches(
        api, tmp_path / "unpriced", ledger=Ledger(tmp_path / "u.jsonl"), report=lines.append
    )
    assert "No estimate yet: no call in the ledger has token counts" in lines[0]


def test_cache_reads_in_a_batch_are_priced_with_the_batch_discount(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 40,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1_800,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 9,
    }
    api = FakeBatchAPI(news_answer, usage=usage)
    ledger = Ledger(tmp_path / "ledger.jsonl", prices=PRICES)
    run_batches(api, tmp_path, ledger=ledger)
    summary = ledger.summary()
    assert summary.total.cache_read_tokens == 5 * 1_800
    assert summary.total.input_tokens == 5 * 1_840
    assert summary.total.cost == pytest.approx(5 * (40 * 1.0 + 1_800 * 0.1 + 9 * 5.0) / 1e6 * 0.5)


# Limits, state files and arguments --------------------------------------------------------------


def test_a_round_is_split_within_the_limits_of_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batches, "_MAX_REQUESTS", 3)
    api = FakeBatchAPI(news_answer)
    result = run_batches(api, tmp_path)
    assert [len(body["requests"]) for body in api.created()] == [3, 1, 1]
    assert len(result.run.results) == 4
    monkeypatch.setattr(batches, "_MAX_REQUESTS", 100_000)
    monkeypatch.setattr(batches, "_MAX_BYTES", 1)  # every request alone
    api = FakeBatchAPI(news_answer)
    (tmp_path / "bytes").mkdir()
    run_batches(api, tmp_path / "bytes")
    assert [len(body["requests"]) for body in api.created()] == [1, 1, 1, 1, 1]


def test_a_state_file_of_another_run_is_refused(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path)
    with pytest.raises(ValueError, match="belongs to another run"):
        run_batches(api, tmp_path, texts={"n1": "Another invented text."})
    with pytest.raises(ValueError, match="belongs to another run"):
        run_batches(api, tmp_path, backend=backend_for(api, cache="5m"))
    (tmp_path / "state.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a reliacode batch state file"):
        run_batches(api, tmp_path)


def test_the_state_file_holds_no_text_no_answer_and_no_key(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path)
    saved = (tmp_path / "state.json").read_text(encoding="utf-8")
    for secret in [*TEXTS.values(), *TEXTS, "economic", "government", "not json", KEY]:
        assert secret not in saved
    state = json.loads(saved)
    assert set(state) == {"reliacode_batch_state", "run", "batches", "unconfirmed"}
    assert set(state["batches"][0]) == {
        "batch_id",
        "ask",
        "custom_ids",
        "submitted_at",
        "worst_case_tokens_each",
        "worst_case_cost_each",
        "recorded",
    }


def test_results_no_longer_kept_are_explained(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path)
    api.archived.add("msgbatch_test1")
    with pytest.raises(BackendError, match="keeps a batch's results for 29 days"):
        run_batches(api, tmp_path)
    api.archived.clear()
    api.results_gone.add("msgbatch_test1")
    with pytest.raises(BackendError) as caught:
        run_batches(api, tmp_path)
    assert caught.value.reason == "batch_results_gone"


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"backend": FakeBackend(["x"])}, "should be an AnthropicBackend"),
        ({"poll_every": 0}, "more than 0"),
        ({"poll_every": "60"}, "a number of seconds"),
        ({"resubmit_failed": "yes"}, "True or False"),
        ({"report": "print"}, "a function"),
        ({"attempts": 0}, "attempts should be"),
        ({"recipe": "per_text"}, "recipe should be"),
    ],
)
def test_arguments_that_cannot_be_used_are_refused(tmp_path: Path, settings, problem) -> None:
    api = FakeBatchAPI(news_answer)
    with pytest.raises((TypeError, ValueError), match=problem):
        run_batches(api, tmp_path, **settings)
    assert api.requests == []


def test_a_state_path_that_cannot_be_used_is_refused(tmp_path: Path) -> None:
    api = FakeBatchAPI(news_answer)
    backend = backend_for(api)
    with pytest.raises(TypeError, match="file name"):
        code_in_batches(NEWS, TEXTS, backend=backend, state=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is a folder"):
        code_in_batches(NEWS, TEXTS, backend=backend, state=tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        code_in_batches(NEWS, TEXTS, backend=backend, state=tmp_path / "no" / "state.json")


def test_the_key_goes_only_into_the_authorization_header_of_every_batch_request(
    tmp_path: Path,
) -> None:
    api = FakeBatchAPI(news_answer)
    run_batches(api, tmp_path, ledger=Ledger(tmp_path / "ledger.jsonl"))
    for request in api.requests:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["anthropic-version"] == "2026-10-21"
        assert KEY not in str(request.url)
        assert KEY.encode() not in request.content
    assert KEY not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")

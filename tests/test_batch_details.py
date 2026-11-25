"""The details of the batch path: the API's times, odd results, unknown states, and how the
backend's batch requests handle refusals and lost connections. Every answer comes from a mock
of the API; no request leaves the test."""

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from helpers import EXAMPLES, FakeBatchAPI, FakeClock, FakeServer, anthropic_message
from reliacode import AnthropicBackend, BackendError, BatchInfo, code_in_batches, load_codebook
from reliacode.batches import _duration, _time

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
NEWS = load_codebook(EXAMPLES / "news_frames.yaml")
TEXTS = {"d1": "The library opened a reading room for night workers."}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "test-key-for-batch-details")


def backend_for(api, clock: FakeClock | None = None, **settings) -> AnthropicBackend:
    clock = clock or FakeClock()
    return AnthropicBackend(
        "claude-test-model",
        api_key_env=KEY_ENV,
        transport=api.transport,
        sleep=clock.sleep,
        clock=clock,
        **settings,
    )


def answer(custom_id: str, params: dict) -> str:
    system = params["system"][0]["text"]
    return '{"code": "economic"}' if "dominant_frame" in system else '{"code": "government"}'


def run(api, tmp_path: Path, **settings):
    settings.setdefault("report", lambda line: None)
    backend = settings.pop("backend", None) or backend_for(api)
    return code_in_batches(NEWS, TEXTS, backend=backend, state=tmp_path / "state.json", **settings)


# Times ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2026-11-25T10:00:00Z", datetime(2026, 11, 25, 10, tzinfo=timezone.utc)),
        ("2026-11-25T12:00:00+02:00", datetime(2026, 11, 25, 10, tzinfo=timezone.utc)),
        (
            "2026-11-25T10:00:00.123456789Z",
            datetime(2026, 11, 25, 10, 0, 0, 123456, tzinfo=timezone.utc),
        ),
        ("2026-11-25T10:00:00", datetime(2026, 11, 25, 10, tzinfo=timezone.utc)),
        ("yesterday", None),
        (None, None),
        (1795000000, None),
    ],
)
def test_the_apis_times_are_read_in_utc(given: object, expected: datetime | None) -> None:
    assert _time(given) == expected


@pytest.mark.parametrize(
    ("seconds", "shown"), [(45, "45 s"), (271.4, "4 min 31 s"), (7_500, "2 h 5 min")]
)
def test_a_batch_time_is_written_plainly(seconds: float, shown: str) -> None:
    assert _duration(seconds) == shown


def test_a_batch_without_both_times_has_no_duration() -> None:
    info = BatchInfo(batch_id="b", ask=1, requests=1, submitted_at=datetime.now(timezone.utc))
    assert info.seconds is None


# Odd results and states -----------------------------------------------------------------------


def test_a_request_the_results_do_not_mention_is_listed_as_missing(tmp_path: Path) -> None:
    api = FakeBatchAPI(answer)
    real_add = api.add

    def add(batch_id: str, requests: list[dict]) -> None:
        real_add(batch_id, requests)
        api.batches[batch_id]["results"].pop(0)  # the API leaves out the first request

    api.add = add  # type: ignore[method-assign]
    result = run(api, tmp_path)
    assert [(f.custom_id, f.reason) for f in result.failed] == [("a1-t1", "missing")]
    assert [r.variable for r in result.run.results] == ["responsibility_target"]


@pytest.mark.parametrize(
    "odd",
    [
        {"type": "succeeded", "message": "not a message"},
        {"type": "something_new"},
        "not a result",
    ],
)
def test_an_odd_result_is_listed_as_an_unexpected_answer(tmp_path: Path, odd: object) -> None:
    api = FakeBatchAPI(answer)
    real_result = api._result

    def result(request: dict) -> dict:
        if request["custom_id"] == "a1-t2":
            return {"custom_id": "a1-t2", "result": odd}
        return real_result(request)

    api._result = result  # type: ignore[method-assign]
    got = run(api, tmp_path)
    assert [(f.custom_id, f.reason) for f in got.failed] == [("a1-t2", "unexpected_answer")]


def test_a_batch_state_reliacode_does_not_know_is_reported(tmp_path: Path) -> None:
    api = FakeBatchAPI(answer)
    real_state = api._state

    def state(batch_id: str) -> dict:
        return {**real_state(batch_id), "processing_status": "paused"}

    api._state = state  # type: ignore[method-assign]
    with pytest.raises(BackendError, match="the state 'paused'"):
        run(api, tmp_path)


def test_a_batch_being_canceled_is_waited_for(tmp_path: Path) -> None:
    api = FakeBatchAPI(answer, polls=2)
    real_state = api._state
    clock = FakeClock()

    def state(batch_id: str) -> dict:
        data = real_state(batch_id)
        if data["processing_status"] == "in_progress":
            data["processing_status"] = "canceling"
        return data

    api._state = state  # type: ignore[method-assign]
    run(api, tmp_path, backend=backend_for(api, clock), poll_every=45)
    assert clock.sleeps == [45, 45]


def test_an_adopted_batch_id_is_checked(tmp_path: Path) -> None:
    api = FakeBatchAPI(answer)
    api.create_failures.append(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendError):
        run(api, tmp_path)
    with pytest.raises(ValueError, match="should be a batch id"):
        run(api, tmp_path, adopt_batch="msgbatch 1/../x")
    with pytest.raises(BackendError, match="no batch msgbatch_nowhere") as caught:
        run(api, tmp_path, adopt_batch="msgbatch_nowhere")
    assert caught.value.reason == "batch_not_found"


def test_the_codebook_must_be_a_codebook(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="load_codebook"):
        code_in_batches(
            "codebook.yaml",  # type: ignore[arg-type]
            TEXTS,
            backend=backend_for(FakeBatchAPI(answer)),
            state=tmp_path / "s.json",
        )


# The backend's batch requests -----------------------------------------------------------------


def results_server(*answers: object) -> FakeServer:
    return FakeServer(*answers)


def error(status: int, message: str = "invented") -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"message": message}})


def test_reading_results_is_tried_again_after_a_passing_problem() -> None:
    line = '{"custom_id": "a1-t1", "result": {"type": "expired"}}\n\n'
    clock = FakeClock()
    server = results_server(error(529), httpx.Response(200, content=line.encode()))
    backend = backend_for(server, clock)
    assert list(backend._batch_results("msgbatch_1")) == [
        {"custom_id": "a1-t1", "result": {"type": "expired"}}
    ]
    assert clock.sleeps == [1.0]
    assert str(server.requests[0].url).endswith("/v1/messages/batches/msgbatch_1/results")


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (error(500), "server_error"),
        (error(400), "bad_request"),
        (error(404), "batch_results_gone"),
        (httpx.ConnectError("refused"), "no_server"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.Response(200, content=b"<html>\n"), "unexpected_answer"),
        (httpx.Response(200, content=b"[1, 2]\n"), "unexpected_answer"),
    ],
)
def test_a_problem_reading_results_is_explained(answer: object, reason: str) -> None:
    backend = backend_for(results_server(answer), max_retries=0)
    with pytest.raises(BackendError) as caught:
        list(backend._batch_results("msgbatch_1"))
    assert caught.value.reason == reason


@pytest.mark.parametrize("body", [{"type": "message_batch"}, {"id": "../../v1/files"}, {"id": 12}])
def test_a_created_batch_needs_an_id_that_can_be_used(body: dict) -> None:
    backend = backend_for(results_server(httpx.Response(200, json=body)))
    with pytest.raises(BackendError, match="no batch id that can be used") as caught:
        backend._create_batch([{"custom_id": "a1-t1", "params": {}}])
    assert caught.value.reason == "unexpected_answer"


def test_a_batch_id_that_is_not_one_is_never_put_into_an_address() -> None:
    server = results_server(httpx.Response(200, json={}))
    backend = backend_for(server)
    for bad in ("../files", "msgbatch_1?x=1", "", "a" * 129):
        with pytest.raises(BackendError):
            backend._get_batch(bad)
        with pytest.raises(BackendError):
            list(backend._batch_results(bad))
    assert server.requests == []


def test_a_refused_batch_creation_is_explained() -> None:
    backend = backend_for(results_server(error(413, "too large")), max_retries=0)
    with pytest.raises(BackendError, match="larger than the API accepts"):
        backend._create_batch([])
    backend = backend_for(results_server(error(529)), max_retries=1)
    with pytest.raises(BackendError, match="was overloaded"):
        backend._create_batch([])


def test_the_batches_address_comes_from_the_base_url() -> None:
    backend = AnthropicBackend("m", base_url="https://api.anthropic.com/v1")
    assert backend.batches_endpoint == "https://api.anthropic.com/v1/messages/batches"
    assert anthropic_message()["type"] == "message"

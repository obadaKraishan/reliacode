"""Retries: passing problems are tried again with growing waits, on a clock the tests control."""

import logging
import secrets

import httpx
import pytest

from helpers import FakeClock, FakeServer, completion
from reliacode.backends import BackendError, OpenAICompatibleBackend

MESSAGES = [{"role": "user", "content": "The river path reopens after the spring floods."}]


def ok() -> httpx.Response:
    return httpx.Response(200, json=completion())


def status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, json={"error": {"message": f"scripted {code}"}}, headers=headers)


def backend_for(server: FakeServer, clock: FakeClock, **settings: object):
    return OpenAICompatibleBackend(
        "http://localhost:8000/v1",
        "tiny-model",
        transport=server.transport,
        sleep=clock.sleep,
        clock=clock,
        **settings,
    )


def test_too_many_requests_is_tried_again_with_growing_waits() -> None:
    server, clock = FakeServer(status(429), status(429), ok()), FakeClock(step=0.5)
    response = backend_for(server, clock).complete(MESSAGES)
    assert len(server.requests) == 3
    assert clock.sleeps == [1.0, 2.0]
    assert response.attempts == 3
    assert response.latency_seconds == 0.5  # the try that got the answer, not the waits


def test_server_errors_are_tried_again_then_explained() -> None:
    server, clock = FakeServer(status(503)), FakeClock()
    with pytest.raises(BackendError) as caught:
        backend_for(server, clock).complete(MESSAGES)
    error = caught.value
    assert len(server.requests) == 4  # the first try and three retries
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert (error.reason, error.status) == ("server_error", 503)
    message = str(error)
    assert "had a problem of its own (status 503) 4 times in a row" in message
    assert "usually temporary" in message
    assert 'The server said: "scripted 503"' in message


@pytest.mark.parametrize("code", [500, 502, 504, 599])
def test_every_server_error_is_retried(code: int) -> None:
    server, clock = FakeServer(status(code), ok()), FakeClock()
    assert backend_for(server, clock).complete(MESSAGES).attempts == 2


def test_a_rate_limit_that_holds_is_explained() -> None:
    server, clock = FakeServer(status(429)), FakeClock()
    with pytest.raises(BackendError) as caught:
        backend_for(server, clock, max_retries=2).complete(MESSAGES)
    error = caught.value
    assert (error.reason, error.status) == ("rate_limited", 429)
    assert '"too many requests" (status 429) 3 times in a row' in str(error)
    assert "raise max_retries" in str(error)
    assert 'The server said: "scripted 429"' in str(error)


def test_waits_stop_growing_at_thirty_seconds() -> None:
    server, clock = FakeServer(status(500)), FakeClock()
    with pytest.raises(BackendError):
        backend_for(server, clock, max_retries=7).complete(MESSAGES)
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


@pytest.mark.parametrize(("header", "wait"), [("5", 5.0), ("0", 0.0), ("2.5", 2.5)])
def test_the_wait_the_server_asks_for_is_used(header: str, wait: float) -> None:
    server, clock = FakeServer(status(429, **{"Retry-After": header}), ok()), FakeClock()
    backend_for(server, clock).complete(MESSAGES)
    assert clock.sleeps == [wait]


@pytest.mark.parametrize("header", ["Wed, 18 Nov 2026 07:28:00 GMT", "soon", "-3", "inf"])
def test_a_wait_given_as_a_date_or_nonsense_falls_back_to_the_usual_waits(header: str) -> None:
    server, clock = FakeServer(status(429, **{"Retry-After": header}), ok()), FakeClock()
    backend_for(server, clock).complete(MESSAGES)
    assert clock.sleeps == [1.0]


def test_a_wait_longer_than_a_minute_stops_at_once() -> None:
    server, clock = FakeServer(status(429, **{"Retry-After": "3600"})), FakeClock()
    with pytest.raises(BackendError) as caught:
        backend_for(server, clock).complete(MESSAGES)
    assert len(server.requests) == 1
    assert clock.sleeps == []
    assert caught.value.reason == "rate_limited"
    assert "asked to wait 3600 seconds" in str(caught.value)
    assert "daily quota" in str(caught.value)


def test_no_retries_means_one_try() -> None:
    server, clock = FakeServer(status(500)), FakeClock()
    with pytest.raises(BackendError, match=r"\(status 500\) once\."):
        backend_for(server, clock, max_retries=0).complete(MESSAGES)
    assert len(server.requests) == 1
    assert clock.sleeps == []


def test_a_timeout_is_tried_again() -> None:
    server = FakeServer(httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), ok())
    clock = FakeClock()
    assert backend_for(server, clock).complete(MESSAGES).attempts == 3
    assert clock.sleeps == [1.0, 2.0]


def test_timeouts_that_hold_are_explained() -> None:
    server, clock = FakeServer(httpx.ReadTimeout("slow")), FakeClock()
    with pytest.raises(BackendError) as caught:
        backend_for(server, clock, timeout=20).complete(MESSAGES)
    assert caught.value.reason == "timeout"
    assert caught.value.status is None
    assert "did not answer within 20 seconds, 4 times in a row" in str(caught.value)
    assert "raise timeout" in str(caught.value)


@pytest.mark.parametrize(
    "failure",
    [httpx.RemoteProtocolError("Server disconnected"), httpx.ReadError("Connection reset")],
)
def test_a_dropped_connection_is_tried_again_then_explained(failure: Exception) -> None:
    server, clock = FakeServer(failure, ok()), FakeClock()
    assert backend_for(server, clock).complete(MESSAGES).attempts == 2
    server, clock = FakeServer(failure), FakeClock()
    with pytest.raises(BackendError) as caught:
        backend_for(server, clock, max_retries=1).complete(MESSAGES)
    assert caught.value.reason == "connection_lost"
    assert "broke off before an answer came, 2 times in a row" in str(caught.value)
    assert str(failure) in str(caught.value)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_a_refusal_is_not_tried_again(code: int) -> None:
    server, clock = FakeServer(status(code)), FakeClock()
    with pytest.raises(BackendError):
        backend_for(server, clock).complete(MESSAGES)
    assert len(server.requests) == 1
    assert clock.sleeps == []


def test_each_retry_is_logged_without_the_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    key = "test-" + secrets.token_urlsafe(24)
    monkeypatch.setenv("RELIACODE_TEST_API_KEY", key)
    caplog.set_level(logging.DEBUG)
    server, clock = FakeServer(status(429), ok()), FakeClock()
    backend_for(server, clock, api_key_env="RELIACODE_TEST_API_KEY").complete(MESSAGES)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "The server at http://localhost:8000/v1/chat/completions answered with status 429; "
        "trying again in 1.0 seconds (try 2 of 4)."
    ]
    assert key not in caplog.text


def test_the_log_says_what_went_wrong_for_each_kind_of_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    server = FakeServer(httpx.ReadTimeout("slow"), httpx.ReadError("reset"), ok())
    backend_for(server, FakeClock()).complete(MESSAGES)
    assert "did not answer in time" in caplog.records[0].getMessage()
    assert "dropped the connection" in caplog.records[1].getMessage()


@pytest.mark.parametrize(("value", "error"), [(-1, ValueError), ("3", TypeError), (2.0, TypeError)])
def test_max_retries_must_be_a_whole_number(value: object, error: type) -> None:
    with pytest.raises(error, match="max_retries"):
        OpenAICompatibleBackend("http://localhost:8000/v1", "m", max_retries=value)

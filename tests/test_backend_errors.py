"""Backend errors in plain words: what went wrong, where, and what to try."""

import httpx
import pytest

from helpers import FakeClock, FakeServer, completion
from reliacode.backends import BackendError, OpenAICompatibleBackend

LOCAL = "http://localhost:11434/v1"
HOSTED = "https://api.example.org/v1"
MESSAGES = [{"role": "user", "content": "The school choir sang at the harvest fair."}]


def fail(answer: object, base_url: str = LOCAL, **settings: object) -> BackendError:
    server = FakeServer(answer)
    clock = FakeClock()
    backend = OpenAICompatibleBackend(
        base_url,
        "tiny-model",
        transport=server.transport,
        sleep=clock.sleep,
        clock=clock,
        **settings,
    )
    with pytest.raises(BackendError) as caught:
        backend.complete(MESSAGES)
    error = caught.value
    assert error.endpoint == backend.endpoint
    return error


def refused(code: int, body: object = None, **headers: str) -> httpx.Response:
    if body is None:
        return httpx.Response(code, headers=headers)
    if isinstance(body, str):
        return httpx.Response(code, text=body, headers=headers)
    return httpx.Response(code, json=body, headers=headers)


# No server ---------------------------------------------------------------------------------------


def test_no_local_server_says_how_to_start_one() -> None:
    server = FakeServer(httpx.ConnectError("[Errno 61] Connection refused"))
    backend = OpenAICompatibleBackend(LOCAL, "tiny-model", transport=server.transport)
    with pytest.raises(BackendError) as caught:
        backend.complete(MESSAGES)
    error = caught.value
    assert len(server.requests) == 1  # a refused connection is not tried again
    assert (error.reason, error.status) == ("no_server", None)
    assert str(error) == (
        "There is no server answering at http://localhost:11434/v1 ([Errno 61] Connection "
        "refused). Check that the server is running and listens on this port. For Ollama, open "
        "the Ollama app or run ollama serve in a terminal, then try again."
    )


def test_no_hosted_server_says_to_check_the_address() -> None:
    error = fail(httpx.ConnectError("[Errno 8] nodename nor servname provided"), HOSTED)
    assert error.reason == "no_server"
    assert "There is no server answering at https://api.example.org/v1" in str(error)
    assert "internet connection" in str(error)
    assert "Ollama" not in str(error)


def test_a_connection_error_without_detail_is_still_clear() -> None:
    error = fail(httpx.ConnectError(""))
    assert str(error).startswith("There is no server answering at http://localhost:11434/v1. ")


# The model is not found --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        # Ollama
        refused(404, {"error": {"message": 'model "tiny-model" not found, try pulling it first'}}),
        # OpenAI
        refused(404, {"error": {"message": "The model `tiny-model` does not exist."}}),
        # vLLM
        refused(404, {"object": "error", "message": "The model `tiny-model` does not exist."}),
        # OpenRouter
        refused(400, {"error": {"message": "tiny-model is not a valid model ID", "code": 400}}),
    ],
    ids=["ollama", "openai", "vllm", "openrouter"],
)
def test_a_missing_model_is_named_with_what_to_try(answer: httpx.Response) -> None:
    error = fail(answer)
    assert error.reason == "model_not_found"
    message = str(error)
    assert 'The server at http://localhost:11434/v1 has no model called "tiny-model"' in message
    assert "ollama list" in message
    assert "ollama pull tiny-model" in message
    assert "The server said:" in message


@pytest.mark.parametrize("body", [None, "404 page not found", {"detail": "Not Found"}])
def test_a_wrong_path_says_to_check_the_base_address(body: object) -> None:
    error = fail(refused(404, body))
    assert (error.reason, error.status) == ("no_endpoint", 404)
    assert "has no chat completions endpoint at" in str(error)
    assert "it usually ends in /v1" in str(error)


# The key -----------------------------------------------------------------------------------------


def test_a_refused_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_API_KEY", "test-not-a-real-key-0000")
    error = fail(
        refused(401, {"error": {"message": "No auth credentials found"}}),
        HOSTED,
        api_key_env="SERVICE_API_KEY",
    )
    assert (error.reason, error.status) == ("key_refused", 401)
    assert "usually means a missing or wrong API key" in str(error)
    assert "Check that SERVICE_API_KEY holds a current key for this service" in str(error)
    assert 'The server said: "No auth credentials found"' in str(error)


def test_a_server_that_wants_a_key_says_how_to_give_one() -> None:
    error = fail(refused(403, {"error": "forbidden"}), HOSTED)
    assert (error.reason, error.status) == ("key_refused", 403)
    assert 'api_key_env="OPENROUTER_API_KEY"' in str(error)
    assert 'The server said: "forbidden"' in str(error)


# Bad requests ------------------------------------------------------------------------------------


def test_a_refused_schema_suggests_turning_structured_output_off() -> None:
    body = {"error": {"message": "response_format json_schema is not supported by this model"}}
    error = fail(refused(400, body))
    assert (error.reason, error.status) == ("bad_request", 400)
    assert "structured_output=False" in str(error)


@pytest.mark.parametrize(
    "message",
    [
        "`temperature` is deprecated for this model.",
        "temperature is not supported with this model",
        "Unsupported parameter: 'temperature' is not supported with this model.",
    ],
)
def test_a_refused_temperature_suggests_leaving_it_out(message: str) -> None:
    error = fail(refused(400, {"error": {"message": message}}))
    assert (error.reason, error.status) == ("bad_request", 400)
    assert "temperature=None, and the field is left out of the request" in str(error)
    assert message in str(error)


def test_a_message_about_temperature_is_not_read_as_a_missing_model() -> None:
    message = "Invalid value for temperature with this model: 5"
    error = fail(refused(400, {"error": {"message": message}}))
    assert error.reason == "bad_request"
    assert message in str(error)


def test_without_a_temperature_there_is_no_temperature_hint() -> None:
    message = "`temperature` is deprecated for this model."
    error = fail(refused(400, {"error": {"message": message}}), temperature=None)
    assert error.reason == "bad_request"
    assert "temperature=None" not in str(error)


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (refused(422, {"detail": "temperature must be at most 2"}), "temperature must be at most"),
        (refused(400, {"message": "max_tokens is too large"}), "max_tokens is too large"),
        (refused(400, {"error": "messages must not be empty"}), "messages must not be empty"),
        (refused(413, "<html><h1>Request   Entity\nToo Large</h1></html>"), "Request Entity Too"),
    ],
)
def test_a_bad_request_quotes_the_server(answer: httpx.Response, said: str) -> None:
    error = fail(answer)
    assert error.reason == "bad_request"
    assert f"refused the request (status {answer.status_code})" in str(error)
    assert said in str(error)
    assert "structured_output" not in str(error)
    assert "temperature=None" not in str(error)


def test_a_refusal_without_a_reason_says_so() -> None:
    error = fail(refused(400))
    assert str(error).endswith("The server gave no reason.")


def test_a_long_server_message_is_cut_short() -> None:
    error = fail(refused(400, {"error": {"message": "word " * 200}}))
    assert '..."' in str(error)
    assert len(str(error)) < 500


# Other answers -----------------------------------------------------------------------------------


def test_a_redirect_is_not_followed() -> None:
    server = FakeServer(refused(301, None, location="https://api.example.org/v2/chat/completions"))
    backend = OpenAICompatibleBackend(HOSTED, "tiny-model", transport=server.transport)
    with pytest.raises(BackendError) as caught:
        backend.complete(MESSAGES)
    assert len(server.requests) == 1
    assert caught.value.reason == "redirected"
    assert "sent the request on to https://api.example.org/v2/chat/completions" in str(caught.value)
    assert "Use that address as base_url" in str(caught.value)


def test_a_redirect_without_a_location_is_still_clear() -> None:
    assert "an address it did not give" in str(fail(refused(307)))


def test_an_error_sent_as_a_normal_answer_is_explained() -> None:
    error = fail(httpx.Response(200, json={"error": {"message": "upstream provider failed"}}))
    assert error.reason == "server_error"
    assert "answered with an error instead of a chat completion" in str(error)
    assert 'The server said: "upstream provider failed"' in str(error)


def test_every_error_is_a_backend_error_with_a_reason() -> None:
    reasons = {
        fail(httpx.ConnectError("refused")).reason,
        fail(refused(404)).reason,
        fail(refused(401)).reason,
        fail(refused(400)).reason,
        fail(refused(500)).reason,
        fail(refused(429)).reason,
        fail(httpx.ReadTimeout("slow")).reason,
        fail(httpx.Response(200, text="not json")).reason,
    }
    assert reasons == {
        "no_server",
        "no_endpoint",
        "key_refused",
        "bad_request",
        "server_error",
        "rate_limited",
        "timeout",
        "unexpected_answer",
    }
    assert issubclass(BackendError, RuntimeError)


def test_a_good_answer_after_a_refused_one_still_works() -> None:
    server = FakeServer(refused(400), httpx.Response(200, json=completion()))
    backend = OpenAICompatibleBackend(LOCAL, "tiny-model", transport=server.transport)
    with pytest.raises(BackendError):
        backend.complete(MESSAGES)
    assert backend.complete(MESSAGES).text == '{"code": "positive"}'

"""Prompt caching on AnthropicBackend: the cache marker goes on the shared system prompt only,
its time to live is the caller's choice and is recorded, the cache counts are read apart, and
caching changes no prompt text and no prompt hash. Every answer comes from a mock server."""

import json
import logging
from pathlib import Path

import httpx
import pytest

from helpers import EXAMPLES, FakeServer, anthropic_message
from reliacode import AnthropicBackend, Ledger, code, load_codebook

KEY_ENV = "RELIACODE_TEST_ANTHROPIC_KEY"
MESSAGES = [
    {"role": "system", "content": "A long codebook shared by every call."},
    {"role": "user", "content": "Text to code:\n\nThe ferry ran late all week."},
]
CACHED_USAGE = {
    "input_tokens": 40,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 1800,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 9,
}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "test-key-for-caching")


def backend_for(server: FakeServer, **settings: object) -> AnthropicBackend:
    return AnthropicBackend(
        "claude-test-model", api_key_env=KEY_ENV, transport=server.transport, **settings
    )


def ok(usage: dict | None = None, text: str = '{"code": "positive"}') -> httpx.Response:
    return httpx.Response(200, json=anthropic_message(text, usage=usage or CACHED_USAGE))


def markers(value: object, path: str = "") -> list[str]:
    """Where in a request body a cache marker is, as paths such as "system.0"."""
    found = []
    if isinstance(value, dict):
        for name, inner in value.items():
            if name == "cache_control":
                found.append(path)
            found += markers(inner, f"{path}.{name}" if path else name)
    elif isinstance(value, list):
        for number, inner in enumerate(value):
            found += markers(inner, f"{path}.{number}")
    return found


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_the_marker_goes_on_the_system_prompt_with_the_chosen_time_to_live(ttl: str) -> None:
    server = FakeServer(ok())
    response = backend_for(server, cache=ttl).complete(MESSAGES)
    body = server.body()
    assert body["system"] == [
        {
            "type": "text",
            "text": MESSAGES[0]["content"],
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]
    assert markers(body) == ["system.0"]
    assert body["messages"] == [MESSAGES[1]]  # the text being coded is never marked
    assert response.parameters.cache == ttl


def test_without_caching_there_is_no_marker_and_the_record_says_so() -> None:
    server = FakeServer(ok())
    response = backend_for(server).complete(MESSAGES)
    assert markers(server.body()) == []
    assert response.parameters.cache is None


def test_only_the_last_system_block_is_marked() -> None:
    server = FakeServer(ok())
    messages = [{"role": "system", "content": "Part one."}, *MESSAGES]
    backend_for(server, cache="1h").complete(messages)
    assert markers(server.body()) == ["system.1"]


def test_with_no_system_prompt_nothing_is_marked_and_caching_is_not_recorded() -> None:
    server = FakeServer(ok())
    response = backend_for(server, cache="1h").complete([MESSAGES[1]])
    assert markers(server.body()) == []
    assert response.parameters.cache is None


@pytest.mark.parametrize("ttl", ["1m", "60m", "5M", 300, True])
def test_a_time_to_live_the_api_does_not_offer_is_refused(ttl: object) -> None:
    with pytest.raises(ValueError, match='cache should be "5m"'):
        AnthropicBackend("claude-test-model", cache=ttl)


def test_cache_writes_and_reads_are_counted_apart_and_the_input_is_whole() -> None:
    write = {
        "input_tokens": 40,
        "cache_creation_input_tokens": 1800,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1800},
        "output_tokens": 9,
    }
    server = FakeServer(ok(write), ok())
    backend = backend_for(server, cache="1h")
    first, second = backend.complete(MESSAGES), backend.complete(MESSAGES)
    assert (first.cache_write_tokens, first.cache_write_1h_tokens, first.cache_read_tokens) == (
        1800,
        1800,
        0,
    )
    assert (second.cache_write_tokens, second.cache_write_1h_tokens, second.cache_read_tokens) == (
        0,
        0,
        1800,
    )
    # prompt_tokens is the whole input, as on the other backends: uncached, written and read.
    assert first.prompt_tokens == second.prompt_tokens == 1840
    assert first.total_tokens == 1849


def test_a_missing_breakdown_by_time_to_live_stays_missing() -> None:
    usage = {"input_tokens": 5, "cache_creation_input_tokens": 700, "output_tokens": 3}
    response = backend_for(FakeServer(ok(usage)), cache="5m").complete(MESSAGES)
    assert response.cache_write_tokens == 700
    assert response.cache_write_1h_tokens is None
    assert response.cache_read_tokens is None
    assert response.prompt_tokens == 705


def test_a_prompt_that_was_not_cached_is_warned_about_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nothing = {"input_tokens": 300, "cache_creation_input_tokens": 0, "output_tokens": 3}
    nothing["cache_read_input_tokens"] = 0
    backend = backend_for(FakeServer(ok(nothing)), cache="5m")
    with caplog.at_level(logging.WARNING, logger="reliacode.anthropic_backend"):
        backend.complete(MESSAGES)
        backend.complete(MESSAGES)
    warnings = [r for r in caplog.records if "Nothing was read from or written" in r.message]
    assert len(warnings) == 1
    assert "minimum cacheable length" in warnings[0].message
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="reliacode.anthropic_backend"):
        backend_for(FakeServer(ok(nothing))).complete(MESSAGES)  # caching off: no warning
    assert caplog.records == []


@pytest.mark.parametrize("recipe", ["per_variable", "per_unit"])
def test_caching_changes_no_prompt_text_and_no_prompt_hash(recipe: str, tmp_path: Path) -> None:
    codebook = load_codebook(EXAMPLES / "incivility.yaml")
    texts = {"c1": "Thanks for the update on the road works."}

    def run(cache: str | None) -> tuple[object, FakeServer]:
        def answer(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if recipe == "per_unit":
                return ok(text='{"incivility_form": "none", "severity": "none"}')
            system = body["system"][0]["text"]
            text = '{"code": "none"}'
            assert "Variable: incivility_form" in system or "Variable: severity" in system
            return ok(text=text)

        server = FakeServer(answer)
        backend = backend_for(server, cache=cache)
        return code(codebook, texts, backend=backend, recipe=recipe), server

    plain, plain_server = run(None)
    cached, cached_server = run("1h")
    assert [p.hash for p in plain.prompts] == [p.hash for p in cached.prompts]
    assert plain.prompts == cached.prompts
    assert plain.table() == cached.table()
    for without, with_cache in zip(plain_server.requests, cached_server.requests, strict=True):
        a, b = json.loads(without.content), json.loads(with_cache.content)
        for block in b["system"]:
            block.pop("cache_control", None)
        assert a == b  # the same request in every other byte of meaning
    assert all(r.parameters.cache == "1h" for result in cached.results for r in result.responses)


def test_a_second_ask_keeps_the_marker_on_the_system_prompt_only(tmp_path: Path) -> None:
    codebook = load_codebook(EXAMPLES / "news_frames.yaml")
    server = FakeServer(ok(text="not json"), ok(text='{"code": "economic"}'))
    backend = backend_for(server, cache="5m")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    run = code(
        codebook,
        ["The council raised the bus fare."],
        backend=backend,
        variables="dominant_frame",
        ledger=ledger,
    )
    assert run.results[0].code == "economic"
    second = server.body(1)
    assert markers(second) == ["system.0"]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    lines = ledger.entries()
    assert [line.parameters.cache for line in lines if line.parameters] == ["5m", "5m"]

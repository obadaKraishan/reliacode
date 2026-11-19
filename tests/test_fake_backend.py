"""FakeBackend: scripted, deterministic answers for tests, with no network."""

from datetime import timezone

import pytest

from reliacode.backends import Backend, BackendError, FakeBackend, Response, check_messages

USER = [{"role": "user", "content": "The council approved the new park."}]


def test_answers_come_back_in_order() -> None:
    backend = FakeBackend(["first", "second"])
    assert [backend.complete(USER).text for _ in range(2)] == ["first", "second"]


def test_the_response_is_a_full_record() -> None:
    backend = FakeBackend(["yes"], model="scripted")
    response = backend.complete(USER, json_schema={"type": "object"})
    assert isinstance(response, Response)
    assert response.model == "scripted"
    assert response.parameters.endpoint == "fake"
    assert response.parameters.model == "scripted"
    assert response.parameters.structured_output is True
    assert response.finish_reason == "stop"
    assert response.prompt_tokens is None  # a fake has no token counts, so none are invented
    assert response.created_at.tzinfo == timezone.utc


def test_calls_and_schemas_are_kept() -> None:
    backend = FakeBackend(["a", "b"])
    backend.complete(USER, json_schema={"type": "object"})
    backend.complete([*USER, {"role": "assistant", "content": "a"}])
    assert backend.calls[0] == USER
    assert len(backend.calls[1]) == 2
    assert backend.json_schemas == [{"type": "object"}, None]
    assert repr(backend) == "FakeBackend(model='fake-model', calls=2)"


def test_a_function_can_make_the_answers() -> None:
    def rule(messages):
        return "park" if "park" in messages[-1]["content"] else "other"

    backend = FakeBackend(rule)
    assert backend.complete(USER).text == "park"
    assert backend.complete([{"role": "user", "content": "A bridge."}]).text == "other"


def test_a_scripted_exception_is_raised() -> None:
    backend = FakeBackend(["ok", TimeoutError("scripted failure")])
    assert backend.complete(USER).text == "ok"
    with pytest.raises(TimeoutError, match="scripted failure"):
        backend.complete(USER)


def test_running_out_of_answers_says_what_to_do() -> None:
    backend = FakeBackend(["only"])
    backend.complete(USER)
    with pytest.raises(RuntimeError, match="given 1 answer and this is call 2"):
        backend.complete(USER)


def test_answers_must_be_a_list_or_a_function() -> None:
    with pytest.raises(TypeError, match=r"write \[answer\]"):
        FakeBackend("a single answer")


def test_a_function_must_return_text() -> None:
    backend = FakeBackend(lambda messages: 3)
    with pytest.raises(TypeError, match="should be text"):
        backend.complete(USER)


def test_it_is_a_backend() -> None:
    assert isinstance(FakeBackend([]), Backend)


@pytest.mark.parametrize(
    ("messages", "error", "match"),
    [
        ("hello", TypeError, "should be a list"),
        ([], ValueError, "is empty"),
        ([{"role": "user"}], ValueError, "exactly two fields"),
        (["hello"], ValueError, "exactly two fields"),
        ([{"role": "robot", "content": "hi"}], ValueError, 'role "robot"'),
        ([{"role": "user", "content": 5}], TypeError, "should be text"),
    ],
)
def test_bad_messages_are_explained(messages, error, match) -> None:
    with pytest.raises(error, match=match):
        check_messages(messages)


def test_checked_messages_are_a_plain_copy() -> None:
    original = [{"role": "system", "content": "Code the text."}, *USER]
    checked = check_messages(tuple(original))
    assert checked == original
    assert checked[0] is not original[0]


# Scripted token counts, to test a budget without a server ----------------------------------------


def test_one_pair_of_token_counts_is_reported_for_every_answer() -> None:
    backend = FakeBackend(["a", "b"], tokens=(120, 8))
    first, second = backend.complete(USER), backend.complete(USER)
    assert (first.prompt_tokens, first.completion_tokens, first.total_tokens) == (120, 8, 128)
    assert (second.prompt_tokens, second.completion_tokens) == (120, 8)


def test_a_list_of_token_counts_is_used_in_order() -> None:
    backend = FakeBackend(["a", "b", "c"], tokens=[(100, 5), None, (300, 7)])
    replies = [backend.complete(USER) for _ in range(3)]
    assert [(r.prompt_tokens, r.completion_tokens, r.total_tokens) for r in replies] == [
        (100, 5, 105),
        (None, None, None),
        (300, 7, 307),
    ]


def test_token_counts_stay_in_step_with_scripted_errors() -> None:
    backend = FakeBackend(
        ["a", BackendError("Down.", reason="no_server"), "c"], tokens=[(1, 1), (2, 2), (3, 3)]
    )
    assert backend.complete(USER).prompt_tokens == 1
    with pytest.raises(BackendError):
        backend.complete(USER)
    assert backend.complete(USER).prompt_tokens == 3


def test_a_function_can_make_the_token_counts() -> None:
    def count(messages: list[dict[str, str]]) -> tuple[int, int]:
        return sum(len(m["content"]) for m in messages), 4

    reply = FakeBackend(["a"], tokens=count).complete(USER)
    assert (reply.prompt_tokens, reply.completion_tokens) == (len(USER[0]["content"]), 4)
    assert FakeBackend(["a"], tokens=lambda messages: None).complete(USER).prompt_tokens is None


def test_a_function_must_return_a_pair() -> None:
    backend = FakeBackend(["a"], tokens=lambda messages: 12)  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="should return a pair or None"):
        backend.complete(USER)


def test_running_out_of_token_counts_is_explained() -> None:
    backend = FakeBackend(["a", "b"], tokens=[(1, 1)])
    backend.complete(USER)
    with pytest.raises(
        RuntimeError, match="no token counts left: it was given 1 and this is call 2"
    ):
        backend.complete(USER)


@pytest.mark.parametrize(
    "tokens",
    [(120,), (120, 8, 3), (-1, 8), (True, 8), (1.5, 8), [120, 8], "120,8", 120, [(1, 1), (2,)]],
)
def test_token_counts_that_cannot_be_used_are_refused(tokens: object) -> None:
    with pytest.raises(TypeError, match=r"a pair \(prompt_tokens, completion_tokens\)"):
        FakeBackend(["a"], tokens=tokens)  # type: ignore[arg-type]

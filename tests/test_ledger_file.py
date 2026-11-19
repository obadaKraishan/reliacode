"""The ledger file: JSON Lines, one line per call, flushed after each, never rewritten. A line
that cannot be read is reported with its line number, never skipped in silence."""

import json
from pathlib import Path

import pytest

from helpers import CODEBOOK_HASH, NOON, PROMPT_HASH, TickingClock, response
from reliacode.backends import BackendError
from reliacode.ledger import LEDGER_VERSION, Ledger, LedgerEntry, LedgerError, read_ledger

TEXT = "The new bridge opened two months late."
WORK = {"variable": "tone", "codebook_hash": CODEBOOK_HASH, "prompt_hash": PROMPT_HASH}


def entry(number: int = 1, **changes: object) -> LedgerEntry:
    reply = response(prompt_tokens=40 + number, completion_tokens=6)
    return LedgerEntry.from_response(
        reply, outcome="answered", text=f"{TEXT} ({number})", **{**WORK, **changes}
    )


def failure() -> LedgerEntry:
    error = BackendError("No server.", reason="no_server", endpoint="x")
    return LedgerEntry.from_failure(error, model="example-model", text=TEXT, at=NOON, **WORK)


def good_line(number: int = 1) -> str:
    return json.dumps(entry(number).model_dump(mode="json"))


# Writing -----------------------------------------------------------------------------------------


def test_each_call_is_one_json_line(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(entry(1))
    ledger.record(failure())
    raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["outcome"] for line in lines] == ["answered", "failed"]
    assert all(json.loads(line)["schema_version"] == LEDGER_VERSION for line in lines)
    assert ledger.calls == 2


def test_a_line_is_on_disk_as_soon_as_it_is_recorded(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(entry(1))
    assert read_ledger(tmp_path / "ledger.jsonl") == (entry(1),)


def test_text_outside_ascii_is_written_as_it_is(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(entry(variable="tonalité"))
    assert '"variable":"tonalité"' in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_lines_are_added_and_never_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path)
    first.record(entry(1))
    first.record(entry(2))
    before = path.read_bytes()
    second = Ledger(path)
    assert second.calls == 2  # the calls already in the file are counted
    second.record(entry(3))
    after = path.read_bytes()
    assert after.startswith(before)
    assert [e.prompt_tokens for e in read_ledger(path)] == [41, 42, 43]


def test_a_new_line_starts_on_its_own_line_after_a_last_line_with_no_line_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(good_line(1), encoding="utf-8")  # written by hand, with no line end
    ledger = Ledger(path)
    ledger.record(entry(2))
    ledger.record(entry(3))
    assert path.read_text(encoding="utf-8").count("\n") == 3
    assert [e.prompt_tokens for e in read_ledger(path)] == [41, 42, 43]


def test_the_file_is_made_when_the_ledger_opens(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path)
    assert path.is_file()
    assert read_ledger(path) == ()
    Ledger(path).record(entry(1))  # opened again while still empty
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_a_path_can_be_given_as_text(tmp_path: Path) -> None:
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    assert ledger.path == tmp_path / "ledger.jsonl"


def test_the_ledger_clock_can_be_replaced(tmp_path: Path) -> None:
    clock = TickingClock()
    ledger = Ledger(tmp_path / "ledger.jsonl", clock=clock)
    assert ledger.clock() < ledger.clock()


def test_the_lines_can_be_read_back_from_the_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(entry(1))
    ledger.record(failure())
    assert ledger.entries() == (entry(1), failure())


def test_only_ledger_lines_can_be_recorded(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(TypeError, match="LedgerEntry lines only"):
        ledger.record({"outcome": "answered"})  # type: ignore[arg-type]


def test_the_repr_shows_the_file_and_the_count(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(entry(1))
    assert repr(ledger) == f"Ledger({str(tmp_path / 'ledger.jsonl')!r}, calls=1)"


# Paths -------------------------------------------------------------------------------------------


def test_a_missing_folder_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="folder for the ledger file does not exist"):
        Ledger(tmp_path / "no-such-folder" / "ledger.jsonl")


def test_a_folder_is_not_a_ledger_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is a folder"):
        Ledger(tmp_path)


@pytest.mark.parametrize("path", [None, "", "   ", 3])
def test_the_path_must_be_a_file_name(path: object) -> None:
    with pytest.raises(TypeError, match="file name"):
        Ledger(path)  # type: ignore[arg-type]


def test_reading_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="There is no ledger file"):
        read_ledger(tmp_path / "ledger.jsonl")


# Reading lines that cannot be read ---------------------------------------------------------------


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(f"\n{good_line(1)}\n   \n{good_line(2)}\n\n", encoding="utf-8")
    assert len(read_ledger(path)) == 2


def test_a_line_cut_off_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(f"{good_line(1)}\n{good_line(2)}\n{good_line(3)[:40]}", encoding="utf-8")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    error = caught.value
    assert error.path == path
    assert [number for number, _ in error.problems] == [3]
    assert str(error) == (
        f"The ledger file {path} has 1 line that cannot be read:\n"
        "  Line 3: it is not a complete JSON object. A line cut off when a program stopped "
        "while writing looks like this.\n"
        "Keep a copy of the file, then fix or remove these lines in a text editor. reliacode "
        "only adds lines to a ledger and never changes the ones already there."
    )


def test_every_line_that_cannot_be_read_is_listed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    bad_field = json.dumps({**entry(4).model_dump(mode="json"), "text_length": -5})
    newer = json.dumps({**entry(5).model_dump(mode="json"), "schema_version": 99, "new": 1})
    lines = [
        good_line(1).encode(),
        b"not json",
        b"[1, 2, 3]",
        bad_field.encode(),
        newer.encode(),
        b"\xff\xfe broken bytes",
        good_line(2).encode(),
    ]
    path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    problems = dict(caught.value.problems)
    assert sorted(problems) == [2, 3, 4, 5, 6]
    assert "not a complete JSON object" in problems[2]
    assert "not a JSON object" in problems[3]
    assert problems[4].startswith("it is not a valid ledger line. text_length:")
    assert "newer version of reliacode (ledger version 99)" in problems[5]
    assert problems[6] == "it is not UTF-8 text."
    assert "has 5 lines that cannot be read" in str(caught.value)


def test_a_line_with_many_problems_names_the_first_three(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    (_, problem), *_ = caught.value.problems
    assert problem.startswith("it is not a valid ledger line. at: Field required; outcome:")
    assert "more)." in problem


def test_a_line_that_breaks_a_rule_is_explained(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    broken = {**entry(1).model_dump(mode="json"), "error_reason": "timeout"}
    path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    assert caught.value.problems[0][1] == (
        "it is not a valid ledger line. only a failed call has an error_reason or an http_status."
    )


def test_a_long_list_of_problems_is_cut_short(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text("x\n" * 14, encoding="utf-8")
    with pytest.raises(LedgerError) as caught:
        read_ledger(path)
    assert len(caught.value.problems) == 14
    message = str(caught.value)
    assert "  Line 10:" in message
    assert "  Line 11:" not in message
    assert "... and 4 more." in message


def test_a_ledger_does_not_open_on_a_file_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(f"{good_line(1)}\n{{broken\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(LedgerError, match="Line 2"):
        Ledger(path)
    assert path.read_bytes() == before  # nothing was added after the broken line

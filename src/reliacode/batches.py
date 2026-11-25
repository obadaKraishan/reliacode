"""Coding texts through Anthropic's Message Batches API: :func:`code_in_batches`.

A batch costs half the price of the same requests sent one at a time, and its answers come back
within 24 hours instead of seconds. :func:`code_in_batches` codes like :func:`reliacode.code`:
the same prompts, the same check of every answer, the same second ask, and the same records.
The requests go as batches instead: the first asks in one round of batches, and the second asks
for the answers that cannot be used in a second round.

A state file makes a run safe to stop and start again. It holds the id of every batch sent and
the requests in it, so a restart waits for those batches and sends only the requests that were
never sent. A request is never sent twice by itself: one that expired or failed is listed, and
sent again only when you ask. The state file holds no text, no answer and no key: only ids, the
estimates of what each batch could cost, and a fingerprint of the run's inputs.
"""

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from reliacode.anthropic_backend import _BATCH_ID, AnthropicBackend, _results_gone
from reliacode.backends import BackendError, utc_now
from reliacode.codebook import Codebook
from reliacode.coding import (
    CodingRun,
    Recipe,
    Rule,
    UnitId,
    _Plan,
    _plan,
    _ties,
    _UnitTask,
    _VariableTask,
)
from reliacode.errors import describe
from reliacode.ledger import BudgetError, Ledger, LedgerEntry, _number, hash_text

logger = logging.getLogger(__name__)

#: Version of the state file's format.
STATE_VERSION = 1
#: The Message Batches API's limits on one batch, as its documentation gives them: 100,000
#: requests or 256 MB, whichever comes first. Batches are kept under 240 MiB, below 256 MB in
#: either reading of "MB".
_MAX_REQUESTS = 100_000
_MAX_BYTES = 240 * 1024 * 1024
#: A custom id: the ask and the task's number, such as "a1-t17". It holds no id of yours.
_CUSTOM_ID = re.compile(r"a([1-9][0-9]*)-t([1-9][0-9]*)")
_FAILURES = ("errored", "canceled", "expired")
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?")

_Task = _VariableTask | _UnitTask


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BatchInfo(_Record):
    """One batch that a run sent: its id, which ask it holds, and how its requests ended.

    ``created_at`` and ``ended_at`` are the API's times. The counts are the API's
    ``request_counts`` when the batch ended. ``worst_case_tokens`` and ``worst_case_cost`` are
    what the batch could have used at the worst case, as estimated before it was sent (``None``
    when no estimate could be made).
    """

    batch_id: str
    ask: int
    requests: int
    submitted_at: datetime
    created_at: datetime | None = None
    ended_at: datetime | None = None
    succeeded: int | None = None
    errored: int | None = None
    canceled: int | None = None
    expired: int | None = None
    worst_case_tokens: int | None = None
    worst_case_cost: float | None = None

    @property
    def seconds(self) -> float | None:
        """How long the batch took, from its creation to its end, by the API's times."""
        if self.created_at is None or self.ended_at is None:
            return None
        return max((self.ended_at - self.created_at).total_seconds(), 0.0)


class BatchFailure(_Record):
    """A request that got no answer, so its text has no result yet.

    ``reason`` is the API's result type (``"errored"``, ``"canceled"`` or ``"expired"``),
    ``"unexpected_answer"`` for an answer that is not a message, or ``"missing"`` for a request
    the results do not mention. ``error_type`` is the API's kind of error for an errored request,
    such as ``"invalid_request_error"``; its message is never kept. None of these requests was
    billed, except possibly an unexpected answer.
    """

    unit: UnitId
    variable: str
    ask: int
    batch_id: str
    custom_id: str
    reason: Literal["errored", "canceled", "expired", "unexpected_answer", "missing"]
    error_type: str | None = None


class BatchRun(_Record):
    """What :func:`code_in_batches` produced.

    Attributes:
        run: the results, as :func:`reliacode.code` gives them, for every text whose last ask
            has an answer. A text with a failed request is not in it yet.
        batches: every batch of the run, in the order they were sent, this call's and earlier
            ones'.
        failed: the requests that got no answer. Run again with ``resubmit_failed=True`` to
            send them again.
        submitted: the requests this call sent. A run started again after every batch was sent
            sends none.
        state: the state file.
    """

    run: CodingRun
    batches: tuple[BatchInfo, ...]
    failed: tuple[BatchFailure, ...]
    submitted: int
    state: str


def code_in_batches(
    codebook: Codebook,
    texts: Sequence[str] | Mapping[UnitId, str],
    *,
    backend: AnthropicBackend,
    state: str | os.PathLike[str],
    variables: Sequence[str] | str | None = None,
    attempts: int = 2,
    ledger: Ledger | None = None,
    recipe: Recipe = "per_variable",
    rules: Sequence[Rule] | None = None,
    poll_every: float = 60.0,
    resubmit_failed: bool = False,
    adopt_batch: str | None = None,
    resubmit_unconfirmed: bool = False,
    report: Callable[[str], object] = print,
) -> BatchRun:
    """Code every text for every variable (or the ones named) through the Message Batches API.

    It works like :func:`reliacode.code`, with the same arguments and the same records: one
    result per text and variable, and one ledger line per request, which also carries the
    request's batch id and custom id. The requests go as batches within the API's limits. The
    function waits for each batch to end, asking the API for its state every ``poll_every``
    seconds, and reads the results by custom id. The second asks, for the answers that cannot
    be used, go as a second round of batches; the last answer decides, as with ``code()``.

    Before a round is sent, its worst case is reported and kept back from the ledger's budget:
    every request as large as the largest call in the ledger so far, with all of its input
    written to the cache and none read, at the batch price. The reserve is given back as the
    results are recorded. With a cost or token limit, the ledger needs at least one counted
    call first (a few texts coded with ``code()``), so that the size of a call is known.

    Run it again with the same ``state`` file after a stop: the batches already sent are waited
    for and read again (the API keeps results for 29 days), lines already in the ledger are not
    added again, and only the requests never sent are sent. Nothing is ever sent twice by
    itself.

    Args:
        codebook, texts, variables, attempts, ledger, recipe, rules: as for
            :func:`reliacode.code`.
        backend: an :class:`~reliacode.AnthropicBackend`. Its settings (model, token limit,
            cache, thinking, effort, structured output) apply to every request.
        state: the state file, such as ``"run-state.json"``. It is created if it does not
            exist. Give each run its own: a state file from a run with other inputs is refused.
        poll_every: how many seconds to wait between two looks at a batch's state.
        resubmit_failed: send again the requests that expired, failed or were canceled in an
            earlier call. By default they are only listed in the result's ``failed``.
        adopt_batch: the id of a batch that was being sent when the connection was lost (see
            below), found in the Claude Console, so that its results are used.
        resubmit_unconfirmed: after such a loss, send the requests again, once you have checked
            that no batch was created.
        report: where the estimate and the progress lines go: ``print`` by default.

    If the connection is lost while a batch is being sent, reliacode cannot tell whether the
    batch was created, so it stops and sends nothing more until you say which it was, with
    ``adopt_batch`` or ``resubmit_unconfirmed``. Guessing could pay for the requests twice.

    Raises:
        BackendError: the API could not be reached or refused a request, a batch's results are
            no longer available, or a batch's creation could not be confirmed
            (``reason="batch_unconfirmed"``).
        BudgetError: the ledger's budget refused a round. Its ``run`` holds the results
            finished before the stop.
        TypeError, ValueError: an argument is not as described, or the state file belongs to
            another run.
    """
    if not isinstance(codebook, Codebook):
        raise TypeError(
            'codebook should be a codebook loaded with reliacode.load_codebook("file.yaml").'
        )
    if not isinstance(backend, AnthropicBackend):
        raise TypeError(
            "backend should be an AnthropicBackend(...): the Message Batches API is Anthropic's, "
            f"but the backend is {describe(backend)}. For another backend, use reliacode.code()."
        )
    plan = _plan(
        codebook,
        texts,
        variables=variables,
        attempts=attempts,
        ledger=ledger,
        recipe=recipe,
        rules=rules,
        backend=backend,
    )
    if isinstance(poll_every, bool) or not isinstance(poll_every, (int, float)):
        raise TypeError(f"poll_every should be a number of seconds, but it is {poll_every!r}.")
    if not math.isfinite(poll_every) or poll_every <= 0:
        raise ValueError(f"poll_every should be more than 0 seconds, but it is {poll_every}.")
    for name, value in (
        ("resubmit_failed", resubmit_failed),
        ("resubmit_unconfirmed", resubmit_unconfirmed),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} should be True or False.")
    if not callable(report):
        raise TypeError("report should be a function that takes a line of text, such as print.")
    path = _state_path(state)
    tasks = list(plan.tasks(codebook))
    saved = _State.open(path, _fingerprint(plan, backend))
    runner = _Runner(plan, backend, saved, tasks, float(poll_every), report)
    started = utc_now()
    try:
        runner.resolve_unconfirmed(adopt_batch, resubmit_unconfirmed)
        runner.run(resubmit_failed)
    except BudgetError as error:
        error.run = plan.run(codebook, runner.results(), started)
        raise
    return BatchRun(
        run=plan.run(codebook, runner.results(), started),
        batches=runner.batch_infos(),
        failed=runner.failures(),
        submitted=runner.submitted,
        state=str(path),
    )


class _State:
    """The state file: every batch sent, the requests in each, and those already recorded."""

    def __init__(self, path: Path, fingerprint: str, data: dict[str, Any]) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.batches: list[dict[str, Any]] = data.get("batches", [])
        self.unconfirmed: dict[str, Any] | None = data.get("unconfirmed")

    @classmethod
    def open(cls, path: Path, fingerprint: str) -> "_State":
        if not path.exists():
            opened = cls(path, fingerprint, {})
            opened.save()
            return opened
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            data = None
        if not isinstance(data, dict) or data.get("reliacode_batch_state") != STATE_VERSION:
            raise ValueError(
                f"The file {path} is not a reliacode batch state file that this version can "
                "read. Give the run a new state file, or check the path."
            )
        if data.get("run") != fingerprint:
            raise ValueError(
                f"The state file {path} belongs to another run: its codebook, texts, variables, "
                "recipe, rules, attempts or backend settings differ from this one's, and "
                "resuming it would mix two runs. Give this run its own state file, or run again "
                "with exactly the same inputs."
            )
        return cls(path, fingerprint, data)

    def save(self) -> None:
        """Write the file in full, by way of a temporary file, so it is never half written."""
        data = {
            "reliacode_batch_state": STATE_VERSION,
            "run": self.fingerprint,
            "batches": self.batches,
            "unconfirmed": self.unconfirmed,
        }
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(partial, self.path)


class _Runner:
    """One call of :func:`code_in_batches`: its tasks, its batches and what it recorded."""

    def __init__(
        self,
        plan: _Plan,
        backend: AnthropicBackend,
        saved: _State,
        tasks: list[_Task],
        poll_every: float,
        report: Callable[[str], object],
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.saved = saved
        self.tasks = tasks
        self.ledger = plan.ledger
        self.poll_every = poll_every
        self.report = report
        self.submitted = 0
        self.polled: dict[str, dict[str, Any]] = {}
        self._failures: dict[int, BatchFailure] = {}
        # What is already in the ledger, so that a restart never adds a line twice.
        self.recorded: set[tuple[str, str]] = {
            (entry["batch_id"], custom_id)
            for entry in saved.batches
            for custom_id in entry.get("recorded", [])
        }
        if self.ledger is not None:
            self.recorded |= {
                (line.batch_id, line.custom_id)
                for line in self.ledger.entries()
                if line.batch_id is not None and line.custom_id is not None
            }

    # The rounds -------------------------------------------------------------------------------

    def run(self, resubmit_failed: bool) -> None:
        self._reserve_sent()
        for ask in range(1, self.plan.attempts + 1):
            for entry in [e for e in self.saved.batches if e["ask"] == ask]:
                self._finish(entry)
            waiting = [n for n, task in enumerate(self.tasks) if not task.done and task.ask == ask]
            never = [n for n in waiting if not self._sent(n, ask)]
            again = [n for n in waiting if self._sent(n, ask)] if resubmit_failed else []
            if never or again:
                for entry in self._submit(ask, never + again):
                    self._finish(entry)

    def results(self) -> list[Any]:
        return [result for task in self.tasks if task.done for result in task.results()]

    def failures(self) -> tuple[BatchFailure, ...]:
        listed = []
        for number, task in enumerate(self.tasks):
            if task.done:
                continue
            failure = self._failures.get(number)
            if failure is None:  # pragma: no cover - every waiting task was sent and read
                custom_id = _custom_id(task.ask, number)
                failure = BatchFailure(
                    unit=task.unit,
                    variable=task.work.label,
                    ask=task.ask,
                    batch_id=self._last_batch(number, task.ask) or "",
                    custom_id=custom_id,
                    reason="missing",
                )
            listed.append(failure)
        return tuple(listed)

    def batch_infos(self) -> tuple[BatchInfo, ...]:
        infos = []
        for entry in self.saved.batches:
            data = self.polled.get(entry["batch_id"], {})
            counts = data.get("request_counts")
            counts = counts if isinstance(counts, dict) else {}
            each_tokens, each_cost = (
                entry.get("worst_case_tokens_each"),
                entry.get("worst_case_cost_each"),
            )
            requests = len(entry["custom_ids"])
            infos.append(
                BatchInfo(
                    batch_id=entry["batch_id"],
                    ask=entry["ask"],
                    requests=requests,
                    submitted_at=_time(entry["submitted_at"]) or utc_now(),
                    created_at=_time(data.get("created_at")),
                    ended_at=_time(data.get("ended_at")),
                    succeeded=_whole(counts.get("succeeded")),
                    errored=_whole(counts.get("errored")),
                    canceled=_whole(counts.get("canceled")),
                    expired=_whole(counts.get("expired")),
                    worst_case_tokens=None if each_tokens is None else each_tokens * requests,
                    worst_case_cost=None if each_cost is None else each_cost * requests,
                )
            )
        return tuple(infos)

    # Sending ----------------------------------------------------------------------------------

    def _submit(self, ask: int, numbers: list[int]) -> list[dict[str, Any]]:
        """Send the requests for ``ask`` of the tasks numbered, as one or more batches."""
        requests = []
        for number in numbers:
            task = self.tasks[number]
            body, _ = self.backend._request(task.messages, task.work.prompt.json_schema)
            requests.append({"custom_id": _custom_id(ask, number), "params": body})
        chunks = _chunks(requests)
        each_tokens, each_cost = (
            self.ledger._worst_case(self.plan.model, self.backend.cache)
            if self.ledger is not None
            else (None, None)
        )
        count = len(requests)
        self.report(self._estimate(ask, count, len(chunks), each_tokens, each_cost))
        tokens = None if each_tokens is None else each_tokens * count
        cost = None if each_cost is None else each_cost * count
        if self.ledger is not None:
            self.ledger._reserve(count, tokens, cost)
        entries = []
        for index, chunk in enumerate(chunks):
            ids = [request["custom_id"] for request in chunk]
            self.saved.unconfirmed = {"ask": ask, "custom_ids": ids, "started_at": _now()}
            self.saved.save()
            try:
                data = self.backend._create_batch(chunk)
            except BackendError as error:
                left = sum(len(rest) for rest in chunks[index + 1 :])
                lost = error.reason in ("timeout", "connection_lost")
                kept_back = left if lost else left + len(chunk)
                self._give_back(kept_back, each_tokens, each_cost)
                if lost:
                    self.saved.save()
                    raise self._unconfirmed_error(self.saved.unconfirmed) from None
                self.saved.unconfirmed = None
                self.saved.save()
                raise
            entry = {
                "batch_id": data["id"],
                "ask": ask,
                "custom_ids": ids,
                "submitted_at": _now(),
                "worst_case_tokens_each": each_tokens,
                "worst_case_cost_each": each_cost,
                "recorded": [],
            }
            self.saved.batches.append(entry)
            self.saved.unconfirmed = None
            self.saved.save()
            self.submitted += len(ids)
            self.report(f"Batch {data['id']} sent: {len(ids):,} {_requests(len(ids))} (ask {ask}).")
            entries.append(entry)
        return entries

    def _estimate(
        self,
        ask: int,
        count: int,
        batches: int,
        each_tokens: int | None,
        each_cost: float | None,
    ) -> str:
        head = (
            f"Sending {count:,} {_requests(count)} for ask {ask} in {batches:,} "
            f"{'batch' if batches == 1 else 'batches'}."
        )
        ledger = self.ledger
        if ledger is None:
            return f"{head} No cost estimate: the run has no ledger."
        if each_tokens is None:
            return (
                f"{head} No estimate yet: no call in the ledger has token counts, so the size "
                "of a request is not known. Code a few texts with code() first for one."
            )
        sizes = (
            f"each request taken as large as the largest call in the ledger so far, "
            f"{ledger._largest_prompt:,} input and {ledger._largest_completion:,} output tokens"
        )
        tokens = f"{each_tokens * count:,} tokens"
        if each_cost is None or ledger.prices is None:
            return (
                f"{head} Worst case: {tokens} ({sizes}). No cost: the price table has no price "
                f'for the model "{self.plan.model}" or for its cache writes.'
            )
        cache = self.backend.cache
        where = (
            f"all of its input written to the {cache} cache and none read"
            if cache
            else "its input priced as plain input"
        )
        amount = f"{_number(each_cost * count)} {ledger.prices.currency}"
        return (
            f"{head} Worst case: {amount} for {tokens} ({sizes}, with {where}, at the batch price)."
        )

    def _unconfirmed_error(self, pending: dict[str, Any] | None) -> BackendError:
        pending = pending or {}
        count = len(pending.get("custom_ids", []))
        when = pending.get("started_at", "an unknown time")
        return BackendError(
            f"A batch of {count:,} {_requests(count)} (ask {pending.get('ask')}) was being sent "
            f"at {when} UTC when the answer was lost, so reliacode cannot tell whether Anthropic "
            "created it, and it sends nothing more until you say. Look in the Batches page of "
            f"the Claude Console for a batch created then with {count:,} {_requests(count)}. If "
            'it is there, run again with adopt_batch="msgbatch_..." (its id) to use its '
            "results. If it is not, run again with resubmit_unconfirmed=True to send the "
            "requests again. Sending them again without checking could pay for them twice.",
            reason="batch_unconfirmed",
            endpoint=self.backend.batches_endpoint,
        )

    def resolve_unconfirmed(self, adopt_batch: str | None, resubmit_unconfirmed: bool) -> None:
        pending = self.saved.unconfirmed
        if adopt_batch is not None and resubmit_unconfirmed:
            raise ValueError(
                "Give adopt_batch or resubmit_unconfirmed, not both: a batch was either created "
                "(adopt it) or not (send its requests again)."
            )
        if pending is None:
            if adopt_batch is not None or resubmit_unconfirmed:
                raise ValueError(
                    f"The state file {self.saved.path} has no batch whose creation is unconfirmed, "
                    "so adopt_batch and resubmit_unconfirmed have nothing to act on. Leave them "
                    "out."
                )
            return
        if adopt_batch is not None:
            if not isinstance(adopt_batch, str) or not _BATCH_ID.fullmatch(adopt_batch):
                raise ValueError(
                    'adopt_batch should be a batch id, such as "msgbatch_01HkcTjaV5uDC8jWR4".'
                )
            if any(entry["batch_id"] == adopt_batch for entry in self.saved.batches):
                raise ValueError(f"The batch {adopt_batch} is already in the state file.")
            self.backend._get_batch(adopt_batch)  # it must exist in this workspace
            each_tokens, each_cost = (
                self.ledger._worst_case(self.plan.model, self.backend.cache)
                if self.ledger is not None
                else (None, None)
            )
            self.saved.batches.append(
                {
                    "batch_id": adopt_batch,
                    "ask": pending["ask"],
                    "custom_ids": pending["custom_ids"],
                    "submitted_at": pending["started_at"],
                    "worst_case_tokens_each": each_tokens,
                    "worst_case_cost_each": each_cost,
                    "recorded": [],
                }
            )
            self.saved.unconfirmed = None
            self.saved.save()
            self.report(f"Batch {adopt_batch} adopted for the requests whose sending was lost.")
            return
        if resubmit_unconfirmed:
            self.saved.unconfirmed = None
            self.saved.save()
            self.report("The requests whose sending was lost will be sent again.")
            return
        raise self._unconfirmed_error(pending)

    # Reading the results ----------------------------------------------------------------------

    def _wait(self, entry: dict[str, Any]) -> dict[str, Any]:
        batch_id = entry["batch_id"]
        while True:
            data = self.backend._get_batch(batch_id)
            status = data.get("processing_status")
            if status == "ended":
                break
            if status not in ("in_progress", "canceling"):
                raise BackendError(
                    f"The API gave batch {batch_id} the state {status!r}, which reliacode does "
                    "not know. Check the batch in the Claude Console.",
                    reason="unexpected_answer",
                    endpoint=self.backend.batches_endpoint,
                )
            counts = data.get("request_counts")
            left = counts.get("processing") if isinstance(counts, dict) else None
            logger.info("Batch %s is %s (%s requests still processing).", batch_id, status, left)
            self.backend._sleep(self.poll_every)
        self.polled[batch_id] = data
        if data.get("archived_at"):
            raise _results_gone(batch_id, self.backend.batches_endpoint)
        return data

    def _finish(self, entry: dict[str, Any]) -> None:
        """Wait for a batch to end, then take every result it holds."""
        batch_id = entry["batch_id"]
        data = self._wait(entry)
        created, ended = _time(data.get("created_at")), _time(data.get("ended_at"))
        seconds = max((ended - created).total_seconds(), 0.0) if created and ended else 0.0
        counts = data.get("request_counts")
        counts = counts if isinstance(counts, dict) else {}
        took = f" after {_duration(seconds)}" if created and ended else ""
        shown = ", ".join(f"{_whole(counts.get(kind)) or 0:,} {kind}" for kind in _COUNTS)
        self.report(f"Batch {batch_id} ended{took}: {shown}.")
        planned = set(entry["custom_ids"])
        seen: set[str] = set()
        for item in self.backend._batch_results(batch_id):
            custom_id = item.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in planned:
                raise ValueError(
                    f"The batch {batch_id} holds a request that is not one of this run's, so "
                    "its results are not used. Check that adopt_batch names the right batch."
                )
            if custom_id in seen:
                continue
            seen.add(custom_id)
            self._take(entry, custom_id, item.get("result"), seconds, ended)
        for custom_id in entry["custom_ids"]:
            if custom_id not in seen:
                ask, number = _parse(custom_id)
                self._fail(entry, number, ask, "missing", None, record=False)
        entry["recorded"] = sorted(
            custom_id for custom_id in entry["custom_ids"] if (batch_id, custom_id) in self.recorded
        )
        self.saved.save()

    def _take(
        self,
        entry: dict[str, Any],
        custom_id: str,
        result: object,
        seconds: float,
        ended: datetime | None,
    ) -> None:
        batch_id = entry["batch_id"]
        ask, number = _parse(custom_id)
        task = self.tasks[number]
        result = result if isinstance(result, dict) else {}
        kind = result.get("type")
        if kind != "succeeded":
            reason = kind if kind in _FAILURES else "unexpected_answer"
            error = result.get("error")
            inner = error.get("error") if isinstance(error, dict) else None
            error_type = inner.get("type") if isinstance(inner, dict) else None
            self._fail(
                entry, number, ask, reason, error_type if isinstance(error_type, str) else None
            )
            return
        if task.done or task.ask != ask:  # pragma: no cover - a request is sent once per ask
            logger.warning(
                "Batch %s holds a second answer for %s; it is not used.", batch_id, custom_id
            )
            return
        _, parameters = self.backend._request(task.messages, task.work.prompt.json_schema)
        parameters = parameters.model_copy(update={"endpoint": self.backend.batches_endpoint})
        message = result.get("message")
        try:
            if not isinstance(message, dict):
                raise BackendError("not a message", reason="unexpected_answer")
            response = self.backend._response(
                message, parameters, latency=seconds, attempts=1, request_id=None, key=None
            )
        except BackendError:
            self._fail(entry, number, ask, "unexpected_answer", None)
            return
        response = response.model_copy(
            update={
                "batch_id": batch_id,
                "custom_id": custom_id,
                "created_at": ended or response.created_at,
            }
        )
        already = (batch_id, custom_id) in self.recorded
        self._failures.pop(number, None)
        try:
            task.take(response, record=not already)
        finally:
            self._recorded(entry, custom_id, already)

    def _fail(
        self,
        entry: dict[str, Any],
        number: int,
        ask: int,
        reason: str,
        error_type: str | None,
        *,
        record: bool = True,
    ) -> None:
        task = self.tasks[number]
        custom_id = _custom_id(ask, number)
        batch_id = entry["batch_id"]
        self._failures[number] = BatchFailure(
            unit=task.unit,
            variable=task.work.label,
            ask=ask,
            batch_id=batch_id,
            custom_id=custom_id,
            reason=reason,  # type: ignore[arg-type] - one of the listed reasons
            error_type=error_type,
        )
        if not record:
            return
        already = (batch_id, custom_id) in self.recorded
        ledger = self.ledger
        if ledger is not None and not already:
            label = "unexpected_answer" if reason == "unexpected_answer" else f"batch_{reason}"
            ledger.record(
                LedgerEntry.from_failure(
                    BackendError(label, reason=label),
                    model=self.plan.model,
                    text=task.text,
                    ask=ask,
                    batch_id=batch_id,
                    custom_id=custom_id,
                    **_ties(task.work, ledger),
                )
            )
        self._recorded(entry, custom_id, already)

    def _recorded(self, entry: dict[str, Any], custom_id: str, already: bool) -> None:
        if already:
            return
        self.recorded.add((entry["batch_id"], custom_id))
        self._give_back(1, entry.get("worst_case_tokens_each"), entry.get("worst_case_cost_each"))

    # The reserve ------------------------------------------------------------------------------

    def _reserve_sent(self) -> None:
        """Keep back what the batches sent by an earlier call may still use."""
        if self.ledger is None:
            return
        for entry in self.saved.batches:
            left = [
                custom_id
                for custom_id in entry["custom_ids"]
                if (entry["batch_id"], custom_id) not in self.recorded
            ]
            if left:
                tokens, cost = (
                    entry.get("worst_case_tokens_each"),
                    entry.get("worst_case_cost_each"),
                )
                self.ledger._reserve(
                    len(left),
                    None if tokens is None else tokens * len(left),
                    None if cost is None else cost * len(left),
                    refuse=False,
                )

    def _give_back(self, count: int, each_tokens: int | None, each_cost: float | None) -> None:
        if self.ledger is None or count == 0:
            return
        self.ledger._release(
            count,
            None if each_tokens is None else each_tokens * count,
            None if each_cost is None else each_cost * count,
        )

    # Lookups ----------------------------------------------------------------------------------

    def _sent(self, number: int, ask: int) -> bool:
        return self._last_batch(number, ask) is not None

    def _last_batch(self, number: int, ask: int) -> str | None:
        custom_id = _custom_id(ask, number)
        found = None
        for entry in self.saved.batches:
            if entry["ask"] == ask and custom_id in entry["custom_ids"]:
                found = entry["batch_id"]
        return found


_COUNTS = ("succeeded", "errored", "canceled", "expired")


def _custom_id(ask: int, number: int) -> str:
    """The custom id of a task's request for an ask: the ask and the task's number from 1."""
    return f"a{ask}-t{number + 1}"


def _parse(custom_id: str) -> tuple[int, int]:
    match = _CUSTOM_ID.fullmatch(custom_id)
    assert match is not None  # every custom id read is one of the run's own
    return int(match.group(1)), int(match.group(2)) - 1


def _chunks(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The requests in batches of at most _MAX_REQUESTS requests and _MAX_BYTES bytes."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for request in requests:
        length = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + 2
        if current and (len(current) >= _MAX_REQUESTS or size + length > _MAX_BYTES):
            chunks.append(current)
            current, size = [], 0
        current.append(request)
        size += length
    if current:
        chunks.append(current)
    return chunks


def _fingerprint(plan: _Plan, backend: AnthropicBackend) -> str:
    """A hash of everything that decides a run's requests, so a state file fits one run."""
    payload = {
        "reliacode_batch_state": STATE_VERSION,
        "codebook_hash": plan.codebook_hash,
        "recipe": plan.recipe,
        "attempts": plan.attempts,
        "prompts": [prompt.hash for prompt in plan.prompts.values()],
        "backend": backend._settings(),
        "texts": [[unit, hash_text(text)] for unit, text in plan.units],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _state_path(state: object) -> Path:
    if (isinstance(state, str) and state.strip()) or isinstance(state, os.PathLike):
        path = Path(state)
    else:
        raise TypeError('state should be a file name, such as "run-state.json".')
    if path.is_dir():
        raise ValueError(f"The state path {path} is a folder. Give a file name inside it.")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"The folder for the state file does not exist: {path.parent}. Create it first."
        )
    return path


def _now() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _time(value: object) -> datetime | None:
    """A time the API gave (RFC 3339), in UTC, or None."""
    if not isinstance(value, str):
        return None
    match = _TIME.fullmatch(value.strip())
    if match is None:
        return None
    whole, fraction, zone = match.groups()
    fraction = (fraction or "")[:7]  # at most microseconds
    zone = "+00:00" if zone in (None, "Z") else zone
    return datetime.fromisoformat(whole + fraction + zone).astimezone(timezone.utc)


def _whole(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def _requests(count: int) -> str:
    return "request" if count == 1 else "requests"

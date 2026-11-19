"""The ledger: one line for every call to a model, with what it used and what it cost.

Each line records one call. It says when the call was made, which model was asked and which one
answered, and the settings used. It gives the token counts the server reported, the latency,
the tries and the outcome. It also records what ties the call to the work: the codebook's
content hash, the prompt hash, the variable, and a hash of the text with its length. A line
never holds the text that was coded, the model's answer or an API key, so a ledger can be
shared even when the texts cannot.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationError,
    field_validator,
    model_validator,
)

from reliacode.backends import BackendError, RequestParameters, Response, utc_now

#: Version of the ledger's line format. Every line records the version it was written with,
#: and a line from a newer version is refused when the file is read.
LEDGER_VERSION = 1

#: What came of a call. "answered": the answer could be used. "asked_again": it could not, and
#: the model was asked again. "missing": it could not, and no try was left. "failed": no answer
#: came at all.
Outcome = Literal["answered", "asked_again", "missing", "failed"]

_HASH = r"^sha256:[0-9a-f]{64}$"
#: What a field name looks like, so that an error never quotes anything else from a line.
_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,39}")


def hash_text(text: str) -> str:
    """The SHA-256 hash of ``text`` (in UTF-8, exactly as given), written ``"sha256:<hex>"``.

    It ties a ledger line to the text that was coded without keeping the text. Anyone who has
    the text can check that a line belongs to it, but the hash does not give the text back. A
    very short or common text can still be guessed and checked, so share the hashes of such
    texts with the same care as the texts.
    """
    if not isinstance(text, str):
        raise TypeError("hash_text() takes a text.")
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LedgerEntry(_Record):
    """One call to a model: one line of the ledger.

    The token counts are the ones the server reported, and ``token_source`` is then
    ``"server"``. When the server reported none, the counts are ``None``: never zero and never
    estimated. A failed call got no answer, so it has no counts, no returned model and no
    latency, and ``error_reason`` says what kind of problem stopped it.

    The line never holds the text that was coded, the model's answer or an API key. The text is
    represented only by ``text_hash`` and ``text_length`` (in characters).
    """

    schema_version: int = LEDGER_VERSION
    at: datetime
    outcome: Outcome
    variable: str
    ask: int = Field(default=1, ge=1)
    model_requested: str | None = None
    model_returned: str | None = None
    parameters: RequestParameters | None = None
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None
    token_source: Literal["server"] | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    attempts: int | None = Field(default=None, ge=1)
    error_reason: str | None = None
    http_status: int | None = None
    codebook_hash: str = Field(pattern=_HASH)
    prompt_hash: str = Field(pattern=_HASH)
    text_hash: str = Field(pattern=_HASH)
    text_length: NonNegativeInt

    @field_validator("at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("at needs a time zone; use datetime.now(timezone.utc)")
        return value.astimezone(timezone.utc)

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value < 1 or value > LEDGER_VERSION:
            raise ValueError(
                f"this line was written with ledger version {value}, and this version of "
                f"reliacode reads version {LEDGER_VERSION}. Update reliacode to read it."
            )
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "LedgerEntry":
        counted = self.prompt_tokens is not None or self.completion_tokens is not None
        if counted and self.token_source is None:
            raise ValueError('token counts need token_source="server", where they came from')
        if not counted and self.token_source is not None:
            raise ValueError("token_source is set, but there are no token counts")
        if self.outcome == "failed":
            if self.error_reason is None:
                raise ValueError("a failed call needs an error_reason")
            if counted:
                raise ValueError("a failed call got no answer, so it has no token counts")
        elif self.error_reason is not None or self.http_status is not None:
            raise ValueError("only a failed call has an error_reason or an http_status")
        return self

    @property
    def counted(self) -> bool:
        """True when the server reported both token counts."""
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def model(self) -> str | None:
        """The model asked for, or else the model the server returned."""
        return self.model_requested or self.model_returned

    @classmethod
    def from_response(
        cls,
        response: Response,
        *,
        outcome: Outcome,
        variable: str,
        codebook_hash: str,
        prompt_hash: str,
        text: str,
        ask: int = 1,
        at: datetime | None = None,
    ) -> "LedgerEntry":
        """The line for a call that got an answer. ``text`` is hashed and measured, not kept.

        ``at`` is the time of the call; the answer's own time is used when it is not given.
        """
        counted = response.prompt_tokens is not None or response.completion_tokens is not None
        return cls(
            at=at if at is not None else response.created_at,
            outcome=outcome,
            variable=variable,
            ask=ask,
            model_requested=response.parameters.model,
            model_returned=response.model,
            parameters=response.parameters,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            token_source="server" if counted else None,
            latency_seconds=response.latency_seconds,
            finish_reason=response.finish_reason,
            response_id=response.response_id,
            request_id=response.request_id,
            attempts=response.attempts,
            codebook_hash=codebook_hash,
            prompt_hash=prompt_hash,
            text_hash=hash_text(text),
            text_length=len(text),
        )

    @classmethod
    def from_failure(
        cls,
        error: BaseException,
        *,
        variable: str,
        codebook_hash: str,
        prompt_hash: str,
        text: str,
        model: str | None = None,
        ask: int = 1,
        at: datetime | None = None,
    ) -> "LedgerEntry":
        """The line for a call that got no answer. ``text`` is hashed and measured, not kept.

        Only the kind of problem is kept: a :class:`~reliacode.BackendError`'s ``reason`` and
        HTTP status, or else the name of the exception. The error's message is never kept,
        because a server's message can quote the request. ``model`` is the model that was asked
        for, when it is known. ``at`` is the time of the call; now is used when it is not given.
        """
        if isinstance(error, BackendError):
            reason, status = error.reason, error.status
        else:
            reason, status = type(error).__name__, None
        return cls(
            at=at if at is not None else utc_now(),
            outcome="failed",
            variable=variable,
            ask=ask,
            model_requested=model,
            error_reason=reason,
            http_status=status,
            codebook_hash=codebook_hash,
            prompt_hash=prompt_hash,
            text_hash=hash_text(text),
            text_length=len(text),
        )


class ModelPrice(_Record):
    """What one model costs: the price of a million input tokens and of a million output tokens.

    Write 0 for both for a local model that costs nothing to run, so that its calls cost 0 and
    their tokens are still counted.
    """

    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)


class PriceTable(_Record):
    """The prices of the models you use, in one currency, as you read them on one day.

    reliacode ships no prices. Prices change, and a wrong price built into a package is worse
    than none, so you write down the prices you read and the date you read them::

        prices = PriceTable(
            currency="USD",
            read_on="2026-11-19",
            prices={
                "example-hosted-model": ModelPrice(input_per_million=0.50, output_per_million=1.50),
                "example-local-model": ModelPrice(input_per_million=0, output_per_million=0),
            },
        )

    (The model names and prices above are invented.) A model's price is looked up under the
    name that was asked for, and then under the name the server returned. A model that is not
    in the table has its tokens counted and no cost.
    """

    currency: str = Field(min_length=1)
    read_on: date
    prices: dict[str, ModelPrice]

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('give the currency of the prices, such as "USD" or "EUR"')
        return value

    @field_validator("prices")
    @classmethod
    def _models(cls, value: dict[str, ModelPrice]) -> dict[str, ModelPrice]:
        if not value:
            raise ValueError("the price table lists no model; add the price of each model used")
        if any(not name.strip() for name in value):
            raise ValueError("a model in the price table has an empty name")
        return value

    def price(self, model: str | None) -> ModelPrice | None:
        """The price of ``model``, or ``None`` when the table has none."""
        return self.prices.get(model) if model is not None else None

    def price_of(self, entry: LedgerEntry) -> ModelPrice | None:
        """The price for a ledger line: under the model asked for, else the model returned."""
        return self.price(entry.model_requested) or self.price(entry.model_returned)

    def cost(self, entry: LedgerEntry) -> float | None:
        """What a call cost, in the table's currency.

        It is ``None``, never zero, when the server did not report both token counts or the
        table has no price for the model. A failed call has no counts, so it has no cost.
        """
        price = self.price_of(entry)
        if price is None or entry.prompt_tokens is None or entry.completion_tokens is None:
            return None
        return (
            entry.prompt_tokens * price.input_per_million
            + entry.completion_tokens * price.output_per_million
        ) / 1_000_000


class Budget(_Record):
    """Limits that the calls in a ledger may not pass: a cost, a number of tokens, a number of
    calls, or any of them together.

    The ledger checks the budget before every call, and refuses the call that would pass a
    limit with a :class:`BudgetError`. The budget covers every call in the ledger file,
    including calls recorded by earlier runs.

    Args:
        cost_limit: the most the calls may cost, in the currency of the ledger's price table.
        token_limit: the most tokens the calls may use, input and output together.
        call_limit: the most calls that may be made.
    """

    cost_limit: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    token_limit: int | None = Field(default=None, gt=0)
    call_limit: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _has_a_limit(self) -> "Budget":
        if self.cost_limit is None and self.token_limit is None and self.call_limit is None:
            raise ValueError(
                "a budget needs at least one limit, such as Budget(cost_limit=5) or "
                "Budget(call_limit=1000)"
            )
        return self


class BudgetError(RuntimeError):
    """The budget stopped a call before it was made, or it cannot be kept.

    Attributes:
        limit: which limit stopped the call: ``"cost"``, ``"tokens"`` or ``"calls"``.
        allowed: the limit itself.
        used: what the calls in the ledger have used so far.
        run: when :func:`reliacode.code` was stopped, a :class:`~reliacode.CodingRun` with the
            results finished before the stop, so codes already paid for are not lost.
            Otherwise ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        limit: Literal["cost", "tokens", "calls"],
        allowed: float,
        used: float,
        run: object = None,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.allowed = allowed
        self.used = used
        self.run = run


#: How many unreadable lines a LedgerError lists before it only counts the rest.
_LISTED = 10


class LedgerError(ValueError):
    """A ledger file has lines that cannot be read. The message lists each one by its number.

    Attributes:
        path: the ledger file.
        problems: ``(line number, what is wrong)`` for every line that cannot be read.
    """

    def __init__(self, path: Path, problems: list[tuple[int, str]]) -> None:
        self.path = path
        self.problems = tuple(problems)
        count = len(problems)
        lines = [
            f"The ledger file {path} has {count} {'line' if count == 1 else 'lines'} that cannot "
            "be read:"
        ]
        lines += [f"  Line {number}: {problem}" for number, problem in problems[:_LISTED]]
        if count > _LISTED:
            lines.append(f"  ... and {count - _LISTED} more.")
        lines.append(
            "Keep a copy of the file, then fix or remove these lines in a text editor. reliacode "
            "only adds lines to a ledger and never changes the ones already there."
        )
        super().__init__("\n".join(lines))


def read_ledger(path: str | os.PathLike[str]) -> tuple[LedgerEntry, ...]:
    """Read every line of a ledger file, in the order the calls were made.

    Blank lines are skipped. Every other line must be a ledger line.

    Raises:
        FileNotFoundError: there is no file at ``path``.
        LedgerError: some lines cannot be read. The error lists each one with its line number,
            so that no call is ever dropped without notice.
    """
    path = _checked_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"There is no ledger file at {path}. Check the path.")
    entries: list[LedgerEntry] = []
    problems: list[tuple[int, str]] = []
    with path.open("rb") as file:
        for number, raw in enumerate(file, 1):
            entry, problem = _read_line(raw)
            if problem is not None:
                problems.append((number, problem))
            elif entry is not None:
                entries.append(entry)
    if problems:
        raise LedgerError(path, problems)
    return tuple(entries)


def _read_line(raw: bytes) -> tuple[LedgerEntry | None, str | None]:
    """One line of the file as a ledger line, or what is wrong with it. A blank line is neither."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "it is not UTF-8 text."
    if not text.strip():
        return None, None
    try:
        data = json.loads(text)
    except ValueError:
        return None, (
            "it is not a complete JSON object. A line cut off when a program stopped while "
            "writing looks like this."
        )
    if not isinstance(data, dict):
        return None, "it is not a JSON object, so it is not a ledger line."
    version = data.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version > LEDGER_VERSION:
        return None, (
            f"it was written by a newer version of reliacode (ledger version {version}). Update "
            "reliacode to read it."
        )
    try:
        return LedgerEntry.model_validate(data), None
    except ValidationError as exc:
        return None, _validation_problem(exc)


def _validation_problem(exc: ValidationError) -> str:
    """What is wrong with a line, without quoting any value from it."""
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    parts = []
    for error in errors[:3]:
        field = ".".join(str(part) for part in error["loc"])
        if error["type"] == "extra_forbidden" and not _FIELD_NAME.fullmatch(field):
            field = "a field"  # a name that does not look like one is not quoted either
        message = error["msg"].removeprefix("Value error, ")
        parts.append(f"{field}: {message}" if field else message)
    more = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
    return f"it is not a valid ledger line. {'; '.join(parts)}{more}."


def _checked_path(path: object) -> Path:
    if isinstance(path, str) and path.strip():
        return Path(path)
    if isinstance(path, os.PathLike):
        return Path(path)
    raise TypeError('The ledger path should be a file name, such as "ledger.jsonl".')


class Ledger:
    """A ledger file: one JSON line for every call to a model, added as each call ends.

    Pass a ledger to :func:`reliacode.code` and every call it makes is recorded, including
    second asks and failures::

        ledger = Ledger("ledger.jsonl", prices=prices, budget=Budget(cost_limit=5))
        run = code(codebook, texts, backend=backend, ledger=ledger)

    The file is created if it does not exist. Lines already in it are kept, and new lines are
    added at the end, each written and flushed as its call ends, so a run that stops loses no
    line. reliacode never rewrites a line. Only one program should write to a ledger file at a
    time: two writers can mix their lines.

    Args:
        path: the ledger file, such as ``"ledger.jsonl"``. Its folder must exist.
        prices: the :class:`PriceTable` that gives each call its cost. Without one, calls have
            tokens and no cost.
        budget: the :class:`Budget` to keep. It covers every call in the file.
        clock: a function that returns the current time in UTC, used for the time of each
            line. Tests replace it.

    Raises:
        LedgerError: the file has lines that cannot be read. New lines are never added after
            a line that cannot be read.
        ValueError: the budget has a cost limit but there is no price table.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        prices: PriceTable | None = None,
        budget: Budget | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.path = _checked_path(path)
        if prices is not None and not isinstance(prices, PriceTable):
            raise TypeError("prices should be a PriceTable(currency=..., read_on=..., prices=...).")
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError("budget should be a Budget, such as Budget(cost_limit=5).")
        if budget is not None and budget.cost_limit is not None and prices is None:
            raise ValueError(
                "The budget has a cost limit, but the ledger has no prices, so the cost of a "
                "call cannot be known. Pass the prices you use, as in Ledger(path, prices="
                "PriceTable(...), budget=...), or use a token or call limit instead."
            )
        self.prices = prices
        self.budget = budget
        self.clock = clock
        if self.path.is_dir():
            raise ValueError(
                f"The ledger path {self.path} is a folder. Give a file name inside it, such as "
                f"{self.path / 'ledger.jsonl'}."
            )
        if not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"The folder for the ledger file does not exist: {self.path.parent}. Create it "
                "first, or choose another path."
            )
        self._calls = 0
        self._tokens = 0
        self._cost = 0.0
        self._largest_tokens = 0
        self._largest_cost = 0.0
        #: The first answered call without both token counts, and the first without a price.
        self._without_tokens: tuple[int, LedgerEntry] | None = None
        self._without_price: tuple[int, LedgerEntry] | None = None
        self._needs_line_end = False
        if self.path.exists():
            for entry in read_ledger(self.path):
                self._count(entry)
            self._needs_line_end = _ends_without_line_end(self.path)
        self.path.touch()

    @property
    def calls(self) -> int:
        """The number of calls recorded in the file, including those from earlier runs."""
        return self._calls

    def record(self, entry: LedgerEntry) -> None:
        """Add one line to the end of the file and flush it to disk."""
        if not isinstance(entry, LedgerEntry):
            raise TypeError("A ledger records LedgerEntry lines only.")
        line = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            if self._needs_line_end:
                file.write("\n")
            file.write(line + "\n")
            file.flush()
        self._needs_line_end = False
        self._count(entry)

    def entries(self) -> tuple[LedgerEntry, ...]:
        """Every line in the file, read again from disk."""
        return read_ledger(self.path)

    def check(self) -> None:
        """Refuse the next call if it would pass a limit of the budget. Run it before each call.

        A call's size is not known before it is made, so the next call is taken to be as large
        as the largest call so far. The first call is made unless a limit is already reached.

        Raises:
            BudgetError: the next call would pass a limit, or a limit cannot be kept because
                a call had no token counts or its model has no price.
        """
        budget = self.budget
        if budget is None:
            return
        if budget.call_limit is not None and self._calls + 1 > budget.call_limit:
            raise BudgetError(
                f"The budget allows {budget.call_limit:,} calls, and the ledger {self.path} "
                f"already has {self._calls:,}, so reliacode stopped before the next call. To "
                "make more calls, raise call_limit in the Budget.",
                limit="calls",
                allowed=budget.call_limit,
                used=self._calls,
            )
        if budget.token_limit is not None:
            self._check_counted("token limit", f"{budget.token_limit:,}", "tokens")
            if self._tokens + self._largest_tokens > budget.token_limit:
                raise BudgetError(
                    f"The next call could pass the token limit of {budget.token_limit:,}, so "
                    f"reliacode stopped before it. The calls in the ledger {self.path} have used "
                    f"{self._tokens:,} tokens (input and output together), and the largest one "
                    f"used {self._largest_tokens:,}. To go on, raise token_limit in the Budget.",
                    limit="tokens",
                    allowed=budget.token_limit,
                    used=self._tokens,
                )
        if budget.cost_limit is not None:
            limit = self._amount(budget.cost_limit)
            self._check_counted("cost limit", limit, "cost")
            if self._without_price is not None:
                number, entry = self._without_price
                raise self._no_price(entry.model, f" (call {number:,} in the ledger {self.path})")
            if self._cost + self._largest_cost > budget.cost_limit:
                raise BudgetError(
                    f"The next call could pass the cost limit of {limit}, so reliacode stopped "
                    f"before it. The calls in the ledger {self.path} have cost "
                    f"{self._amount(self._cost)} so far, and the largest one cost "
                    f"{self._amount(self._largest_cost)}. To go on, raise cost_limit in the "
                    "Budget.",
                    limit="cost",
                    allowed=budget.cost_limit,
                    used=self._cost,
                )

    def check_price(self, model: str | None) -> None:
        """Refuse a cost limit at the start of a run when the model to be used has no price.

        Nothing is checked when ``model`` is ``None`` (not known) or there is no cost limit.

        Raises:
            BudgetError: the budget has a cost limit and the price table has no price for
                ``model``.
        """
        if self.budget is None or self.budget.cost_limit is None or model is None:
            return
        if self.prices is not None and self.prices.price(model) is None:
            raise self._no_price(model, "")

    def _check_counted(self, name: str, limit: str, kind: Literal["cost", "tokens"]) -> None:
        if self._without_tokens is None:
            return
        number, entry = self._without_tokens
        fake = entry.parameters is not None and entry.parameters.endpoint == "fake"
        hint = " FakeBackend reports token counts only when it is given tokens=..." if fake else ""
        raise BudgetError(
            f"The {name} of {limit} cannot be kept: the server reported no token counts for "
            f'call {number:,} in the ledger {self.path} (model "{entry.model}"), so reliacode '
            f"cannot tell what the calls use. Use a call limit instead, as in "
            f"Budget(call_limit=1000), or a server that reports token counts.{hint}",
            limit=kind,
            allowed=self._limit(kind),
            used=self._tokens if kind == "tokens" else self._cost,
        )

    def _no_price(self, model: str | None, where: str) -> BudgetError:
        limit = self._limit("cost")
        return BudgetError(
            f"The cost limit of {self._amount(limit)} cannot be kept: the price table has no "
            f'price for the model "{model}"{where}, so the cost of its calls is not known. Add '
            "the model to the price table (write 0 for a local model that costs nothing), or "
            "use a token or call limit instead.",
            limit="cost",
            allowed=limit,
            used=self._cost,
        )

    def _limit(self, kind: Literal["cost", "tokens"]) -> float:
        budget = self.budget
        assert budget is not None  # only called while a budget is checked
        value = budget.cost_limit if kind == "cost" else budget.token_limit
        assert value is not None
        return value

    def _amount(self, value: float) -> str:
        assert self.prices is not None  # a cost limit always comes with prices
        return f"{_number(value)} {self.prices.currency}"

    def _count(self, entry: LedgerEntry) -> None:
        self._calls += 1
        if entry.outcome == "failed":
            return  # a failed call counts toward the call limit only
        if entry.prompt_tokens is None or entry.completion_tokens is None:
            self._without_tokens = self._without_tokens or (self._calls, entry)
            return
        tokens = entry.prompt_tokens + entry.completion_tokens
        self._tokens += tokens
        self._largest_tokens = max(self._largest_tokens, tokens)
        if self.prices is None:
            return
        cost = self.prices.cost(entry)
        if cost is None:
            self._without_price = self._without_price or (self._calls, entry)
            return
        self._cost += cost
        self._largest_cost = max(self._largest_cost, cost)

    def __repr__(self) -> str:
        return f"Ledger({str(self.path)!r}, calls={self._calls})"


def _number(value: float) -> str:
    """A cost written with as many decimals as it needs (up to 8) and at least 2."""
    text = f"{value:,.8f}".rstrip("0")
    whole, _, decimals = text.partition(".")
    return f"{whole}.{decimals.ljust(2, '0')}"


def _ends_without_line_end(path: Path) -> bool:
    """True when the file's last line has no line end, so a new line must start with one."""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            return False
        file.seek(-1, os.SEEK_END)
        return file.read(1) != b"\n"

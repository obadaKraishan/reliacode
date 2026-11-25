"""The ledger: one line for every call to a model, with what it used and what it cost.

Each line records one call. It says when the call was made, which model was asked and which one
answered, and the settings used. It gives the token counts the server reported, the latency,
the tries and the outcome. It also records what ties the call to the work: the codebook's
content hash, the prompt hash, the variable, and a hash of the text with its length. A line
never holds the text that was coded, the model's answer or an API key, so a ledger can be
shared even when the texts cannot.
"""

import csv
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
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

from reliacode.backends import (
    BackendError,
    CredentialFieldError,
    RequestParameters,
    Response,
    utc_now,
)

#: Version of the ledger's line format. Every line records the version it was written with,
#: and a line from a newer version is refused when the file is read. Version 2 added
#: ``reasoning_tokens`` and the extra request fields (``parameters.extra_body``). Version 3
#: added the cache counts (``cache_write_tokens``, ``cache_write_1h_tokens``,
#: ``cache_read_tokens``), the batch ids (``batch_id``, ``custom_id``) and four request settings
#: (``parameters.api_version``, ``thinking``, ``effort`` and ``cache``). Lines written with
#: versions 1 and 2 are still read, with the fields they lack left empty.
LEDGER_VERSION = 3

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
    estimated. ``reasoning_tokens`` counts the tokens a model spent reasoning, when the server
    reports them; they are part of ``completion_tokens``, so costs count them once. A failed
    call got no answer, so it has no counts, no returned model and no latency, and
    ``error_reason`` says what kind of problem stopped it.

    ``cache_read_tokens`` and ``cache_write_tokens`` are the input tokens read from and written
    to a prompt cache, and ``cache_write_1h_tokens`` the written ones that went to a one-hour
    cache, as the server reported them. They are part of ``prompt_tokens``. ``batch_id`` and
    ``custom_id`` name the batch and the request within it, for a call sent in a batch; such a
    call has no latency of its own, so its ``latency_seconds`` is ``None``.

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
    reasoning_tokens: NonNegativeInt | None = None
    cache_write_tokens: NonNegativeInt | None = None
    cache_write_1h_tokens: NonNegativeInt | None = None
    cache_read_tokens: NonNegativeInt | None = None
    batch_id: str | None = None
    custom_id: str | None = None

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
        counted = any(count is not None for count in _counts(self))
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
        counted = any(count is not None for count in _counts(response))
        batched = response.batch_id is not None
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
            # A request in a batch has no latency of its own: the batch's times are its waits.
            latency_seconds=None if batched else response.latency_seconds,
            finish_reason=response.finish_reason,
            response_id=response.response_id,
            request_id=response.request_id,
            attempts=response.attempts,
            codebook_hash=codebook_hash,
            prompt_hash=prompt_hash,
            text_hash=hash_text(text),
            text_length=len(text),
            reasoning_tokens=response.reasoning_tokens,
            cache_write_tokens=response.cache_write_tokens,
            cache_write_1h_tokens=response.cache_write_1h_tokens,
            cache_read_tokens=response.cache_read_tokens,
            batch_id=response.batch_id,
            custom_id=response.custom_id,
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
        batch_id: str | None = None,
        custom_id: str | None = None,
    ) -> "LedgerEntry":
        """The line for a call that got no answer. ``text`` is hashed and measured, not kept.

        Only the kind of problem is kept: a :class:`~reliacode.BackendError`'s ``reason`` and
        HTTP status, or else the name of the exception. The error's message is never kept,
        because a server's message can quote the request. ``model`` is the model that was asked
        for, when it is known. ``at`` is the time of the call; now is used when it is not given.
        ``batch_id`` and ``custom_id`` name the request, for a request in a batch.
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
            batch_id=batch_id,
            custom_id=custom_id,
        )


def _counts(record: "LedgerEntry | Response") -> tuple[int | None, ...]:
    """Every token count a record can hold."""
    return (
        record.prompt_tokens,
        record.completion_tokens,
        record.reasoning_tokens,
        record.cache_write_tokens,
        record.cache_write_1h_tokens,
        record.cache_read_tokens,
    )


_PRICE = {"ge": 0, "allow_inf_nan": False}


class ModelPrice(_Record):
    """What one model costs: the price of a million input tokens and of a million output tokens.

    Write 0 for both for a local model that costs nothing to run, so that its calls cost 0 and
    their tokens are still counted.

    A service with a prompt cache or a batch interface can give more prices:

    Args:
        cache_write_5m_per_million, cache_write_1h_per_million: the price of a million input
            tokens written to a cache that lasts five minutes or one hour. A call that wrote to
            a cache whose price is not given has no cost, since a write costs more than plain
            input.
        cache_read_per_million: the price of a million input tokens read from a cache. Without
            it, tokens read from a cache are priced as plain input, as before.
        batch_discount: the share taken off every price for a request sent in a batch, such as
            0.5 for half price. Without it, a request in a batch costs what it would cost alone.
        long_prompt_above, long_prompt_price: for a model priced by prompt length, the prompt
            length in tokens (cache reads and writes included) above which a call pays the
            prices in ``long_prompt_price`` instead, for all of its tokens.
    """

    input_per_million: float = Field(**_PRICE)
    output_per_million: float = Field(**_PRICE)
    cache_write_5m_per_million: float | None = Field(default=None, **_PRICE)
    cache_write_1h_per_million: float | None = Field(default=None, **_PRICE)
    cache_read_per_million: float | None = Field(default=None, **_PRICE)
    batch_discount: float | None = Field(default=None, ge=0, lt=1, allow_inf_nan=False)
    long_prompt_above: int | None = Field(default=None, gt=0)
    long_prompt_price: "ModelPrice | None" = None

    @model_validator(mode="after")
    def _long_prompt(self) -> "ModelPrice":
        if (self.long_prompt_above is None) != (self.long_prompt_price is None):
            raise ValueError(
                "give long_prompt_above and long_prompt_price together: the prompt length above "
                "which the long-prompt prices apply, and those prices"
            )
        inner = self.long_prompt_price
        if inner is not None and inner.long_prompt_price is not None:
            raise ValueError("long_prompt_price cannot have long-prompt prices of its own")
        return self

    def cost(self, entry: "LedgerEntry") -> float | None:
        """What the call in ``entry`` cost at these prices, or ``None`` when it cannot be known.

        The input is priced in parts: the plain input, the tokens written to a cache (at the
        price for the cache's time to live) and the tokens read from a cache. The output is
        priced as output. A call in a batch gets the batch discount.
        """
        prompt, completion = entry.prompt_tokens, entry.completion_tokens
        if prompt is None or completion is None:
            return None
        price = self
        if self.long_prompt_above is not None and prompt > self.long_prompt_above:
            assert self.long_prompt_price is not None  # given together, as checked above
            price = self.long_prompt_price
        parameters = entry.parameters
        cache_ttl = parameters.cache if parameters is not None else None
        written, read = entry.cache_write_tokens, entry.cache_read_tokens
        if cache_ttl is not None and (written is None or read is None):
            return None  # caching was asked for, and the server did not say what it did
        written, read = written or 0, read or 0
        plain = prompt - written - read
        if plain < 0:
            return None  # counts that do not add up are not priced
        total = plain * price.input_per_million + completion * price.output_per_million
        if read:
            read_price = price.cache_read_per_million
            total += read * (price.input_per_million if read_price is None else read_price)
        if written:
            one_hour = entry.cache_write_1h_tokens
            if one_hour is None:
                if cache_ttl not in ("5m", "1h"):
                    return None  # which cache the tokens went to is not known
                one_hour = written if cache_ttl == "1h" else 0
            five_minutes = written - one_hour
            if five_minutes < 0:
                return None
            for tokens, write_price in (
                (five_minutes, price.cache_write_5m_per_million),
                (one_hour, price.cache_write_1h_per_million),
            ):
                if tokens and write_price is None:
                    return None
                total += tokens * (write_price or 0)
        cost = total / 1_000_000
        if entry.batch_id is not None and price.batch_discount is not None:
            cost *= 1 - price.batch_discount
        return cost


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
        """What a call cost, in the table's currency (see :meth:`ModelPrice.cost`).

        It is ``None``, never zero, when the server did not report both token counts or the
        table has no price for the model. A failed call has no counts, so it has no cost.
        """
        price = self.price_of(entry)
        return None if price is None else price.cost(entry)


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
    except CredentialFieldError:
        return None, (
            "it is not a valid ledger line. parameters.extra_body: a field name looks like a "
            "credential, and reliacode never records such a field."
        )


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
        #: The largest input and output of a counted call, for the estimate before a batch.
        self._largest_prompt: int | None = None
        self._largest_completion = 0
        #: What batches already sent may still use: kept back from the budget until their
        #: results are recorded (see reliacode.code_in_batches).
        self._reserved_calls = 0
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
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

    def summary(self) -> "LedgerSummary":
        """The totals of every call in the file, costed with the ledger's prices."""
        return summarize(self.entries(), self.prices)

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
        if budget.call_limit is not None and self._calls + self._reserved_calls + 1 > (
            budget.call_limit
        ):
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
            if self._tokens + self._reserved_tokens + self._largest_tokens > budget.token_limit:
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
            if self._cost + self._reserved_cost + self._largest_cost > budget.cost_limit:
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

    def _worst_case(self, model: str | None, cache: str | None) -> tuple[int | None, float | None]:
        """The tokens and the cost of one request in a batch, at the worst case: as large as the
        largest call in the ledger so far, with all of its input written to the cache (or
        priced as plain input without caching) and none read, at the batch price. Either is
        ``None`` when it cannot be known: no call has token counts yet, or a price is missing.
        """
        if self._largest_prompt is None:
            return None, None
        prompt, completion = self._largest_prompt, self._largest_completion
        price = self.prices.price(model) if self.prices is not None else None
        cost = None
        if price is not None:
            if price.long_prompt_above is not None and prompt > price.long_prompt_above:
                assert price.long_prompt_price is not None
                price = price.long_prompt_price
            input_price = {
                None: price.input_per_million,
                "5m": price.cache_write_5m_per_million,
                "1h": price.cache_write_1h_per_million,
            }.get(cache)
            if input_price is not None:
                cost = (prompt * input_price + completion * price.output_per_million) / 1_000_000
                if price.batch_discount is not None:
                    cost *= 1 - price.batch_discount
        return prompt + completion, cost

    def _reserve(
        self, calls: int, tokens: int | None, cost: float | None, *, refuse: bool = True
    ) -> None:
        """Keep back what a batch about to be sent may use, after checking that the budget
        allows it on top of the calls recorded and the batches already sent.

        ``tokens`` and ``cost`` are the batch's worst case, or ``None`` when it is not known.
        With ``refuse=False`` nothing is checked: for batches that were already sent.

        Raises:
            BudgetError: the batch could pass a limit, or a limit cannot be kept.
        """
        budget = self.budget
        if budget is not None and refuse:
            sent = self._reserved_calls
            if budget.call_limit is not None and self._calls + sent + calls > budget.call_limit:
                raise BudgetError(
                    f"The budget allows {budget.call_limit:,} calls. The ledger {self.path} has "
                    f"{self._calls:,}, batches already sent may make {sent:,} more, and this "
                    f"batch holds {calls:,}, so reliacode did not send it. To send it, raise "
                    "call_limit in the Budget.",
                    limit="calls",
                    allowed=budget.call_limit,
                    used=self._calls,
                )
            if budget.token_limit is not None:
                self._check_counted("token limit", f"{budget.token_limit:,}", "tokens")
                if tokens is None:
                    raise self._no_size("token limit", f"{budget.token_limit:,}", "tokens")
                if self._tokens + self._reserved_tokens + tokens > budget.token_limit:
                    raise BudgetError(
                        f"This batch could pass the token limit of {budget.token_limit:,}, so "
                        f"reliacode did not send it. Its worst case is {tokens:,} tokens; the "
                        f"calls in the ledger {self.path} have used {self._tokens:,}, and "
                        f"batches already sent may use {self._reserved_tokens:,} more. To send "
                        "it, raise token_limit in the Budget.",
                        limit="tokens",
                        allowed=budget.token_limit,
                        used=self._tokens,
                    )
            if budget.cost_limit is not None:
                limit = self._amount(budget.cost_limit)
                self._check_counted("cost limit", limit, "cost")
                if self._without_price is not None:
                    number, entry = self._without_price
                    where = f" (call {number:,} in the ledger {self.path})"
                    raise self._no_price(entry.model, where)
                if cost is None:
                    raise self._no_size("cost limit", limit, "cost")
                if self._cost + self._reserved_cost + cost > budget.cost_limit:
                    raise BudgetError(
                        f"This batch could pass the cost limit of {limit}, so reliacode did not "
                        f"send it. Its worst case is {self._amount(cost)}; the calls in the "
                        f"ledger {self.path} have cost {self._amount(self._cost)}, and batches "
                        f"already sent may cost {self._amount(self._reserved_cost)} more. To "
                        "send it, raise cost_limit in the Budget.",
                        limit="cost",
                        allowed=budget.cost_limit,
                        used=self._cost,
                    )
        self._reserved_calls += calls
        self._reserved_tokens += tokens or 0
        self._reserved_cost += cost or 0.0

    def _release(self, calls: int, tokens: int | None, cost: float | None) -> None:
        """Give back what :meth:`_reserve` kept for requests whose results are now recorded."""
        self._reserved_calls = max(self._reserved_calls - calls, 0)
        self._reserved_tokens = max(self._reserved_tokens - (tokens or 0), 0)
        self._reserved_cost = max(self._reserved_cost - (cost or 0.0), 0.0)

    def _no_size(self, name: str, limit: str, kind: Literal["cost", "tokens"]) -> BudgetError:
        what = "the price of a cache write or of the model" if kind == "cost" else "its size"
        return BudgetError(
            f"The {name} of {limit} cannot be kept for a batch: the worst case of a request is "
            f"not known, because no call in the ledger {self.path} has token counts yet, or "
            f"{what} is missing from the price table. Code a few texts with code() first, so "
            "the ledger knows how large a call is, add the missing price, or use a call limit "
            "instead, as in Budget(call_limit=1000).",
            limit=kind,
            allowed=self._limit(kind),
            used=self._tokens if kind == "tokens" else self._cost,
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
        self._largest_prompt = max(self._largest_prompt or 0, entry.prompt_tokens)
        self._largest_completion = max(self._largest_completion, entry.completion_tokens)
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


#: The name under which calls with no known model are summed.
UNKNOWN_MODEL = "(unknown)"


class Totals(_Record):
    """What a group of calls used: all of them, or those to one model or for one variable.

    Token counts and costs add up only the calls that have them, and say how many do not:
    a missing count is never taken as zero. A per-text average is given only when every call
    in the group could be counted (or, for the cost, priced); otherwise it is ``None``.

    Attributes:
        calls: the calls made.
        answered, asked_again, missing, failed: the calls by outcome (see :class:`LedgerEntry`).
        attempts: the tries sent to the servers by the calls that got an answer, retries
            included. A failed call's tries are not known.
        texts: the different texts the calls were for (told apart by their hashes).
        input_tokens, output_tokens: the token counts of the calls whose server reported
            both, or ``None`` when no call could be counted.
        calls_without_tokens: the calls whose token counts are not known, failed calls
            included.
        cost: the cost of the calls that have one, in the price table's currency, or ``None``.
        calls_without_cost: the calls with no cost, because a count or a price is missing.
        latency_seconds: the time spent waiting for answers, added up.
        calls_per_text, input_tokens_per_text, output_tokens_per_text, cost_per_text: averages
            over the texts.
        cache_write_tokens, cache_read_tokens: the input tokens written to and read from a
            prompt cache, added up over the calls whose server reported them, or ``None`` when
            none did. They are part of ``input_tokens``.
    """

    calls: int
    answered: int
    asked_again: int
    missing: int
    failed: int
    attempts: int
    texts: int
    input_tokens: int | None
    output_tokens: int | None
    calls_without_tokens: int
    cost: float | None
    calls_without_cost: int
    latency_seconds: float | None
    calls_per_text: float | None
    input_tokens_per_text: float | None
    output_tokens_per_text: float | None
    cost_per_text: float | None
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None


class LedgerSummary(_Record):
    """The totals of a ledger, as the source for the cost and token use reported in a paper.

    Attributes:
        total: all calls.
        by_model: the calls to each model, under the name that was asked for.
        by_variable: the calls for each variable.
        first_at, last_at: the times of the first and the last call, in UTC.
        codebook_hashes, prompt_hashes: the codebook and prompt hashes seen, in the order they
            first appear.
        served_as: for each model asked for, the names the server returned.
        prices: the price table the costs come from, with its currency and date.
    """

    total: Totals
    by_model: dict[str, Totals]
    by_variable: dict[str, Totals]
    first_at: datetime | None
    last_at: datetime | None
    codebook_hashes: tuple[str, ...]
    prompt_hashes: tuple[str, ...]
    served_as: dict[str, tuple[str, ...]]
    prices: PriceTable | None

    def rows(self) -> list[dict[str, object]]:
        """One row per group: ``"all"``, then each model, then each variable.

        Each row has the group, its name, every total and the currency.
        ``pandas.DataFrame(summary.rows())`` turns it into a data frame.
        """
        currency = self.prices.currency if self.prices is not None else None
        groups = [("all", "all calls", self.total)]
        groups += [("model", name, totals) for name, totals in self.by_model.items()]
        groups += [("variable", name, totals) for name, totals in self.by_variable.items()]
        return [
            {"group": group, "name": name, **totals.model_dump(), "currency": currency}
            for group, name, totals in groups
        ]

    def to_csv(self, path: str | os.PathLike[str]) -> None:
        """Write :meth:`rows` to a CSV file. A value that is not known is left empty."""
        fields = ["group", "name", *Totals.model_fields, "currency"]
        with Path(path).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows())

    def to_json(self, path: str | os.PathLike[str]) -> None:
        """Write the whole summary to a JSON file."""
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def summarize(entries: Iterable[LedgerEntry], prices: PriceTable | None = None) -> LedgerSummary:
    """Add up the calls in a ledger: in all, by model and by variable.

    Args:
        entries: the ledger's lines, such as ``read_ledger("ledger.jsonl")``.
        prices: the price table to cost the calls with. Without one, there are tokens and no
            costs. Costs are worked out from the lines and the table each time, so a summary
            can be made again with corrected prices.
    """
    if prices is not None and not isinstance(prices, PriceTable):
        raise TypeError("prices should be a PriceTable(currency=..., read_on=..., prices=...).")
    lines = list(entries)
    if not all(isinstance(entry, LedgerEntry) for entry in lines):
        raise TypeError("summarize() takes ledger lines, such as read_ledger(path).")
    by_model: dict[str, list[LedgerEntry]] = {}
    by_variable: dict[str, list[LedgerEntry]] = {}
    served_as: dict[str, dict[str, None]] = {}
    for entry in lines:
        model = entry.model or UNKNOWN_MODEL
        by_model.setdefault(model, []).append(entry)
        by_variable.setdefault(entry.variable, []).append(entry)
        if entry.model_returned is not None:
            served_as.setdefault(model, {})[entry.model_returned] = None
    times = [entry.at for entry in lines]
    return LedgerSummary(
        total=_totals(lines, prices),
        by_model={name: _totals(group, prices) for name, group in by_model.items()},
        by_variable={name: _totals(group, prices) for name, group in by_variable.items()},
        first_at=min(times, default=None),
        last_at=max(times, default=None),
        codebook_hashes=tuple(dict.fromkeys(entry.codebook_hash for entry in lines)),
        prompt_hashes=tuple(dict.fromkeys(entry.prompt_hash for entry in lines)),
        served_as={name: tuple(names) for name, names in served_as.items()},
        prices=prices,
    )


def _totals(entries: list[LedgerEntry], prices: PriceTable | None) -> Totals:
    counted = [
        e for e in entries if e.prompt_tokens is not None and e.completion_tokens is not None
    ]
    costs = [prices.cost(entry) for entry in entries] if prices is not None else []
    known_costs = [cost for cost in costs if cost is not None]
    latencies = [e.latency_seconds for e in entries if e.latency_seconds is not None]
    outcomes = [entry.outcome for entry in entries]
    texts = len({entry.text_hash for entry in entries})
    input_tokens = sum(e.prompt_tokens or 0 for e in counted) if counted else None
    output_tokens = sum(e.completion_tokens or 0 for e in counted) if counted else None
    cost = sum(known_costs) if known_costs else None
    all_counted = len(counted) == len(entries)
    all_costed = len(known_costs) == len(entries)
    writes = [e.cache_write_tokens for e in counted if e.cache_write_tokens is not None]
    reads = [e.cache_read_tokens for e in counted if e.cache_read_tokens is not None]
    return Totals(
        calls=len(entries),
        answered=outcomes.count("answered"),
        asked_again=outcomes.count("asked_again"),
        missing=outcomes.count("missing"),
        failed=outcomes.count("failed"),
        attempts=sum(entry.attempts or 0 for entry in entries),
        texts=texts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls_without_tokens=len(entries) - len(counted),
        cost=cost,
        calls_without_cost=len(entries) - len(known_costs),
        latency_seconds=sum(latencies) if latencies else None,
        calls_per_text=_per(len(entries), texts),
        input_tokens_per_text=_per(input_tokens, texts) if all_counted else None,
        output_tokens_per_text=_per(output_tokens, texts) if all_counted else None,
        cost_per_text=_per(cost, texts) if all_costed else None,
        cache_write_tokens=sum(writes) if writes else None,
        cache_read_tokens=sum(reads) if reads else None,
    )


def _per(value: float | None, texts: int) -> float | None:
    return value / texts if value is not None and texts else None

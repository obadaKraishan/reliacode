"""The ledger: one line for every call to a model, with what it used and what it cost.

Each line records one call. It says when the call was made, which model was asked and which one
answered, and the settings used. It gives the token counts the server reported, the latency,
the tries and the outcome. It also records what ties the call to the work: the codebook's
content hash, the prompt hash, the variable, and a hash of the text with its length. A line
never holds the text that was coded, the model's answer or an API key, so a ledger can be
shared even when the texts cannot.
"""

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
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

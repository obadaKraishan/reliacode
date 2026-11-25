# The ledger: tokens, cost and time for every call

A ledger records every call that `reliacode.code()` makes to a model: which model answered, how
many tokens the call used, how long it took, what came of it, and which codebook, prompt and
variable it was for. With the prices you supply, it adds up what a run cost. It can also stop a
run before it passes a budget. A ledger holds no coded text, so you can keep it with your data
and share it when the texts themselves cannot be shared.

## An example

The model names and prices on this page are invented. Read the prices of the model you use from
its provider on the day you run, and write them down with that date.

```python
from reliacode import (
    Budget,
    Ledger,
    ModelPrice,
    OpenAICompatibleBackend,
    PriceTable,
    code,
    load_codebook,
)

codebook = load_codebook("examples/codebooks/news_frames.yaml")
backend = OpenAICompatibleBackend(
    "https://api.example.org/v1", "example-hosted-model", api_key_env="EXAMPLE_API_KEY"
)
prices = PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={
        "example-hosted-model": ModelPrice(input_per_million=0.50, output_per_million=1.50),
    },
)
ledger = Ledger("ledger.jsonl", prices=prices, budget=Budget(cost_limit=5.00))
run = code(codebook, texts, backend=backend, ledger=ledger)

summary = ledger.summary()
summary.total.calls, summary.total.input_tokens, summary.total.output_tokens, summary.total.cost
summary.to_csv("ledger-summary.csv")
```

## What the file holds

The ledger is a [JSON Lines](https://jsonlines.org) file with one line for each call. Each line
is written and flushed to disk as its call ends, so a run that stops loses no line. reliacode
only adds lines at the end and never changes the ones already there. Here is one invented line,
spread over several lines to be readable; in the file it is a single line:

```json
{
  "schema_version": 3,
  "at": "2026-11-19T15:04:05Z",
  "outcome": "answered",
  "variable": "dominant_frame",
  "ask": 1,
  "model_requested": "example-hosted-model",
  "model_returned": "example-hosted-model-2026-10",
  "parameters": {
    "endpoint": "https://api.example.org/v1/chat/completions",
    "model": "example-hosted-model",
    "temperature": 0.0,
    "seed": null,
    "max_tokens": null,
    "structured_output": true,
    "extra_body": {},
    "api_version": null,
    "thinking": null,
    "effort": null,
    "cache": null
  },
  "prompt_tokens": 612,
  "completion_tokens": 7,
  "token_source": "server",
  "latency_seconds": 0.84,
  "finish_reason": "stop",
  "response_id": "chatcmpl-8f2c",
  "request_id": "req-41d7",
  "attempts": 1,
  "error_reason": null,
  "http_status": null,
  "codebook_hash": "sha256:b4bdbacd8a7fae5da8725e4bd5c2f5f5a44187922e5ec380a2b21f2ee491bdb1",
  "prompt_hash": "sha256:496fc7e5326a23fa332fe78c97a30e6db3ac9c2ce5fd837d90e804f16c76ff30",
  "text_hash": "sha256:bae6f67f699680cfbec6973dd1b2c6ebb2e84483c3d37d000f08c8aaee76c44a",
  "text_length": 53,
  "reasoning_tokens": null,
  "cache_write_tokens": null,
  "cache_write_1h_tokens": null,
  "cache_read_tokens": null,
  "batch_id": null,
  "custom_id": null
}
```

- `at`: when the call ended, in UTC.
- `outcome`: what came of the call:
  - `answered`: the answer could be used.
  - `asked_again`: it could not, and the model was asked again.
  - `missing`: it could not, and no try was left.
  - `failed`: no answer came.
- `schema_version`: the version of the line's format, now 3.
  - Version 2 added `reasoning_tokens` and `parameters.extra_body`.
  - Version 3 added the cache counts, the batch ids, and four request settings
    (`parameters.api_version`, `thinking`, `effort` and `cache`). All of them are new fields at
    the end.
  - Lines written with versions 1 and 2 are still read, with the fields they lack left `null`,
    and a file can hold lines of all three versions.
- `ask`: 1 for the first question about a text and variable, 2 for the second.
- `model_requested` and `model_returned`: the model asked for, and the name the server gave in
  its answer, which can be more exact.
- `parameters`: the settings of the request, as in `reliacode.RequestParameters`.
  - Its `temperature` is `null` when no temperature was sent.
  - Its `extra_body` holds the extra request fields, as given (see
    [Request fields](backends.md#request-fields-temperature-reasoning-and-other-settings)).
  - Its `api_version`, `thinking`, `effort` and `cache` are set by `AnthropicBackend`: the
    `anthropic-version` header, the thinking setting and the effort sent (`null` when none was),
    and the cache's time to live (`"5m"` or `"1h"`) when the shared prompt was marked for
    caching (see [Anthropic's own API](backends.md#anthropics-own-api-prompt-caching-and-message-batches)).
- `prompt_tokens` and `completion_tokens`: the input and output tokens, as the server reported
  them in its answer's usage field. `token_source` is `"server"` when they are there.
  `prompt_tokens` is always the whole input, the tokens read from or written to a cache included.
- `cache_read_tokens` and `cache_write_tokens`: the input tokens read from and written to a
  prompt cache, as the server reported them. They are part of `prompt_tokens`. A server that
  does not report them leaves `null`. `AnthropicBackend` reports both (from
  `cache_read_input_tokens` and `cache_creation_input_tokens`), and `OpenAICompatibleBackend`
  reports the reads when the server gives `prompt_tokens_details.cached_tokens`.
- `cache_write_1h_tokens`: how many of the written tokens went to a one-hour cache, the rest
  going to a five-minute cache, when the server says (Anthropic's
  `cache_creation.ephemeral_1h_input_tokens`).
- `batch_id` and `custom_id`: for a request sent in a batch by `code_in_batches()`, the batch
  and the request within it. A request in a batch has no latency of its own, so its
  `latency_seconds` is `null`.
- `reasoning_tokens`: the tokens the model spent reasoning, when the server reports them
  (`usage.completion_tokens_details.reasoning_tokens`). They are part of `completion_tokens`, so
  they are counted once in the cost. A server that does not report them leaves `null`.
- `latency_seconds`, `finish_reason`, `response_id` and `request_id`: as the server gave them.
- `attempts`: the tries sent to the server for this call, retries after passing problems
  included.
- `error_reason` and `http_status`: for a failed call, the kind of problem (such as
  `"timeout"`, `"rate_limited"`, or for a request in a batch `"batch_expired"`,
  `"batch_errored"` and `"batch_canceled"`) and the server's status code.
- `codebook_hash`, `prompt_hash` and `variable`: what the call was for. With one call per unit
  (`recipe="per_unit"`), one call asks for several variables. `variable` then names every one,
  joined by `+`, such as `incivility_form+severity`. `outcome` is then `missing` when any of
  them is still missing after the last answer.
- `text_hash` and `text_length`: the SHA-256 hash of the coded text and its length in
  characters, which tie the call to its text.

`reliacode.read_ledger("ledger.jsonl")` reads the lines back as `LedgerEntry` records. If a
line cannot be read, for example because a program stopped while writing it, you get a
`LedgerError` that lists every such line by its number. Lines are never skipped in silence, and a
ledger does not open on such a file. Fix or remove those lines in a text editor (keep a copy
first). Blank lines are allowed.

Only one program should write to a ledger file at a time. Two runs that write to the same file
at once can mix their lines. Give each run its own file, or run them one after the other.

## What the file never holds

- **the text that was coded.** Only its hash and its length are kept. Anyone who has a text
  can check that a line belongs to it, but the hash does not give the text back. A very short
  or common text can still be guessed and checked, so share the hashes of such texts with the
  same care as the texts.
- **the prompt or the messages sent.** The prompt hash identifies them.
- **the model's answer.** The codes are in the results of `code()`, not in the ledger.
- **the API key.** An extra request field whose name looks like a credential is refused before
  any request, so it never reaches a line. The extra fields you give are kept as given, so never
  put a text or a key in them.
- **a server's error message.** It can quote the request, so only the kind of problem is kept.

## Token counts that are missing

Token counts come only from the server's answer. When a server reports none, the line has
`null` counts and no cost. reliacode never makes a count up and never takes a missing count as
zero. Every total in the summary says how many calls it could not count, and a failed call
always counts as one of them, since its use is not known.

## Prices

reliacode ships no prices. Prices change often, and a wrong price built into a package would be
worse than none. You write down the prices you read in a `PriceTable`: one currency, the date
you read them, and for each model the price of a million input tokens and of a million output
tokens. A call costs its input tokens at the input price plus its output tokens at the output
price. At the invented prices above, the line shown costs
(612 × 0.50 + 7 × 1.50) / 1,000,000 = 0.0003165 USD.

- A model's price is looked up under the name asked for, then under the name the server
  returned.
- A model that is not in the table gets its tokens counted and no cost.
- A local model costs nothing to run, so list it at 0. Its calls then cost 0, and their tokens
  are still counted:

  ```python
  ModelPrice(input_per_million=0, output_per_million=0)
  ```
- A price table can be kept in a JSON file and read with
  `PriceTable.model_validate_json(Path("prices.json").read_text())`.

A service with a prompt cache or a batch interface has more prices, and a `ModelPrice` can hold
them, so that costs stay exact:

- `cache_write_5m_per_million` and `cache_write_1h_per_million`: the price of tokens written to
  a cache that lasts five minutes or one hour. A call that wrote to a cache whose price is not
  in the table has no cost, since a write costs more than plain input.
- `cache_read_per_million`: the price of tokens read from a cache. Without it, those tokens are
  priced as plain input, as in earlier versions.
- `batch_discount`: the share taken off every price for a request sent in a batch, such as
  `0.5` for half price. Without it, a request in a batch costs what it would cost alone.
- `long_prompt_above` and `long_prompt_price`: for a model priced by prompt length, the prompt
  length (cache reads and writes included) above which a call pays the second set of prices,
  for all of its tokens.

A call's input is then priced in three parts: the plain input
(`prompt_tokens - cache_write_tokens - cache_read_tokens`) at the input price, the cache writes
at the price for their time to live, and the cache reads at the read price. Its output is priced
at the output price, and a call in a batch gets the batch discount. A cost that cannot be known
is `None`, never a guess, for instance when caching was asked for and the server did not say
what it did. For an example with Claude Haiku 5.5's prices, see
[backends.md](backends.md#an-example-from-start-to-finish-one-call-per-text-cached).

The ledger keeps tokens and not costs, so the costs can always be worked out again from the
file and a price table. If a price was wrong, summarize the file again with the corrected table:
`summarize(read_ledger("ledger.jsonl"), corrected_prices)`.

## Setting a budget

A `Budget` sets a limit on the cost (`cost_limit`, in the price table's currency), on the tokens
(`token_limit`, input and output together), on the number of calls (`call_limit`), or on any of
them together:

```python
ledger = Ledger("ledger.jsonl", prices=prices, budget=Budget(cost_limit=5.00, call_limit=20_000))
```

The ledger checks the budget before every call and refuses the call that would pass a limit.
How large a call will be is not known before it is made, so the ledger takes the next call to
be as large as the largest call so far. A run therefore stops a little before a limit, and it
can pass the limit only if one call is larger than every call before it. The first call is
made unless a limit is already reached. The budget covers every call in the ledger file,
including calls from earlier runs on the same file.

When the budget stops a run, `code()` raises a `BudgetError` that says what the limit is, what
has been used and how to raise it, for example:

> The next call could pass the cost limit of 5.00 USD, so reliacode stopped before it. The calls
> in the ledger ledger.jsonl have cost 4.9981 USD so far, and the largest one cost 0.0021 USD. To
> go on, raise cost_limit in the Budget.

The codes finished before the stop are not lost. They come with the error:

```python
try:
    run = code(codebook, texts, backend=backend, ledger=ledger)
except BudgetError as stop:
    print(stop)
    run = stop.run  # the results finished before the stop
```

A limit that could not be kept is refused instead of being ignored:

- A cost limit needs a price table.
- A cost limit for a model with no price is refused before the first call.
- A token or cost limit stops the run after an answer that came without token counts.

For a server that reports no token counts, use a call limit.

**Batches.** `code_in_batches()` checks the budget once for each round of batches, before it is
sent, against the round's worst case. The worst case takes every request to be as large as the
largest call in the ledger so far, with all of its input written to the cache and none read,
at the batch price. It prints that estimate, and the ledger keeps it back from the budget until
the requests' results are recorded, so a call or another batch cannot spend it twice. With a
cost or token limit, the ledger needs at least one counted call first, so code a few texts with
`code()` before the first batch. A call limit needs no counts.

## The summary

`ledger.summary()`, or `summarize(read_ledger(path), prices)`, adds up the calls in all
(`total`), by model (`by_model`, under the name asked for) and by variable (`by_variable`).
Each group gives:

- the calls, by outcome (answered, asked again, missing and failed)
- the tries sent to the servers
- the number of different texts
- the input and output tokens, and the calls that could not be counted
- the tokens written to and read from a prompt cache (`cache_write_tokens` and
  `cache_read_tokens`, part of the input tokens), when the servers reported them
- the cost, and the calls that could not be costed
- the time spent waiting for answers
- averages per text

An average per text is given only when every call in the group was counted (or, for the cost,
priced). Otherwise it is `None`, because an average over some of the calls would read as one
over all of them.

The summary also gives the times of the first and the last call, the codebook and prompt hashes
seen, the model names the servers returned, and the price table used. `summary.rows()` gives one
row per group (`pandas.DataFrame(summary.rows())` makes a data frame). `summary.to_csv(path)`
and `summary.to_json(path)` write it to a file.

## Testing a budget without a server

`FakeBackend` can report token counts: one pair for every answer, a list of pairs used in
order, or a function of the messages. This runs anywhere, at no cost:

```python
from reliacode import Budget, BudgetError, FakeBackend, Ledger, code, load_codebook

codebook = load_codebook("examples/codebooks/news_frames.yaml")
backend = FakeBackend(lambda messages: '{"code": "economic"}', tokens=(600, 8))
ledger = Ledger("test-ledger.jsonl", budget=Budget(token_limit=1500))
texts = ["Invented text one.", "Invented text two.", "Invented text three."]
try:
    code(codebook, texts, backend=backend, ledger=ledger, variables="dominant_frame")
except BudgetError as stop:
    print(stop)  # the next call could pass the token limit of 1,500 ...
    print(stop.run.table())  # the two texts coded before the stop
```

## Reporting cost in a paper

Report the model as the server named it, the number of calls, the input and output tokens, and
the cost with its currency and the date the prices were read. Also say how many calls could not
be counted. For example (with invented numbers): "Coding the 1,200 posts took 2,431 calls to
example-hosted-model-2026-10 (31 second asks, 2 failed calls), with 1.9 million input and 41,000
output tokens; at the prices read on 19 November 2026, this cost about 1.01 USD."

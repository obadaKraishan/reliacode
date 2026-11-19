# Backends: coding texts with a model

reliacode sends each prompt to a *backend* and checks the answer that comes back. One backend,
`OpenAICompatibleBackend`, works with any server that offers the OpenAI-compatible chat
completions interface: a local server such as Ollama, vLLM, LM Studio or llama.cpp's server, and
hosted services such as OpenRouter. A second one, `FakeBackend`, gives scripted answers for tests.
No vendor library is needed.

**Privacy.** Text sent to a hosted endpoint leaves your computer and is handled under that
service's terms, so read them before you send research data, and never send text you are not
allowed to share. A local server keeps the text on your own machine, which is why the first
example below uses one.

## A local server (Ollama)

Install [Ollama](https://ollama.com), download a model (`ollama pull qwen3:4b`), and make sure
the server is running (open the Ollama app, or run `ollama serve`). Then:

```python
import reliacode

codebook = reliacode.load_codebook("examples/codebooks/incivility.yaml")
backend = reliacode.OpenAICompatibleBackend("http://localhost:11434/v1", "qwen3:4b", seed=1)

texts = {
    "c1": "Thanks for the clear summary of the budget vote.",
    "c2": "Only a fool would believe the council's numbers.",
}
run = reliacode.code(codebook, texts, backend=backend)
run.table()  # one row per text: its id, then one code per variable
```

`texts` can be a list (each text's id is its position, counting from 0) or a mapping from ids to
texts. From a data frame, use `dict(zip(df["id"], df["text"]))`, after removing empty texts.

## A hosted endpoint (OpenRouter)

A hosted service needs an API key. Keep it in an environment variable, never in your code, and
give the backend the variable's *name*. Set the variable in a terminal before you start Python:

```bash
export OPENROUTER_API_KEY=...   # the key from your OpenRouter account
```

```python
backend = reliacode.OpenAICompatibleBackend(
    "https://openrouter.ai/api/v1",
    "meta-llama/llama-3.3-70b-instruct",
    api_key_env="OPENROUTER_API_KEY",
)
run = reliacode.code(codebook, ["The new bus line finally reaches the east side."], backend=backend)
```

The key is read from the variable each time a request is sent and goes only into that request's
Authorization header. It is never stored on the backend, written to a log, shown in an error
message or kept in a result. Requests to a hosted service may cost money; reliacode does not
count tokens or cost yet ([#2](https://github.com/obadaKraishan/reliacode/issues/2)).

## Settings

| Setting | Default | What it does |
|---|---|---|
| `base_url` | (required) | The server's address up to the version, such as `http://localhost:11434/v1` |
| `model` | (required) | The model's name as the server knows it |
| `api_key_env` | `None` | The name of the environment variable that holds the key; `None` for a local server |
| `timeout` | `60` | Seconds to wait for each answer |
| `max_retries` | `3` | How many times to try again after a passing problem (see below) |
| `temperature` | `0` | Sampling temperature; 0 makes answers as repeatable as the server allows |
| `seed` | `None` | A random seed, for servers that support one |
| `max_tokens` | `None` | A limit on each answer's length; the server's default applies when it is `None` |
| `structured_output` | `True` | Send the JSON Schema of the answer to the server; turn it off if the server refuses it |

## How texts are coded

Each text is coded for one variable at a time. The prompt for a variable is built from the
codebook alone: its title, description, unit of analysis, language and instructions, then the
variable's definition, instructions and decision rules, and every category with its code, label,
level, definition and examples. To read the exact prompt:

```python
from reliacode.coding import build_prompt

print(build_prompt(codebook, "severity").system)
```

The model is asked to answer with a JSON object such as `{"code": "mild"}`. An answer counts only
if its code is exactly one of the variable's codes: upper and lower case must match, and a near
miss is never corrected. When an answer cannot be used, the model is shown its answer and the
reason and asked again (`attempts=2` allows one more try). If no answer can be used, the code is
recorded as missing (`None`), with the reason and every raw answer kept. `run.missing()` lists
those results so you can check them by hand.

Because each variable is coded on its own, a decision rule that refers to another variable (such
as "code none here if and only if incivility_form is none") is in the prompt, but the model does
not see the other variable's answer.

## What is recorded

Every result (`run.results`) holds the text's id, the variable, the code, the raw answers, the
codebook's content hash and the prompt hash, and one `Response` for each answer. A `Response`
records the answer's text, the model name the server returned, the request's parameters (the
endpoint, the model asked for, the temperature, the seed, the token limit and whether structured
output was used), the token counts, the finish reason, the server's ids and fingerprint, the
latency, the number of tries and the time in UTC. A token count the server does not report is
`None`, never zero.

The run (`CodingRun`) adds the codebook's name, version and content hash, the prompts, the start
and end times and the reliacode version. Save it as JSON to keep the full record next to your
data:

```python
from pathlib import Path

Path("coding-run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
```

Report the codebook's content hash and the prompt hash with your results, so readers can check
exactly which codebook and prompt were used. The prompt hash covers the prompt, the message
templates, the JSON Schema of the answer and the version of the prompt recipe.

## Retries

"Too many requests" (status 429), server errors (status 500 and up), timeouts and broken
connections are passing problems, so the backend tries again, up to `max_retries` times. It
waits 1 second, then 2, then 4, and so on up to 30 seconds, or as long as the server asks
(`Retry-After`). If the server asks for more than a minute, the backend stops at once and says
so. Other refusals are not tried again, because a second try would get the same answer.

## When something goes wrong

A `BackendError` says what went wrong, where, and what to try. Its `reason` names the kind of
problem, for code that handles some problems differently, and `status` holds the server's HTTP
status code when there was one. A `BackendError` stops `code()`, because the requests that
follow would fail the same way.

| `reason` | What happened | What to try |
|---|---|---|
| `no_server` | Nothing answers at the address | Start the server (for Ollama, `ollama serve`), or check `base_url` |
| `model_not_found` | The server has no model with that name | `ollama list` and `ollama pull`, or the service's list of models |
| `no_endpoint` | The address has no chat completions endpoint | `base_url` usually ends in `/v1` |
| `missing_key`, `bad_key` | The key variable is not set, is empty, or holds characters a key cannot have | Set the variable before starting Python |
| `key_refused` | The server refused the key (status 401 or 403) | Check the key, or pass `api_key_env` |
| `bad_request` | The server refused the request | Read the server's message; for `response_format`, use `structured_output=False` |
| `redirected` | The server sent the request to another address | Use that address as `base_url` |
| `rate_limited` | "Too many requests" held after the retries | Wait, send fewer requests, or raise `max_retries` |
| `server_error` | The server failed after the retries | Try again later |
| `timeout` | No answer in time after the retries | Raise `timeout`, lower `max_tokens`, or use a smaller model |
| `connection_lost` | The connection broke off after the retries | Check the server and the network |
| `unexpected_answer` | The answer is not a chat completion | Check that `base_url` points to an OpenAI-compatible server |

## Testing your own code

`FakeBackend` answers from a script and never opens a connection, so tests of code that uses
reliacode run anywhere, at no cost, with the same answers every time:

```python
from reliacode import FakeBackend, code, load_codebook

codebook = load_codebook("examples/codebooks/news_frames.yaml")
backend = FakeBackend(['{"code": "economic"}', '{"code": "government"}'])
run = code(codebook, ["The council raised the bus fare to pay for new buses."], backend=backend)
run.table()  # [{'unit': 0, 'dominant_frame': 'economic', 'responsibility_target': 'government'}]
```

Answers are used in order. A function of the messages can make the answers instead, and an
exception in the list (such as a `BackendError`) is raised in place of an answer, to test how
your code handles problems. `backend.calls` keeps every request's messages. With
`tokens=(120, 8)`, the fake also reports token counts, to test a budget (see
[ledger.md](ledger.md)).

Any object with a `complete(messages, *, json_schema=None)` method that returns a
`reliacode.Response` can be used as a backend.

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
message or kept in a result. Requests to a hosted service may cost money: a ledger records the
tokens and the cost of every call, and a budget stops a run before it passes a limit (see
[ledger.md](ledger.md)).

## Settings

| Setting | Default | What it does |
|---|---|---|
| `base_url` | (required) | The server's address up to the version, such as `http://localhost:11434/v1` |
| `model` | (required) | The model's name as the server knows it |
| `api_key_env` | `None` | The name of the environment variable that holds the key; `None` for a local server |
| `timeout` | `60` | Seconds to wait for each answer |
| `max_retries` | `3` | How many times to try again after a passing problem (see below) |
| `temperature` | `0` | Sampling temperature; 0 makes answers as repeatable as the server allows, and `None` leaves it out of the request |
| `seed` | `None` | A random seed, for servers that support one |
| `max_tokens` | `None` | A limit on each answer's length; the server's default applies when it is `None` |
| `structured_output` | `True` | Send the JSON Schema of the answer to the server; turn it off if the server refuses it |
| `extra_body` | `None` | Extra request fields sent as given, such as `{"reasoning_effort": "low"}` (see [Request fields](#request-fields-temperature-reasoning-and-other-settings)) |

## Request fields: temperature, reasoning and other settings

Servers differ in the request fields they take. Three things help:

- **`temperature=None`** leaves the temperature out of the request. Some models refuse the
  field. The model then samples by its own default, and the record's `parameters.temperature` is
  `None`, so a report can say that no temperature was sent. The default stays 0. When a server
  answers that a temperature is not supported, the error suggests `temperature=None`.
- **`extra_body`** holds extra request fields, sent as given with every request. Examples:
  - `{"reasoning_effort": "low"}` for a reasoning model;
  - `{"chat_template_kwargs": {"enable_thinking": False}}` for a vLLM model whose chat template
    can turn thinking off;
  - Anthropic's `{"thinking": {...}}`.

  The values must be JSON values. The fields are recorded with every answer
  (`parameters.extra_body`) and in every ledger line, so a methods section can state them.
- **Refused fields.** Two kinds of field are refused when the backend is made, before any
  request is sent:
  - **A field the backend sets itself:** `model`, `messages`, `stream`, `temperature`, `seed`,
    `max_tokens` and `response_format`. Use the backend's own settings for these.
  - **A field whose name looks like a credential,** at any depth: its name contains key, token,
    secret, password, auth, credential, bearer or cookie, in any case. A count such as
    `max_completion_tokens` or `budget_tokens` is not one. A name such as `author` is refused
    too, since it holds "auth": rename such a field. The error names the field but never shows
    its value. A key belongs in an environment variable, named with `api_key_env`.

  Never put a coded text or a key in `extra_body`: the fields are kept in every record.
- **Reasoning tokens** are the tokens a model spends reasoning (or thinking) before it answers.
  When the server reports them (`usage.completion_tokens_details.reasoning_tokens`), the answer's
  `reasoning_tokens` and the ledger line's `reasoning_tokens` hold the count. In that format they
  are part of the output tokens, so costs count them once. A server that does not report them
  leaves the count `None`, never 0.

### An example from start to finish

This example uses a vLLM server on your own machine serving `openai/gpt-oss-120b`. It leaves the
temperature to the model's own defaults and sets the reasoning effort:

```python
import reliacode

codebook = reliacode.load_codebook("examples/codebooks/news_frames.yaml")
backend = reliacode.OpenAICompatibleBackend(
    "http://localhost:8000/v1",
    "openai/gpt-oss-120b",
    temperature=None,
    extra_body={"reasoning_effort": "low"},
)
texts = {"a1": "The council raised the bus fare to pay for new buses."}
run = reliacode.code(codebook, texts, backend=backend)

result = run.results[0]
print(result.variable, result.code)
print(result.parameters.temperature, result.parameters.extra_body)
print(result.responses[-1].completion_tokens, result.responses[-1].reasoning_tokens)
```

```text
dominant_frame economic
None {'reasoning_effort': 'low'}
104 96
```

The request had no `temperature` field and had `"reasoning_effort": "low"`. The record says so,
and the server reported 96 reasoning tokens among the 104 output tokens. (The answer and the
counts here are invented.)

### What two servers accept, as their documentation says

These are quoted from each server's documentation as read on 2026-11-24. Nothing was tried: no
request was sent to either server. Both pages change, so read them again before a run.

**Anthropic's OpenAI SDK compatibility endpoint** (base URL `https://api.anthropic.com/v1`). The
quotes are from [OpenAI SDK compatibility](https://platform.claude.com/docs/en/api/openai-sdk),
which shows no date.

- **`temperature`:** "Between 0 and 1 (inclusive). Values greater than 1 are capped at 1. On
  Claude 4.7 and later models and Claude Mythos Preview, any value below 1 returns a 400 error,
  so omit it." With reliacode, that means `temperature=None`.
- **Thinking:** "On Claude 5 models and Claude Mythos Preview, thinking is on by default, so you
  can omit the `thinking` field. To use thinking on other models through the compatibility
  layer, set the `thinking` field of the request body to `{"type": "enabled", "budget_tokens":
  N}`. `N` must be at least 1,024 and less than `max_tokens` or `max_completion_tokens`,
  whichever you set."
  - The page also says: "On models that reject `enabled` in the native Claude API, the
    compatibility layer converts `enabled` to adaptive thinking. On those models,
    `budget_tokens` is required but has no effect."
  - How to turn thinking off is not stated.
- **`reasoning_effort` and `seed`:** "Ignored".
- **`response_format`:** "Supported with limits. Only `json_schema` with `strict: true` is
  accepted, and the structured outputs JSON Schema limitations apply. Any other value, including
  `json_object` and `text`, returns a 400 error." reliacode sends `json_schema` with
  `strict: true`.
- **Reasoning tokens:** `usage.completion_tokens_details` is "Always empty", so `reasoning_tokens`
  stays `None`.
- **Other fields:** "Most unsupported fields are silently ignored rather than producing errors."

So a backend for a Claude 5 model leaves the temperature out. Thinking can be set explicitly,
with a token limit above its budget:

```python
backend = reliacode.OpenAICompatibleBackend(
    "https://api.anthropic.com/v1",
    "claude-haiku-5-5",
    api_key_env="ANTHROPIC_API_KEY",
    temperature=None,
    max_tokens=4000,
    extra_body={"thinking": {"type": "enabled", "budget_tokens": 2000}},
)
```

**vLLM's OpenAI-compatible server** (for `openai/gpt-oss-120b`). The quotes are from vLLM's
"latest" documentation; each page's footer date is given.

- **Extra fields:** "vLLM supports a set of parameters that are not part of the OpenAI API."
  They can be sent with the OpenAI client's `extra_body`, "Or directly merge them into the JSON
  payload if you are using HTTP call directly", which is what reliacode's `extra_body` does.
  `chat_template_kwargs` is described as "Additional keyword args to pass to the template
  renderer." ([OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/),
  November 24, 2026.)
- **Temperature:** "By default, the server applies `generation_config.json` from the Hugging
  Face model repository if it exists." The page continues: "This means the default values of
  certain sampling parameters can be overridden by those recommended by the model creator." So
  with `temperature=None`, the model's recommended setting applies. (Same page.)
- **Thinking and reasoning effort:**
  - For models whose chat template takes `enable_thinking`: "To disable it, you must pass
    `enable_thinking=False` in your `chat_template_kwargs`." (This is said of the Qwen3 series.)
  - The same page maps "`reasoning_effort` = `"low"`, `"medium"`, or `"high"` →
    `enable_thinking = true`" and "`reasoning_effort` = `"none"` → `enable_thinking = false`".
    ([Reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html),
    November 5, 2026.)
  - Neither page, nor vLLM's gpt-oss-120b recipe
    ([recipes.vllm.ai/openai/gpt-oss-120b](https://recipes.vllm.ai/openai/gpt-oss-120b), no
    date, "Requires vLLM >= 0.10.1."), says how gpt-oss applies `reasoning_effort`. Check the
    reasoning tokens and the answers in a pilot.
- **Structured output:** the structured outputs page uses `response_format` with
  `"type": "json_schema"` on chat completions, and says: "Note that you can use reasoning with
  any provided structured outputs feature."
  ([Structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html),
  November 24, 2026.)
- **Reasoning tokens:** "With `--reasoning-parser`, `/v1/chat/completions/batch` sums
  `reasoning_tokens` across conversations in `usage.completion_tokens_details`". The page says
  this is as `/v1/chat/completions` does for one conversation, so a server started with a
  reasoning parser reports them. (OpenAI-compatible server page.)
- **The gpt-oss recipe's known limits** include: "Responses API: streaming is basic,
  annotations/citations unsupported, usage accounting returns zeros." reliacode uses chat
  completions, not the Responses API.

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
not see the other variable's answer. To ask for all the variables together, and to check such
rules, use [one call per unit](#one-call-per-unit-every-variable-together-with-rules).

## One call per unit: every variable together, with rules

With `recipe="per_unit"`, one call per text asks for all the variables, or the ones named in
`variables`. The prompt holds the same parts of the codebook as the default prompts, with every
variable in turn. The answer is one JSON object with one field per variable, such as
`{"incivility_form": "none", "severity": "none"}`. The JSON Schema of that object is sent when
`structured_output` is on, and each answer's `parameters.structured_output` says whether it was.

Rules across variables are given with the call, as `Rule`s. Each has three parts:
- a sentence, stated in the prompt under "Rules across variables";
- the variables the rule names;
- a check, which gets the codes of those variables, as a dict, and returns `True` when they
  follow the rule.

The codebook format does not change.

Each answer is checked in this order:

1. **Each code** must be one of its own variable's codes, with the same leniency as one call per
   variable.
2. **Each rule** is checked once every variable it names has a valid code.
3. **A broken answer is asked once more** (with `attempts=2`). An answer is broken when it is
   not JSON, has a variable missing, has a code that is not one of its variable's codes, or breaks
   a rule. The second ask names each problem in plain words and never quotes the text.
4. **The last answer decides every code.** So the rules are always checked on codes given
   together:
   - a variable it gives no usable code for is missing, with the reason;
   - a broken rule makes the variables it names missing;
   - the other variables keep their codes.

Here a `FakeBackend` stands in for the model, so the example runs anywhere. Its answers are
invented, and so are the comments:

```python
from reliacode import FakeBackend, Rule, code, load_codebook

codebook = load_codebook("examples/codebooks/incivility.yaml")
severity_rule = Rule(
    text="Severity is none if and only if incivility_form is none.",
    variables=["incivility_form", "severity"],
    check=lambda codes: (codes["severity"] == "none") == (codes["incivility_form"] == "none"),
)
comments = {
    "c1": "Thanks for the clear summary of the budget vote.",
    "c2": "Only a clown would vote for this mayor.",
    "c3": "This is the dumbest budget proposal I have ever read.",
}
backend = FakeBackend(
    [
        '{"incivility_form": "none", "severity": "none"}',
        '{"incivility_form": "name_calling", "severity": "none"}',  # breaks the rule
        '{"incivility_form": "name_calling", "severity": "strong"}',
        '{"incivility_form": "aspersion", "severity": "Mild"}',  # not one of the codes
        '{"incivility_form": "aspersion", "severity": "harsh"}',  # still not
    ]
)
run = code(codebook, comments, backend=backend, recipe="per_unit", rules=[severity_rule])

for row in run.table():
    print(row)
for result in run.missing():
    print(result.unit, result.variable, result.missing_reason)
```

```text
{'unit': 'c1', 'incivility_form': 'none', 'severity': 'none'}
{'unit': 'c2', 'incivility_form': 'name_calling', 'severity': 'strong'}
{'unit': 'c3', 'incivility_form': 'aspersion', 'severity': None}
c3 severity The last of the 2 answers could not be used for this variable. The code for "severity" is not one of its codes: none, mild, strong.
```

The first answer for `c2` broke the rule, so the model was asked again. This is what it was
sent:

```python
print(backend.calls[2][-1]["content"])
```

```text
That answer cannot be used as it is:
- The codes break this rule: Severity is none if and only if incivility_form is none.
Answer again with only a JSON object with one field for each variable (incivility_form, severity), each holding one of that variable's codes. Write nothing else.
```

What stays the same, and what is new:

- **The results** are still one per text and variable. So `run.table()`, `read_codes(...,
  run=run)`, the reliability functions, the ledger and the budget work as they do with one call
  per variable.
- **The ledger** keeps one line per call. Its `variable` names every variable asked, joined by
  `+` (here `incivility_form+severity`). Its `outcome` is `missing` when any variable is still
  missing after the last answer.
- **The prompt.** To read the exact prompt:

  ```python
  from reliacode.coding import build_unit_prompt

  print(build_unit_prompt(codebook, rules=[severity_rule]).system)
  ```

- **The recipe has its own version** (`UNIT_PROMPT_VERSION`) **and its own prompt hash**,
  recorded with every result as `prompt_hash`.
  - The hash covers the prompt, the templates, the JSON Schema, and each rule's sentence and
    variables.
  - It does not cover the code of a rule's check, so keep the checks with your analysis script.
  - `run.recipe` is `"per_unit"`, and `run.prompts` holds the one `UnitPrompt`.
- **The default recipe** stays one call per variable, and its prompts and hashes do not change.
  `rules` is refused with it, because the model never sees its other answers.

Which recipe to use is a choice for your study. One call per text sends the codebook once per
text instead of once per variable, and lets the model see its own codes for the other variables.
It can also give codes that lean on each other. A pilot that codes the same texts both ways
shows how far the codes differ.

## What is recorded

Every result (`run.results`) holds the text's id, the variable, the code, the raw answers, the
codebook's content hash and the prompt hash, and one `Response` for each answer. A `Response`
records the answer's text, the model name the server returned, the request's parameters (the
endpoint, the model asked for, the temperature (`None` when none was sent), the seed, the token
limit, whether structured output was used, and the extra request fields), the token counts
(with the reasoning tokens when the server reports them), the finish reason, the server's ids and
fingerprint, the latency, the number of tries and the time in UTC. A token count the server does
not report is `None`, never zero.

The run (`CodingRun`) adds the codebook's name, version and content hash, the prompts, the
recipe (`"per_variable"` or `"per_unit"`), the start and end times and the reliacode version. Save it as JSON to keep the full record next to your
data:

```python
from pathlib import Path

Path("coding-run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
```

Report the codebook's content hash and the prompt hash with your results, so readers can check
exactly which codebook and prompt were used. The prompt hash covers the prompt, the message
templates, the JSON Schema of the answer and the version of the prompt recipe. With one call per
unit it also covers the rules' sentences and the variables they name.

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

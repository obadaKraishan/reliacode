# reliacode

**Validated LLM content analysis for communication research.**

reliacode helps researchers use large language models as coders in quantitative content
analysis without giving up the standards we hold human coders to. You write the codebook once, in
YAML: variables, categories, definitions, examples and decision rules. reliacode will then run it
against any OpenAI-compatible model, such as a local Ollama model, a vLLM server or a hosted API.
It will measure agreement with human coders (Krippendorff's alpha and Cohen's kappa, with bootstrap
confidence intervals), and correct prevalence estimates with prediction-powered inference. It
will also keep a ledger of tokens and cost, and write a methods-section paragraph that reports the
model, the prompt, the codebook's content hash and the reliability, so the analysis can be audited
and repeated.

> **Status: early development.** The codebook format, the `code()` runner, the ledger and the
> reliability module exist so far. The other modules are planned (see below and the issue
> tracker). Nothing is on PyPI yet, and the API will change before v0.1.0.

## Codebooks
A codebook is a YAML file. Here is a shortened version of
[`examples/codebooks/news_frames.yaml`](examples/codebooks/news_frames.yaml):

```yaml
name: news-frames
unit_of_analysis: One news article (headline and body text).
variables:
  - name: dominant_frame
    definition: The frame that most shapes how the article presents its main event or issue.
    decision_rules:
      - If two frames are equally prominent, code the one that appears first in the article.
    categories:
      - code: conflict
        definition: The article emphasizes disagreement between individuals, groups or institutions.
        examples:
          - Opposition leaders accused the minister of misleading parliament.
      - code: economic
        definition: The article stresses financial gains, losses or costs.
```

```python
import reliacode

codebook = reliacode.load_codebook("examples/codebooks/news_frames.yaml")
codebook.variable("dominant_frame").codes  # ('conflict', 'human_interest', 'economic', ...)
codebook.content_hash()  # 'sha256:b4bdbacd...', to report with results
```

If something is wrong, loading stops with a list of every problem, each with its line number and
a fix, such as `Line 4, variable "dominant_frame": "defintion" is not a recognized field. Did
you mean "definition"?`. Ordinal variables, the full list of checks and the hash recipe are
described in [docs/codebook-format.md](docs/codebook-format.md).

## Coding texts
`reliacode.code()` codes every text for every variable with a model, checks each answer against
the codebook's codes, and records how each code was obtained. With a local
[Ollama](https://ollama.com) server, no text leaves your computer:

```python
backend = reliacode.OpenAICompatibleBackend("http://localhost:11434/v1", "qwen3:4b")
run = reliacode.code(codebook, ["The library stays open late in exam week."], backend=backend)
run.table()  # one row per text, with one code per variable
```

An answer that is not one of the codes is asked again once, then recorded as missing with the
raw answers kept. Every result records the model, the endpoint, the parameters, the codebook's
content hash, the prompt hash and the time. Hosted services work the same way, with the API key
read from an environment variable; text sent to them leaves your computer.

By default each variable gets its own call. With `recipe="per_unit"`, one call per text asks for
all the variables together, and rules across variables can be stated in the prompt and checked
in the answer:

```python
rule = reliacode.Rule(
    text="Severity is none if and only if incivility_form is none.",
    variables=["incivility_form", "severity"],
    check=lambda codes: (codes["severity"] == "none") == (codes["incivility_form"] == "none"),
)
run = reliacode.code(codebook, texts, backend=backend, recipe="per_unit", rules=[rule])
```

A model that refuses a temperature takes `temperature=None`, and extra request fields such as
`{"reasoning_effort": "low"}` go in `extra_body`. Both are recorded with every answer. See
[docs/backends.md](docs/backends.md).

## Ledger
A ledger records every call `code()` makes: the model, the input and output tokens the server
reported, the latency, the tries and the outcome, tied to the codebook hash, the prompt hash and
a hash of the text. It never holds the coded text, the model's answer or the API key. With
prices you supply (reliacode ships none), it adds up the cost, and a budget stops a run before
it passes a limit:

```python
prices = reliacode.PriceTable(
    currency="USD",
    read_on="2026-11-19",
    prices={"example-model": reliacode.ModelPrice(input_per_million=0.5, output_per_million=1.5)},
)  # an invented model and invented prices
ledger = reliacode.Ledger("ledger.jsonl", prices=prices, budget=reliacode.Budget(cost_limit=5))
run = reliacode.code(codebook, texts, backend=backend, ledger=ledger)
ledger.summary().total  # calls, tokens in and out, cost, and the calls that could not be counted
```

A count the server did not report stays missing and is never taken as zero. See
[docs/ledger.md](docs/ledger.md).

## Reliability
reliacode measures how far coders agree, whether they are human coders, a model and human
coders, or two models. It gives Krippendorff's alpha (nominal or ordinal, as the codebook says)
for any number of coders and Cohen's kappa for two. Each figure has a bootstrap confidence
interval with a recorded seed, and sits beside the percent agreement, the number of units and
each coder's share of each code. Human codes come from a CSV file with the columns `unit`,
`coder`, `variable` and `value`, and the model's codes come from its run:

```python
codes = reliacode.read_codes(codebook, "human_codes.csv", run=run)  # the model is "model"
result = reliacode.codebook_reliability(codebook, codes, seed=20261124)
result.table()  # one row per variable: alpha and kappa with their intervals, units, agreement
```

A figure that is not defined, such as alpha when every value is the same code, is `None` with a
reason, never 0 or 1. See [docs/reliability.md](docs/reliability.md).

## Planned modules (v0.1)
| Module | What it does | Status |
|---|---|---|
| Codebook | YAML codebooks validated with pydantic, with plain-language error messages and a stable content hash | done |
| Backend adapter | One `code()` runner for any OpenAI-compatible endpoint | done ([#1](https://github.com/obadaKraishan/reliacode/issues/1)) |
| Cost ledger | Tokens, cost and timing for every call | done ([#2](https://github.com/obadaKraishan/reliacode/issues/2)) |
| Reliability | Krippendorff's alpha and Cohen's kappa with bootstrap CIs | done ([#3](https://github.com/obadaKraishan/reliacode/issues/3)) |
| One call per unit | All of a text's variables in one call, with rules across variables checked in the answer | done ([#22](https://github.com/obadaKraishan/reliacode/issues/22)) |
| PPI | Prediction-powered inference for prevalence estimates | planned ([#4](https://github.com/obadaKraishan/reliacode/issues/4)) |
| Methods exporter | A Markdown and LaTeX methods paragraph | planned ([#5](https://github.com/obadaKraishan/reliacode/issues/5)) |

A documentation site ([#6](https://github.com/obadaKraishan/reliacode/issues/6)) and two case studies on public human-coded data ([#7](https://github.com/obadaKraishan/reliacode/issues/7), [#8](https://github.com/obadaKraishan/reliacode/issues/8)) are planned for v0.1 as well. The [v0.1.0 milestone](https://github.com/obadaKraishan/reliacode/milestone/1) tracks the schedule.

## Development
```bash
git clone https://github.com/obadaKraishan/reliacode
cd reliacode
uv sync                 # Python 3.10+; installs pydantic, PyYAML, httpx, and ruff and pytest
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report problems and propose changes.

## License
MIT, see [LICENSE](LICENSE).

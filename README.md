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

> **Status: early development.** Only the codebook format exists so far. The other modules are
> planned (see below and the issue tracker). Nothing is on PyPI yet, and the API will change
> before v0.1.0.

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
codebook.content_hash()  # 'sha256:faf658c7...', to report with results
```

If something is wrong, loading stops with a list of every problem, each with its line number and
a fix, such as `Line 4, variable "dominant_frame": "defintion" is not a recognized field. Did
you mean "definition"?`. Ordinal variables, the full list of checks and the hash recipe are
described in [docs/codebook-format.md](docs/codebook-format.md).

## Planned modules (v0.1)
| Module | What it does | Status |
|---|---|---|
| Codebook | YAML codebooks validated with pydantic, with plain-language error messages and a stable content hash | done (S1) |
| Backend adapter | One `code()` runner for any OpenAI-compatible endpoint | planned ([#1](https://github.com/obadaKraishan/reliacode/issues/1)) |
| Cost ledger | Tokens, cost and timing for every call | planned ([#2](https://github.com/obadaKraishan/reliacode/issues/2)) |
| Reliability | Krippendorff's alpha and Cohen's kappa with bootstrap CIs | planned ([#3](https://github.com/obadaKraishan/reliacode/issues/3)) |
| PPI | Prediction-powered inference for prevalence estimates | planned ([#4](https://github.com/obadaKraishan/reliacode/issues/4)) |
| Methods exporter | A Markdown and LaTeX methods paragraph | planned ([#5](https://github.com/obadaKraishan/reliacode/issues/5)) |

A documentation site ([#6](https://github.com/obadaKraishan/reliacode/issues/6)) and two case studies on public human-coded data ([#7](https://github.com/obadaKraishan/reliacode/issues/7), [#8](https://github.com/obadaKraishan/reliacode/issues/8)) are planned for v0.1 as well. The [roadmap](docs/roadmap.md) has the schedule.

## Development
```bash
git clone https://github.com/obadaKraishan/reliacode
cd reliacode
uv sync                 # Python 3.10+; installs pydantic, PyYAML, and ruff and pytest for dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report problems and propose changes.

## AI-use disclosure
reliacode is developed with [Claude Code](https://claude.com/claude-code), Anthropic's AI coding
assistant. Claude Code drafts code, tests and documentation. The author, Obada Kraishan, reviews
every change before it is committed and is responsible for the package's design, its statistical
choices and its correctness. Commits made with Claude Code carry a `Co-Authored-By: Claude`
trailer, so the history shows which changes were AI-assisted.

## License
MIT, see [LICENSE](LICENSE).

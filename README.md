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

## Planned modules (v0.1)
| Module | What it does | Status |
|---|---|---|
| Codebook | YAML codebooks validated with pydantic, with plain-language error messages and a stable content hash | in progress |
| Backend adapter | One `code()` runner for any OpenAI-compatible endpoint | planned |
| Cost ledger | Tokens, cost and timing for every call | planned |
| Reliability | Krippendorff's alpha and Cohen's kappa with bootstrap CIs | planned |
| PPI | Prediction-powered inference for prevalence estimates | planned |
| Methods exporter | A Markdown and LaTeX methods paragraph | planned |

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

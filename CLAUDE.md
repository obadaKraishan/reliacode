# reliacode

Guidance for AI coding assistants (such as Claude Code) working in this repository. Human
contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) first; the rules below match it.

## Goal
A Python package for validated LLM content analysis: YAML codebooks, an OpenAI-compatible
`code()` runner, reliability with bootstrap CIs, PPI, a cost ledger and a methods exporter.
v0.1.0 is planned for early January 2027 (milestone v0.1.0); see `docs/roadmap.md`.

## Rules
- **The repository is public.** Never commit secrets, API keys, credentials, private data or raw
  research text. Case studies ship loaders that fetch data from its source, never the data.
- **Small, real, incremental commits** with clear messages, pushed as they are made. No code
  dumps, no backdating, no squashing, no force pushes.
- **`main` is protected.** Work on a branch and open a pull request per issue (`Closes #N`). Merge
  with a merge commit once CI is green; see the Workflow section of CONTRIBUTING.md.
- **AI use is disclosed.** Commits made with an AI assistant keep their `Co-Authored-By:`
  trailer, and the README's AI-use disclosure must stay accurate.
- **Never publish** (PyPI or TestPyPI, Zenodo, git tags, GitHub releases) without the
  maintainer's explicit approval.
- **The codebook content hash is a published identifier.** Any change to what it covers or how it
  is computed needs a new `HASH_VERSION`, a CHANGELOG entry and an updated golden test
  (`tests/test_hash.py`).
- Example codebooks cite the source they adapt and state their provenance in a header comment.
  Their example texts are invented.

## Coding conventions
- Python >=3.10 (develop on 3.12). Runtime dependencies stay minimal, currently pydantic v2 and
  PyYAML; a new one needs a reason in the PR and the CHANGELOG.
- Models are immutable pydantic models with `extra="forbid"`.
- Error messages and docs are written for communication researchers, not programmers: plain
  words, name the variable, category and line, and say how to fix the problem.
- American spelling. Ruff with line length 100.

## Tests and formatting
- `uv run pytest --cov=reliacode` must pass with coverage of at least 85%.
- Before every push, run `uv run ruff check .` and `uv run ruff format --check .`. Ruff also
  formats the Python code blocks inside Markdown files.
- Tests use small synthetic data and make no network calls.
- Statistical code needs regression tests against a reference implementation or a published
  result.
- Every user-visible change gets a line in `CHANGELOG.md` under `[Unreleased]`, and each working
  session adds five lines to `docs/session-log.md`: did, decided, next, open questions and usage
  notes.

## Layout
- `src/reliacode/`: `codebook.py` (models, loading, hash), `errors.py` (plain-language
  errors), `_yaml.py` (strict YAML loader with line numbers)
- `tests/` with `tests/fixtures/`; `examples/codebooks/`; `docs/`
- `.github/workflows/ci.yml`: ruff and pytest on Python 3.10 to 3.13

## Commands
```bash
uv sync
uv run pytest --cov=reliacode
uv run ruff check . && uv run ruff format --check .
gh run list --limit 3          # CI status
```

## Context pointers
@docs/roadmap.md

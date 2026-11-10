# reliacode (owner: Obada Kraishan, TTU CoMC; advisor: Dr. Kerk Kee)

## Goal
A Python package for validated LLM content analysis (YAML codebooks, an OpenAI-compatible
`code()` runner, reliability with bootstrap CIs, PPI, a cost ledger and a methods exporter).
v0.1.0 in about 8 weeks (around early January 2027); JOSS no earlier than May to June 2027.

## Hard rules
- Hub rules in ~/research/CLAUDE.md apply here: prompts/ and reports/ naming, commit and push
  after each step, plan mode first, and the ask-before list.
- **This repo is public** (or about to be). Everything committed is visible, so no secrets, no
  private-project details, no data. `prompts/` and `reports/` are gitignored here, and their
  committed copies live in the private hub (`~/research/prompts/`, `~/research/reports/`).
- **Steady history for JOSS:** small, real, incremental commits, pushed as they are made. Never a
  code dump, no backdating, no squashing. Keep the Co-Authored-By trailer (the README discloses AI use).
- Never run bulk labeling or inference in this session. Write scripts and run them on Kaggle
  (`kaggle/<job>/`) or the Batch API (`batch/`).
- Never print, log or commit secrets. `ANTHROPIC_API_KEY` must stay unset here.
- Never publish (PyPI, TestPyPI, Zenodo, git tag, GitHub release, Kaggle, HF) unless the file
  `RELEASE_APPROVED` is in this repo's root. Only Obada creates it. Do not register the PyPI name.
- Every data item carries provenance: `human_written` | `llm_drafted_human_edited` |
  `llm_drafted_human_validated`, plus the model name and license if an LLM touched it.
- Published outputs use Apache-2.0 or MIT models unless `docs/licenses.md` says otherwise.
- Raw data is read-only and never committed. Case studies ship loaders, not raw text.
- The codebook content hash is a published identifier. Any change to what it covers or how it is
  computed needs a new `HASH_VERSION`, a CHANGELOG entry and an updated golden test.

## Stack
Python 3.12 for development (supports >=3.10; CI runs 3.10 to 3.13), uv, hatchling, ruff,
pytest, pytest-cov, pydantic v2, PyYAML. LLM calls from scripts go through
`~/research/_shared/llm_client.py`; the package itself will ship its own OpenAI-compatible adapter (S2).

## Layout
- `src/reliacode/`: `codebook.py` (models, loading, hash), `errors.py` (plain-language
  errors), `_yaml.py` (strict YAML loader with line numbers)
- `tests/` with `tests/fixtures/`; `examples/codebooks/`; `docs/`
- `.github/workflows/ci.yml`; `prompts/` and `reports/` (local only)

## Commands
```bash
uv sync
uv run pytest --cov=reliacode
uv run ruff check . && uv run ruff format --check .
gh run list --limit 3          # CI status
```

## Definition of done (each session)
- Tests pass with coverage of at least 85%, and ruff is clean. CI is green.
- CHANGELOG.md is updated.
- docs/session-log.md gets 5 lines: did / decided / next / open questions / usage notes.
- A report is written to reports/ (local) and the hub's reports/, and the hub STATUS.md is updated.

## Context pointers
@docs/roadmap.md

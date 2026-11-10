# Session log

Five lines per session: did / decided / next / open questions / usage notes.

## 2026-11-10 · S1 codebook
- **Did:**
  - Created the repo (private until the maintainer makes it public) with uv, hatchling, ruff,
    pytest and CI on Python 3.10 to 3.13.
  - Built the codebook models, the strict YAML loader, plain-language errors and the content
    hash, plus two example codebooks and the format docs: 81 tests, 98% coverage.
  - Opened issues #1 to #8 for the rest of v0.1 under the v0.1.0 milestone.
- **Decided:**
  - Session prompts and reports stay out of this repository.
  - The roadmap here is the public version.
  - Codes are text, numbers are kept as text, and yes/no must be quoted.
  - One cross-field check can report several problems.
  - The hash leaves out defaults, so adding optional fields later does not change existing hashes.
- **Next:** S2, the backend adapter and cost ledger (#1, #2).
- **Open questions:** the author still needs to review the example codebooks, which say "author
  review pending" until then.
- **Usage notes:** the whole session ran locally; no model calls and no API credit.

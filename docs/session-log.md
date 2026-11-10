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

## 2026-11-10 · S1 review fixes
- **Did:**
  - Applied the author's review to both example codebooks: the incivility vulgarity example,
    the severity definitions and a none-if-and-only-if-none rule, and the news-frames
    responsibility wording.
  - Recorded provenance `llm_drafted_human_validated`.
  - New content hashes: `incivility.yaml` is
    `sha256:0a56109f1da29f0242fae1edf057785de91e1f0d2474227e977764aaa2b234dd`, and
    `news_frames.yaml` is
    `sha256:b4bdbacd8a7fae5da8725e4bd5c2f5f5a44187922e5ec380a2b21f2ee491bdb1`.
- **Decided:**
  - A top-level `decision_rules` field stays refused, with a pointer to `instructions`.
  - Cross-variable constraints are left for a later issue.
- **Next:** make the repo public, then turn on branch protection and a PR workflow.
- **Open questions:** none.
- **Usage notes:** local only; no model calls.

## 2026-11-10 · Going public
- **Did:**
  - Scanned the full history for secrets, emails, local paths and private notes before going
    public. No secrets were found; an early CLAUDE.md held local paths and private workflow notes,
    and it was replaced before the repo went public.
  - The repository was made public on 2026-11-10.
  - Set the repo description and topics, protected `main` (the four CI checks are required,
    including for admins; no force pushes or deletions), and added a Workflow section to
    CONTRIBUTING through the first pull request.
- **Decided:**
  - No required reviewers, since there is one maintainer.
  - Pull requests are merged with merge commits, never squashed.
- **Next:** S2, the backend adapter and cost ledger (#1, #2), each on its own branch and pull
  request.
- **Open questions:** none.
- **Usage notes:** local only; no model calls.

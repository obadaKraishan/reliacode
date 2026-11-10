# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Package skeleton: a uv project with a src layout, a hatchling build, Python >=3.10, pydantic and
  PyYAML at runtime, and ruff, pytest and pytest-cov for development. MIT license.
- README with the project goal, its status, the planned modules and an AI-use disclosure.
- CONTRIBUTING, a code of conduct (Contributor Covenant 2.1), the public roadmap
  (`docs/roadmap.md`) and GitHub Actions CI running ruff and pytest on Python 3.10 to 3.13.
- Codebook format (S1):
  - `reliacode.Codebook`, `Variable`, `Category` and `Example`: immutable pydantic models for
    variables, categories, definitions, examples, decision rules and optional ordinal levels.
  - `reliacode.load_codebook(path)` and `reliacode.parse_codebook(text)` read YAML. Duplicate
    keys are refused rather than silently overwritten.
  - `reliacode.CodebookError` lists every problem at once, in file order. Each entry gives the
    line, the variable and category, and a plain-language fix, including spelling suggestions
    for unknown fields and a hint for YAML's yes/no booleans.
  - `Codebook.content_hash()`: a SHA-256 fingerprint of the codebook's content, hash version 1.
    It ignores formatting, key order, comments and written-out defaults, and changes with any
    change in content or order. `Codebook.canonical_json()` returns the exact text hashed.
  - Example codebooks `examples/codebooks/news_frames.yaml` (adapted from Semetko and
    Valkenburg, 2000) and `examples/codebooks/incivility.yaml` (adapted from Coe, Kenski and
    Rains, 2014, plus an ordinal severity scale).
  - `docs/codebook-format.md`: the format, the checks, YAML pitfalls and the hash recipe.
- A Workflow section in CONTRIBUTING: `main` is protected, and changes go through one pull
  request per issue, merged with a merge commit once CI is green.

### Changed
- The example codebooks were revised after the author's review, so **both content hashes
  changed**:
  - `incivility.yaml`:
    - a new vulgarity example that is not also an aspersion
    - severity "mild" and "strong" definitions that put every accusation of lying on one level
    - a new severity rule: none if and only if `incivility_form` is none
    - new hash `sha256:0a56109f1da29f0242fae1edf057785de91e1f0d2474227e977764aaa2b234dd`
  - `news_frames.yaml`:
    - the responsibility wording now follows Semetko and Valkenburg's cause-or-solution framing
    - new hash `sha256:b4bdbacd8a7fae5da8725e4bd5c2f5f5a44187922e5ec380a2b21f2ee491bdb1`
  - Both files now record provenance `llm_drafted_human_validated`.
- The README's AI-use disclosure now describes the actual workflow: the author directs the work
  and reviews changes, and changes are not each reviewed before they are committed.
- The package description now matches the repository description.

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

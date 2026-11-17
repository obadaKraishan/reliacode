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
- Backend adapter and the `code()` runner:
  - `reliacode.code(codebook, texts, backend=...)` codes every text for every variable (or the
    ones named), one variable at a time, with a prompt built from the codebook. Each answer must
    be a JSON object whose code is exactly one of the variable's codes. An answer that cannot be
    used is asked again once (`attempts=2`), then recorded as missing with the reason and every
    raw answer kept.
  - `reliacode.CodingResult` and `reliacode.CodingRun` record, for every result, the model, the
    endpoint, the parameters, the codebook's content hash, the prompt hash and the time, with
    `table()` (one row per text) and `missing()`.
  - The prompt hash (prompt recipe version 1) covers the prompt, the message templates and the
    JSON Schema of the answer. `reliacode.coding.build_prompt()` shows the exact prompt.
  - `reliacode.OpenAICompatibleBackend` for any `/v1/chat/completions` endpoint (Ollama, vLLM,
    LM Studio, llama.cpp, OpenRouter and other hosted services), with no vendor library:
    temperature 0 by default, an optional seed and token limit, structured output through
    `response_format` (which can be turned off), a timeout, and retries with growing waits on
    "too many requests", server errors, timeouts and broken connections.
  - The API key is read from the environment variable named by `api_key_env` each time a
    request is sent, and goes only into the Authorization header. It is never stored, logged,
    shown in an error or kept in a result.
  - `reliacode.Response` and `reliacode.RequestParameters`: an immutable record of each answer
    (text, model, parameters, token counts, finish reason, server ids, latency, tries, UTC
    time). Token counts the server does not report are `None`, never zero.
  - `reliacode.BackendError`, in plain words, with a `reason` for each kind of problem: no
    server, model not found, no endpoint, missing or refused key, bad request, redirect, rate
    limit, server error, timeout, broken connection and unexpected answer.
  - `reliacode.FakeBackend`, with scripted answers, for testing code that uses reliacode.
  - `reliacode.Backend`, the interface any backend can implement.
  - `docs/backends.md`: a local example, a hosted example, the privacy risk of hosted services,
    the settings, the records, the errors and testing with `FakeBackend`.
- A new runtime dependency, `httpx` (>=0.27, <1), for the HTTP requests. The standard library's
  `urllib` could send them, but httpx lets the tests run the real client code through a mock
  transport with no network, brings its own certificates so HTTPS works on Python installs that
  lack them (such as the python.org installers on macOS), reuses connections across thousands
  of requests, and has clear timeouts and errors to turn into plain words. It brings anyio,
  certifi, h11, httpcore and idna.
- The tests block all network access, so no test can reach a real server.

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

### Removed
- Removed internal development notes and the roadmap from the repository: `CLAUDE.md`,
  `docs/roadmap.md` and `docs/session-log.md` are no longer tracked and are now gitignored, as
  `prompts/` and `reports/` already were. The README no longer has an AI-use section or a
  roadmap link (it links the v0.1.0 milestone instead), and CONTRIBUTING no longer describes
  the maintainer's own AI use. The git history still contains the removed files.

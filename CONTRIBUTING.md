# Contributing to reliacode

Thank you for helping. reliacode is a small research tool maintained by one person, so clear issues
and small pull requests help the most. Everyone who takes part agrees to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help
- **Report a problem.** Open an issue with what you ran, what you expected and what happened. If
  a codebook fails to load, include the smallest YAML file that shows the problem, and leave out
  any research data you are not allowed to share.
- **Say when an error message is unclear.** The codebook errors are meant to make sense to a
  communication researcher who is not a programmer. If one confused you, that counts as a bug.
- **Share a codebook.** Published, openly licensed codebooks make good examples and test cases.
- **Propose a change.** For anything beyond a small fix, open an issue first so we can agree on
  the approach before you write code.

## Development setup
reliacode uses [uv](https://docs.astral.sh/uv/) and supports Python 3.10 to 3.13.

```bash
git clone https://github.com/obadaKraishan/reliacode
cd reliacode
uv sync
uv run pytest                                      # tests
uv run pytest --cov=reliacode                      # with coverage (CI requires 85%)
uv run ruff check . && uv run ruff format --check .  # lint and format
```

## Workflow
`main` is protected: nobody pushes to it directly, the maintainer included.
1. Start from an issue. Open one first if there is none.
2. Work on a branch named for the change, such as `feature/backend-adapter` or `fix/yaml-tabs`,
   in small commits.
3. Open one pull request per issue, and write `Closes #N` in its description.
4. Merge when CI is green on all four Python versions. Pull requests are merged with a merge
   commit, never squashed, so the history keeps every commit.

## Pull requests
- Keep each pull request to one change, with tests for new behavior and for every bug fix.
- Run the tests and ruff before you push. CI runs both on Python 3.10, 3.11, 3.12 and 3.13.
- Add a line to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md).
- Write error messages and documentation for researchers: plain words, with an example where it
  helps.
- Statistical code needs a regression test against a reference implementation or a published
  result.

## Use of AI tools
You may use AI assistants. Please say so in the pull request, and make sure you have read and
understood every line you submit: you are responsible for it.

## Scope
v0.1 is frozen at the modules listed in the README: codebook, backend adapter, cost ledger,
reliability, PPI and the methods exporter. Ideas beyond that are welcome as issues and will be
considered after v0.1.0.

# reliacode roadmap

This is the public version of the project roadmap. Progress is tracked in the
[issues](https://github.com/obadaKraishan/reliacode/issues) and the
[v0.1.0 milestone](https://github.com/obadaKraishan/reliacode/milestones).

## 1. Goal and v1 scope

A Python package for validated LLM content analysis:

- YAML codebooks
- a `code()` runner that works with any OpenAI-compatible endpoint (local Ollama, Kaggle vLLM, OpenRouter, Claude)
- reliability (Krippendorff's alpha, Cohen's kappa) with bootstrap CIs
- prediction-powered inference (PPI) correction for prevalence estimates
- a cost and token ledger
- a methods-section exporter

**Targets:** v0.1 within about 8 weeks; JOSS submission no earlier than about May to June 2027.

## 2. Backends for tutorials and case studies

| Backend | Use |
|---|---|
| **Local models** through Ollama | Qwen3.5 4B or 9B as the default zero-cost backend in tutorials |
| **Kaggle T4x2** | Case-study runs with a 14B to 27B AWQ model via vLLM |
| **Hosted API** | Claude Haiku through the Batch API for one case study |
| **OpenRouter `:free`** | A second "coder" model for LLM-vs-LLM agreement |

## 3. Build sessions

| Session | Deliverable |
|---|---|
| S1 | Codebook schema (pydantic) and validation, loadable from YAML |
| S2 | Backend adapter (OpenAI-compatible) plus cost and token ledger |
| S3 | Reliability module: reuse the `krippendorff` package, add bootstrap CIs |
| S4 | PPI correction (reuse `ppi_py` where it fits) |
| S5 | Methods exporter: Markdown and LaTeX paragraph with model, version, date, prompt hash, alpha with CI |
| S6 | Docs site and two public case studies |
| S7 | CI, CONTRIBUTING, governance file, **v0.1.0 on PyPI** |

## 4. Data, licensing and ethics

- License the code under MIT or BSD-3.
- Case-study data must be public and redistributable. Keep raw text out of the repo and ship loaders instead.
- Document that sending text to third-party APIs carries privacy risk, and show the local Ollama path for sensitive data.

## 5. Quality checks

- Unit test coverage above 85%.
- Regression tests showing alpha matches the `krippendorff` reference on fixed data.
- Reproduce one published human-coded dataset's reliability.

## 6. Release checklist

- [ ] PyPI release and docs site
- [ ] Zenodo DOI per release, CITATION.cff
- [ ] CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT
- [ ] AI-use disclosure in the README
- [ ] At least one external tester

## 7. Timeline

| Week | Work |
|---|---|
| 1 | Public repo and S1 |
| 2 | S2 backend and ledger |
| 3 | S3 reliability |
| 4 | S4 PPI |
| 5 to 6 | S5 exporter and S6 docs and case studies |
| 7 to 8 | S7 and **v0.1.0** |
| Through spring | Monthly minor releases; used in research projects |
| About May to June 2027 | JOSS submission, once six months of history and a research use exist |

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Scope creep | Freeze v0.1 at five modules |
| Statistical errors | Regression tests against reference implementations |

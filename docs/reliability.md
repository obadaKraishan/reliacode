# Reliability: how far coders agree

Before codes can be trusted in an analysis, the coders who gave them have to agree well enough.
That is as true of a language model as of a trained human coder. reliacode measures agreement
between human coders, between a model and human coders, and between models, in the same way.
It gives Krippendorff's alpha for any number of coders and Cohen's kappa for two. Each comes
with a confidence interval and with the figures that tell you how to read it.

The input is codes, never text: for each unit (a comment, an article), each coder and each
variable, one value or a missing value.

## An example

This example uses the codebook in
[`examples/codebooks/incivility.yaml`](../examples/codebooks/incivility.yaml), which has a
nominal variable, `incivility_form`, and an ordinal one, `severity` (`none`, `mild`, `strong`).
Two human coders, anna and ben, coded eight comments. Their codes, all invented, are in a CSV file
in long form, with one line per unit, coder and variable. Ben left the severity of comment c8
empty, so it is missing.

```csv
unit,coder,variable,value
c1,anna,incivility_form,none
c1,anna,severity,none
c1,ben,incivility_form,none
c1,ben,severity,none
c2,anna,incivility_form,name_calling
c2,anna,severity,strong
c2,ben,incivility_form,name_calling
c2,ben,severity,strong
c3,anna,incivility_form,none
c3,anna,severity,none
c3,ben,incivility_form,none
c3,ben,severity,none
c4,anna,incivility_form,aspersion
c4,anna,severity,mild
c4,ben,incivility_form,aspersion
c4,ben,severity,mild
c5,anna,incivility_form,vulgarity
c5,anna,severity,strong
c5,ben,incivility_form,vulgarity
c5,ben,severity,mild
c6,anna,incivility_form,none
c6,anna,severity,none
c6,ben,incivility_form,aspersion
c6,ben,severity,mild
c7,anna,incivility_form,lying
c7,anna,severity,mild
c7,ben,incivility_form,lying
c7,ben,severity,mild
c8,anna,incivility_form,name_calling
c8,anna,severity,strong
c8,ben,incivility_form,none
c8,ben,severity,
```

Read the codes and compare the two coders on every variable:

```python
import reliacode

codebook = reliacode.load_codebook("examples/codebooks/incivility.yaml")
humans = reliacode.read_codes(codebook, "human_codes.csv")
result = reliacode.codebook_reliability(codebook, humans, seed=20261124)

for row in result.table():
    print(
        f"{row['variable']}: alpha {row['alpha']:.2f} [{row['alpha_low']:.2f}, "
        f"{row['alpha_high']:.2f}], kappa {row['kappa']:.2f}, "
        f"{row['percent_agreement']:.0f}% agreement on {row['units']} units"
    )
```

```text
incivility_form: alpha 0.69 [0.17, 1.00], kappa 0.67, 75% agreement on 8 units
severity: alpha 0.76 [0.07, 1.00], kappa 0.58, 71% agreement on 7 units
```

Severity has 7 units, not 8, because comment c8 has a value from anna only, and one value cannot
be paired. The intervals are wide because eight comments say little. That is why the interval is
reported: alpha 0.76 from 7 units is a much weaker claim than the same figure from 300.

Each result keeps the details. `result.variable("severity").alpha.coder_shares` shows that anna
coded 43% of the comments as `none`, while ben coded 29% that way. Kappa for severity carries a
note: it is unweighted, so ben's `mild` for anna's `strong` counts as a full miss, while
alpha, which uses the order of the levels, counts it as a near miss.

Now add a model as a third coder. `code()` codes the same comments, with the same ids, and
`read_codes()` adds the run as the coder `"model"`. Here the model's answers are scripted with
`FakeBackend`, so the example runs without a server. With a real model, use a backend such as
`OpenAICompatibleBackend("http://localhost:11434/v1", "qwen3:4b")`.

```python
import json

comments = {f"c{n}": f"Invented comment {n}." for n in range(1, 9)}
model_answers = [
    ("none", "none"),
    ("name_calling", "strong"),
    ("none", "mild"),
    ("lying", "mild"),
    ("vulgarity", "strong"),
    ("aspersion", "mild"),
    ("lying", "mild"),
    ("name_calling", "strong"),
]
backend = reliacode.FakeBackend(
    [json.dumps({"code": code}) for pair in model_answers for code in pair]
)
run = reliacode.code(codebook, comments, backend=backend)

codes = reliacode.read_codes(codebook, "human_codes.csv", run=run)
for human in ("anna", "ben"):
    pair = reliacode.variable_reliability(
        codebook, codes, "severity", coders=[human, "model"], seed=20261124
    )
    low, high = pair.alpha.interval.low, pair.alpha.interval.high
    print(f"{human} and the model: alpha {pair.alpha.value:.2f} [{low:.2f}, {high:.2f}]")
```

```text
anna and the model: alpha 0.84 [0.19, 1.00]
ben and the model: alpha 0.69 [-0.02, 1.00]
```

`reliacode.codebook_reliability(codebook, codes)` compares all three coders at once. It gives
alpha only, since kappa compares exactly two.

## The codes file

A codes file is a CSV file with the columns `unit`, `coder`, `variable` and `value`, in any
order, with one line per unit, coder and variable.

- **A missing value is an empty cell.** It is left out, never counted as a category. A word
  such as `NA` or `missing` is refused, with a hint to leave the cell empty.
- **Every value is checked against the codebook.** A value that is not one of the variable's
  codes is an error that names the line, the unit, the coder and the variable, such as
  `Line 5, unit "c1", coder "ben", variable "severity": "Mild" is not one of the codes of
  severity: none, mild, strong. Did you mean "mild"? Codes must match exactly, including upper
  and lower case.` Every problem in the file is listed at once.
- **Other columns are ignored**, so a column of notes can stay. The file is read as UTF-8, with
  or without the byte-order mark that spreadsheet programs add.
- **Unit ids are compared as text.** For a model's codes to meet the human codes, the `unit`
  column must hold the ids the texts had in `code()`: the keys of a mapping such as
  `{"c1": text, ...}`, or the positions 0, 1, 2 of a list.

Codes in a data frame can be passed as rows, where `None` and NaN are missing values:

```python
codes = reliacode.read_codes(codebook, df.to_dict("records"))
```

A data frame in wide form, with one column per coder, can be turned into long form first:

```python
long = wide.melt(id_vars=["unit", "variable"], var_name="coder", value_name="value")
```

Several models can be compared with each other by naming their runs:
`reliacode.read_codes(codebook, run={"qwen3": run_a, "gemma3": run_b})`.

## What each figure means

- **Krippendorff's alpha** is 1 minus the ratio of the disagreement observed to the disagreement
  expected by chance. 1 means perfect agreement, 0 means agreement no better than chance, and a
  figure below 0 means systematic disagreement; it is reported as it is. Alpha works for any
  number of coders, and coders may skip units. It uses every unit with values from at least two
  coders, and a unit with one value adds nothing.
- **The level of measurement comes from the codebook.** For a nominal variable every
  disagreement counts the same. For an ordinal variable the categories are ranked by their
  `level`, and Krippendorff's ordinal metric makes a near miss count less than a far one. The
  ranks depend on how often each category is used, so an unused level changes nothing.
- **Cohen's kappa** compares exactly two coders, on the units both coded. Like alpha, it
  corrects the agreement for chance. Alpha takes chance from both coders' values pooled, while
  kappa takes each coder's own distribution, so the two differ when the coders use the categories
  at different rates. Kappa here is unweighted: it treats the categories as unordered, even for
  an ordinal variable. With more than two coders there is no kappa, and the result's `notes` say
  so. Name two coders with `coders=[...]` to compare a pair.
- **Percent agreement** is the share of pairs of values, within a unit, that are the same. For
  two coders, it is the share of units on which they agree. It is not corrected for chance, so
  it is high whenever one code is common. In Bland's (2015) Table 20.6, two raters agree on
  80% of 100 cases, yet kappa is 0, because one of them always says "no". Report it beside alpha,
  not instead of it.
- **Units and pairable values** count what each figure was computed on. For alpha, these are
  the units with at least two values and the values in them. For kappa, they are the units both
  coders coded.
- **Each coder's counts and shares of each code** (`coder_counts`, `coder_shares`) are taken
  over the same values. A low alpha for a rare code, or a high percent agreement for a common
  one, can be read against them, and a coder who uses a code much more often than the others
  stands out.

## The interval and its seed

Each figure comes with a **percentile bootstrap interval over units**. reliacode draws as many
units as were compared, with replacement, and computes the figure again; it does this 2,000 times
by default. The interval holds the middle 95% of those figures (`confidence=0.95`), using linear
interpolation between the two nearest values, as numpy and R do by default.

- **Units are resampled because units are what was sampled.** The texts in a reliability
  sample stand for the texts that were not coded twice. Resampling units keeps each unit's values
  together, with whatever coders it had.
- **The alternative.** Hayes and Krippendorff (2007), in their KALPHA macro, bootstrap alpha by
  resampling pairs of values from the coincidence matrix rather than units. The two schemes
  answer slightly different questions and can give somewhat different intervals, so a methods
  section should say which was used.
- **The seed.** `seed=` fixes the random draws, so the same seed gives the same interval again,
  on every supported Python version. When no seed is given, one is drawn, and either way the
  result records it in `seed`, with `resamples`, `confidence` and `method`. Report the seed with
  the interval, and anyone can repeat it.
- **How well the interval covers.** In this package's tests, 600 simulated samples of 60 units
  each gave 90% intervals that covered the true value 88% of the time for alpha and 87% for
  kappa. Percentile intervals run a little narrow on small samples, so treat an interval from a
  few dozen units as rough.
- **Resamples with one code only.** A resample can, by chance, hold one code only. Alpha and
  kappa are not defined on it. Such resamples are counted and left out, and the result's
  `notes` say how many. When fewer than half of the resamples give a value, there is no
  interval.
- **No interval at all** comes with `resamples=0`, which is useful for a quick look.

## Undefined and degenerate cases

A figure that cannot be computed is `None`, with the reason in `reason`, never 0 or 1:

| Case | What the result says |
|---|---|
| No values for the variable | `No alpha: there are no values for this variable. None was given, or every value is missing.` |
| No unit coded by two coders | `No alpha: no unit has values from two or more coders, so there are no pairs of values to compare.` |
| One code everywhere | `No alpha: every pairable value is "none", so there is no variation. Alpha is undefined here (the disagreement expected by chance is zero), not 0 or 1.` |
| Kappa with one code everywhere | `No kappa: both coders gave "none" to every unit they both coded, so the agreement expected by chance is 1 and kappa is undefined, not 0 or 1.` |
| Kappa with no unit coded by both | `No kappa: no unit has a code from both coders.` |
| Not exactly two coders | `kappa` is `None`, and `notes` says: `No Cohen's kappa: it compares exactly two coders, and these codes have 3 (anna, ben, model). Choose two with coders=["...", "..."] to compare a pair.` |

Two cases give a figure but deserve a warning, and the result's `notes` give it:

- **Every pair agrees.** Alpha is 1, and every resample gives 1 too, so the interval is
  [1, 1]. That interval understates the uncertainty: it shows only that no disagreement was seen
  among these units, not that there would be none among others. The note says so.
- **Any other interval that collapses to a point**, for example when every unit holds the same
  pair of codes, is noted as well.

## Reporting reliability in a methods section

Report alpha for each variable with its interval, the number of units and coders, the level of
measurement, and the percent agreement and code distribution beside it. Name the bootstrap
(percentile, over units), the number of resamples, the seed, the version of reliacode and the
codebook's content hash, so that a reader can compute the same figures again.

For example: "Two coders independently coded a random sample of 300 of the 4,000 comments.
Agreement on the form of incivility was Krippendorff's alpha = 0.81 (nominal; 95% CI [0.76,
0.86], percentile bootstrap over comments, 2,000 resamples, seed 20261124), with 88% agreement,
computed with reliacode 0.1.0 and codebook sha256:0a56109f...". The figures in this sentence are
invented.

## References

- Bland, M. (2015). *An introduction to medical statistics* (4th ed.). Oxford University Press.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
  Psychological Measurement, 20*(1), 37-46. https://doi.org/10.1177/001316446002000104
- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure
  for coding data. *Communication Methods and Measures, 1*(1), 77-89.
  https://doi.org/10.1080/19312450709336664
- Krippendorff, K. (2011). *Computing Krippendorff's alpha-reliability*. Annenberg School for
  Communication, University of Pennsylvania.
  https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf

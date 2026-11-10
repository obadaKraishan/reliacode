# The codebook format

A reliacode codebook is a YAML file that says what coders, human or model, should record for each
unit of analysis. It has the parts of a codebook in a methods appendix: variables, categories,
definitions, examples and decision rules. Two complete examples are in
[`examples/codebooks/`](../examples/codebooks/).

```python
import reliacode

codebook = reliacode.load_codebook("examples/codebooks/news_frames.yaml")
codebook.variable("dominant_frame").codes  # ('conflict', 'human_interest', ...)
codebook.content_hash()  # 'sha256:...'
```

## Structure

```yaml
name: news-frames                 # required: a short name
version: "0.1"                    # optional; put it in quotes (see "YAML pitfalls")
title: Generic news frames        # optional
description: What the codebook measures.          # optional
language: en                      # optional: language of the texts being coded
unit_of_analysis: One news article.               # optional, but recommended
instructions: General instructions for coders.    # optional
authors: [A. Researcher]          # optional list
references:                       # optional list: the published sources it builds on
  - Semetko, H. A., & Valkenburg, P. M. (2000). ...

variables:                        # required: at least one
  - name: dominant_frame          # required: letters, digits and underscores, starting with a letter
    label: Dominant frame         # optional: a readable name
    definition: What the variable measures.       # required
    type: nominal                 # optional: nominal (the default) or ordinal
    instructions: How to code this variable.      # optional
    decision_rules:               # optional list: rules for hard cases
      - If two frames are equally prominent, code the one that appears first.
    categories:                   # required: at least two
      - code: conflict            # required: what gets recorded in the data
        label: Conflict           # optional
        definition: When coders should choose this category.   # required
        examples:                 # optional list
          - A plain example text.
          - text: An example with a note.
            note: Why it belongs here.
```

Field by field:

| Where | Field | Required | Notes |
|---|---|---|---|
| codebook | `name` | yes | A short name, such as `news-frames`. |
| codebook | `version`, `title`, `description`, `language`, `unit_of_analysis`, `instructions` | no | Text. |
| codebook | `authors`, `references` | no | Lists of text. |
| codebook | `variables` | yes | At least one variable. Names must be unique. |
| variable | `name` | yes | Becomes a column name in coded data, so: letters, digits and `_`, starting with a letter. |
| variable | `definition` | yes | What the variable measures. |
| variable | `type` | no | `nominal` (default) or `ordinal`. |
| variable | `label`, `instructions` | no | Text. |
| variable | `decision_rules` | no | A list of text. |
| variable | `categories` | yes | At least two. Codes must be unique within the variable. |
| category | `code` | yes | Text. Numbers are kept as text, so `code: 1` is recorded as `"1"`. |
| category | `definition` | yes | When to choose this category. |
| category | `label` | no | Text. |
| category | `examples` | no | Each is plain text, or `text:` with an optional `note:`. |
| category | `level` | ordinal only | A whole number giving the category's rank. |

## Ordinal variables

Set `type: ordinal` when the categories have a natural order, and give every category a
`level`: a whole number, unique within the variable. Higher levels mean more of the quantity.
Levels do not have to start at 0 or be consecutive, and the categories can be listed in any
order. A nominal variable must not have levels.

```yaml
  - name: severity
    definition: How strongly uncivil the comment is.
    type: ordinal
    categories:
      - {code: none, level: 0, definition: No uncivil language.}
      - {code: mild, level: 1, definition: Incivility aimed at ideas.}
      - {code: strong, level: 2, definition: Incivility aimed at people.}
```

## Validation and error messages

Loading a codebook checks it fully and reports every problem at once, in file order. Each
problem names the variable and category involved, gives the line in the file, and says how to
fix it. For this file, `frames.yaml`:

```yaml
name: news-frames
variables:
  - name: dominant_frame
    defintion: The frame that most shapes the article.
    categories:
      - code: conflict
        label: Conflict
      - code: economic
        definition: Costs and gains.
      - code: no
        definition: No frame.
```

`reliacode.load_codebook("frames.yaml")` raises:

```text
reliacode.errors.CodebookError: The codebook in frames.yaml has 3 problems:
  1. Line 4, variable "dominant_frame": "defintion" is not a recognized field. Did you mean "definition"?
  2. Line 6, variable "dominant_frame" > category "conflict": The required field "definition" is missing. Every category needs a definition that tells coders when to choose it.
  3. Line 10, variable "dominant_frame" > category 3: "code" was read as the value false, not as text. YAML reads unquoted yes, no, on, off, true and false as true/false values. Put the value in quotes, for example "no" instead of no.
```

`CodebookError` is a `ValueError`. Its `problems` attribute holds each problem as a `Problem`
with `message`, `location` and `line`, for tools that want to show them differently.

The checks:

- The file is valid YAML, holds one document, and repeats no key within a block. Plain YAML
  silently keeps the last of two identical keys; reliacode refuses.
- Every required field is present and not empty, and every field name is known. Misspellings
  get a suggestion.
- Lists are lists, text is text, and levels are whole numbers.
- Variable names are unique and usable as column names. Codes are unique within a variable.
- Every variable has at least two categories.
- Ordinal variables have a unique level for every category; nominal variables have none.

Text is cleaned as it loads: surrounding spaces and line breaks are removed, and Unicode is
normalized to NFC, so the same letters typed on different systems compare equal.

## YAML pitfalls

- **yes, no, on, off, true and false** become true/false values unless quoted. Write
  `code: "no"`.
- **Version numbers** such as `1.10` are read as the number 1.1. Write `version: "1.10"`.
- **A colon followed by a space** inside a text ends the key. Quote the text, as in
  `definition: "Conflict: disagreement between actors"`, or use a block (`>` or `|`).
- **Tabs** cannot be used for indentation. Use spaces.
- **Save as UTF-8.** Files in other encodings are refused with a message saying so.

## The content hash

`Codebook.content_hash()` returns a fingerprint such as
`sha256:b4bdbacd…`. Report it with your results. Anyone holding the codebook file can then check
that it is the one you used, and two analyses with the same hash used the same codebook.

The hash depends on what the codebook says, not on how the file is written. These do **not**
change it:

- key order, comments, indentation, quoting style, and block or flow style
- surrounding whitespace, and Unicode normal form
- writing out a default (`type: nominal`, `examples: []`) instead of leaving it out
- writing an example as plain text instead of `text:`

These **do** change it: any change to a name, definition, label, example, rule, code or level;
adding or removing an optional field; and reordering variables, categories, examples or rules.
Order counts because it reaches the prompts a model sees.

### Recipe (hash version 1)

1. Validate the codebook. This trims text and normalizes it to NFC.
2. Convert it to JSON-compatible data, leaving out every field that still has its default value
   (empty lists, `null`, `type: nominal`).
3. Wrap it as `{"reliacode_codebook_hash": 1, "codebook": <data>}`.
4. Serialize as JSON with sorted keys, no spaces (`","` and `":"` separators) and non-ASCII
   characters kept as they are, and encode the result as UTF-8.
5. Take the SHA-256 digest, written as `sha256:` plus 64 lowercase hex digits.

`Codebook.canonical_json()` returns the exact text from step 4, so the hash can be checked with
any SHA-256 tool:

```python
import hashlib

text = codebook.canonical_json()
assert codebook.content_hash() == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Any change to this recipe gets a new hash version number, which changes every hash. The
CHANGELOG records it.

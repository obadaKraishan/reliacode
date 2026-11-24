"""Reliability: how far coders agree, with confidence intervals a methods section can report.

The input is codes, never text: one value, or a missing value, for each unit, coder and
variable. :func:`read_codes` reads human codes from a CSV file in long form and adds the codes of
a :class:`~reliacode.CodingRun`, where the model is one coder, and checks every value against the
codebook.
"""

import csv
import difflib
import io
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, model_validator

from reliacode.codebook import Codebook, Variable
from reliacode.coding import CodingRun
from reliacode.errors import Problem, describe

#: The columns of a codes file, in long form: one line per unit, coder and variable.
COLUMNS = ("unit", "coder", "variable", "value")
#: The coder name a single run gets.
MODEL_CODER = "model"
#: Values that look like a way of writing "missing". In a codes file, missing is an empty cell.
_MISSING_WORDS = {"na", "n/a", "nan", "null", "none", "missing", "-", ".", "?"}
#: How many problems an error lists before it only counts the rest.
_SHOWN = 20


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodedValue(_Record):
    """One coder's value for one unit and variable. ``value`` is ``None`` when it is missing."""

    unit: str
    coder: str
    variable: str
    value: str | None = None


class Codes(_Record):
    """The codes to compare: one value, or a missing value, for each unit, coder and variable.

    Build it with :func:`read_codes`, which checks every value against the codebook. Unit ids
    are text, so the text with id 7 in a run and the unit "7" in a CSV file are the same unit.
    """

    values: tuple[CodedValue, ...]

    @model_validator(mode="after")
    def _one_value_each(self) -> "Codes":
        seen: set[tuple[str, str, str]] = set()
        for item in self.values:
            key = (item.unit, item.coder, item.variable)
            if key in seen:
                raise ValueError(
                    f'Unit "{item.unit}", coder "{item.coder}", variable "{item.variable}" has '
                    "more than one value. Each coder gives one value per unit and variable."
                )
            seen.add(key)
        return self

    @property
    def coders(self) -> tuple[str, ...]:
        """The coders' names, sorted."""
        return tuple(sorted({item.coder for item in self.values}))

    @property
    def units(self) -> tuple[str, ...]:
        """The unit ids, sorted."""
        return tuple(sorted({item.unit for item in self.values}))

    @property
    def variables(self) -> tuple[str, ...]:
        """The variables that have codes, sorted."""
        return tuple(sorted({item.variable for item in self.values}))

    def rows(self) -> list[dict[str, str | None]]:
        """One row per value, with the columns unit, coder, variable and value.

        ``pandas.DataFrame(codes.rows())`` turns it into a data frame.
        """
        return [item.model_dump() for item in self.values]


class CodesError(ValueError):
    """The codes could not be read, or do not fit the codebook. ``problems`` lists every one."""

    def __init__(self, problems: Sequence[Problem], source: str | None = None) -> None:
        self.problems = tuple(problems)
        self.source = source
        super().__init__(self._render())

    def _render(self) -> str:
        count = len(self.problems)
        where = f" in {self.source}" if self.source else ""
        lines = [f"The codes{where} have {count} {'problem' if count == 1 else 'problems'}:"]
        lines += [f"  {n}. {problem}" for n, problem in enumerate(self.problems[:_SHOWN], 1)]
        if count > _SHOWN:
            lines.append(f"  ... and {count - _SHOWN} more.")
        return "\n".join(lines)


def read_codes(
    codebook: Codebook,
    source: str | os.PathLike[str] | Iterable[Mapping[str, Any]] | None = None,
    *,
    run: CodingRun | Mapping[str, CodingRun] | None = None,
) -> Codes:
    """Read the codes to compare, and check every value against the codebook.

    Args:
        codebook: the codebook the units were coded with.
        source: human codes, as the path of a CSV file or as rows, such as
            ``df.to_dict("records")``. Either way there is one row per unit, coder and
            variable, with the columns ``unit``, ``coder``, ``variable`` and ``value``. Other
            columns are ignored. An empty value is a missing value (in rows, also ``None`` or
            NaN), never a category.
        run: a :class:`~reliacode.CodingRun` from :func:`reliacode.code`, whose model becomes
            the coder ``"model"``, or a mapping from coder names to runs, such as
            ``{"qwen3": run_a, "gemma3": run_b}``, to compare models. A code that could not be
            obtained is a missing value. A run's unit ids are the ids its texts had: positions
            0, 1, 2 for a list of texts, or the keys of a mapping.

    Raises:
        CodesError: a value is not one of its variable's codes, a variable is not in the
            codebook, a unit, coder and variable have more than one value, a run was coded with
            another codebook, or the file cannot be read. Every problem is listed in file order,
            with its line, unit, coder and variable.
        FileNotFoundError: there is no file at ``source``.
        TypeError: the arguments are not as described.
    """
    if not isinstance(codebook, Codebook):
        raise TypeError(
            'codebook should be a codebook loaded with reliacode.load_codebook("file.yaml").'
        )
    if source is None and run is None:
        raise TypeError("Give the codes to compare: a CSV file or rows (source), a run, or both.")
    problems = _Problems()
    entries: list[_Entry] = []
    name = None
    if source is not None:
        if isinstance(source, (str, os.PathLike)):
            name = str(source)
            entries += _csv_entries(Path(source), problems)
        else:
            entries += _row_entries(source, problems)
    human_coders = {entry.value.coder for entry in entries}
    for coder, coding_run in _named_runs(run).items():
        if coder in human_coders:
            problems.add(
                _AFTER_SOURCE,
                Problem(
                    f'The coder name "{coder}" is used by the codes and by a run. Give the run '
                    f'another name, for example run={{"{coder}_run": run}}.'
                ),
            )
        entries += _run_entries(codebook, coder, coding_run, problems)
    _check_entries(codebook, entries, problems)
    if problems:
        raise CodesError(problems.in_order(), name)
    return Codes(values=tuple(entry.value for entry in entries))


#: Where the problems and values of runs come in the order: after every line or row.
_AFTER_SOURCE = 2**62


class _Entry(NamedTuple):
    """A value as read, with where it came from: its place in words, its line in a file, and
    its position in the order problems are listed."""

    value: CodedValue
    where: str
    line: int | None
    order: int


class _Problems:
    """The problems found, each with its position, so they can be listed in file order."""

    def __init__(self) -> None:
        self._found: list[tuple[int, Problem]] = []

    def add(self, order: int, problem: Problem) -> None:
        self._found.append((order, problem))

    def __bool__(self) -> bool:
        return bool(self._found)

    def in_order(self) -> list[Problem]:
        return [problem for _, problem in sorted(self._found, key=lambda found: found[0])]


def _csv_entries(path: Path, problems: _Problems) -> list[_Entry]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problems.add(
            0,
            Problem(
                "The file is not saved as UTF-8 text. Open it in a spreadsheet or text editor "
                'and save it again as "CSV UTF-8".'
            ),
        )
        return []
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or not any(cell.strip() for cell in header):
        problems.add(
            0,
            Problem(
                "The file is empty. Its first line should name the columns unit, coder, "
                "variable and value, followed by one line per unit, coder and variable."
            ),
        )
        return []
    names = [cell.strip() for cell in header]
    absent = [column for column in COLUMNS if column not in names]
    if absent:
        problems.add(
            1,
            Problem(
                f"The first line should name the columns unit, coder, variable and value, but "
                f"{_and(absent)} {'is' if len(absent) == 1 else 'are'} missing. The first line "
                f"is: {','.join(names)}.",
                line=1,
            ),
        )
        return []
    column_of = {column: names.index(column) for column in COLUMNS}
    entries = []
    for cells in reader:
        line = reader.line_num
        if not any(cell.strip() for cell in cells):
            continue
        cells = cells + [""] * (len(names) - len(cells))
        unit, coder, variable, value = (cells[column_of[column]].strip() for column in COLUMNS)
        found = _value_of(unit, coder, variable, value or None)
        if isinstance(found, str):
            problems.add(line, Problem(found, line=line))
        else:
            entries.append(_Entry(found, f"line {line}", line, line))
    return entries


def _row_entries(rows: Iterable[Mapping[str, Any]], problems: _Problems) -> list[_Entry]:
    if isinstance(rows, Mapping) or not isinstance(rows, Iterable):
        raise TypeError(
            "source should be the path of a CSV file, or a list of rows with the fields unit, "
            f"coder, variable and value, but it is {describe(rows)}."
        )
    entries = []
    for number, row in enumerate(rows, 1):
        where = f"row {number}"
        if not isinstance(row, Mapping):
            message = (
                f"This row is {describe(row)}, but each row should have the fields unit, coder, "
                "variable and value."
            )
            problems.add(number, Problem(message, where))
            continue
        absent = [column for column in COLUMNS if column not in row]
        if absent:
            problems.add(number, Problem(f"This row has no {_and(absent)}.", where))
            continue
        fields = [_text(row[column], column) for column in COLUMNS[:3]]
        fields.append(_value(row["value"]))
        wrong = [field.problem for field in fields if isinstance(field, _Wrong)]
        if wrong:
            problems.add(number, Problem(wrong[0], where))
            continue
        unit, coder, variable, value = fields
        found = _value_of(unit, coder, variable, value)  # type: ignore[arg-type]
        if isinstance(found, str):
            problems.add(number, Problem(found, where))
        else:
            entries.append(_Entry(found, where, None, number))
    return entries


class _Wrong(NamedTuple):
    """A field of a row that cannot be used, and why."""

    problem: str


def _text(value: object, column: str) -> str | _Wrong:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return _Wrong(f"The {column} should be text or a whole number, but it is {describe(value)}.")


def _value(value: object) -> str | _Wrong | None:
    """The value as text, or None when it is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value.strip() or None
    return _Wrong(
        "The value should be one of the variable's codes, or empty when it is missing, but it "
        f"is {describe(value)}."
    )


def _value_of(unit: str, coder: str, variable: str, value: str | None) -> CodedValue | str:
    """The coded value, or what is wrong with it."""
    texts = (unit, coder, variable)
    empty = [column for column, text in zip(COLUMNS[:3], texts, strict=True) if not text]
    if empty:
        return (
            f"The {_and(empty)} {'is' if len(empty) == 1 else 'are'} empty. Every code needs the "
            "unit that was coded, the coder who coded it and the variable."
        )
    return CodedValue(unit=unit, coder=coder, variable=variable, value=value)


def _named_runs(run: object) -> dict[str, CodingRun]:
    if run is None:
        return {}
    if isinstance(run, CodingRun):
        return {MODEL_CODER: run}
    if isinstance(run, Mapping):
        runs = {}
        for coder, coding_run in run.items():
            if not isinstance(coder, str) or not coder.strip():
                raise TypeError(f"The names of runs should be text, but one is {describe(coder)}.")
            if not isinstance(coding_run, CodingRun):
                raise TypeError(
                    f'The run "{coder}" should be what reliacode.code() returned, but it is '
                    f"{describe(coding_run)}."
                )
            runs[coder.strip()] = coding_run
        return runs
    raise TypeError(
        "run should be what reliacode.code() returned, or a mapping from coder names to runs, "
        f"but it is {describe(run)}."
    )


def _run_entries(
    codebook: Codebook, coder: str, run: CodingRun, problems: _Problems
) -> list[_Entry]:
    expected = codebook.content_hash()
    if run.codebook_hash != expected:
        message = (
            f'The run "{coder}" was coded with another codebook ({run.codebook_hash}) than the '
            f"one given ({expected}). Load the codebook the run used, or code the texts again "
            "with this one."
        )
        problems.add(_AFTER_SOURCE, Problem(message))
        return []
    return [
        _Entry(
            CodedValue(
                unit=str(result.unit), coder=coder, variable=result.variable, value=result.code
            ),
            f'the run "{coder}"',
            None,
            _AFTER_SOURCE,
        )
        for result in run.results
    ]


def _check_entries(codebook: Codebook, entries: list[_Entry], problems: _Problems) -> None:
    codes = {variable.name: variable.codes for variable in codebook.variables}
    first: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        item = entry.value
        place = _place(item, entry.where if entry.line is None else "")
        if item.variable not in codes:
            hint = _suggestion(item.variable, list(codes))
            message = (
                f'The codebook has no variable "{item.variable}". Its variables are: '
                f"{', '.join(codes)}.{hint}"
            )
            problems.add(entry.order, Problem(message, place, entry.line))
            continue
        if item.value is not None and item.value not in codes[item.variable]:
            message = _unknown_code(item.value, item.variable, codes[item.variable])
            problems.add(entry.order, Problem(message, place, entry.line))
        key = (item.unit, item.coder, item.variable)
        if key in first:
            message = (
                f"This unit, coder and variable already have a value ({first[key]}). Each coder "
                "gives one value per unit and variable: keep one and remove the other."
            )
            problems.add(entry.order, Problem(message, place, entry.line))
        else:
            first[key] = entry.where


def _place(item: CodedValue, where: str) -> str:
    about = f'unit "{item.unit}", coder "{item.coder}", variable "{item.variable}"'
    return f"{where}, {about}" if where else about


def _unknown_code(value: str, variable: str, codes: Sequence[str]) -> str:
    message = f'"{value}" is not one of the codes of {variable}: {", ".join(codes)}.'
    folded = [code for code in codes if code.casefold() == value.casefold()]
    if folded:
        return (
            f'{message} Did you mean "{folded[0]}"? Codes must match exactly, including upper '
            "and lower case."
        )
    if value.casefold() in _MISSING_WORDS:
        return f"{message} For a missing value, leave the value empty."
    return message + _suggestion(value, list(codes))


def _suggestion(word: str, options: list[str]) -> str:
    close = difflib.get_close_matches(word, options, n=1)
    return f' Did you mean "{close[0]}"?' if close else ""


def _and(words: Sequence[str]) -> str:
    quoted = [f'"{word}"' for word in words]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


# Agreement -------------------------------------------------------------------------------------


class Agreement(_Record):
    """One agreement coefficient for one variable, with what to read it against.

    ``value`` is ``None`` when the coefficient is not defined for these codes, and ``reason``
    then says why. ``units`` and ``pairable_values`` count what the figure is computed on.
    ``percent_agreement`` (0 to 100) and each coder's counts and shares of each code, over the
    same values, show how common each code is, so a high or low figure can be read against them.
    """

    coefficient: Literal["krippendorff_alpha", "cohen_kappa"]
    value: float | None
    reason: str | None = None
    units: int
    pairable_values: int
    percent_agreement: float | None
    coder_counts: dict[str, dict[str, int]]
    coder_shares: dict[str, dict[str, float]]


class VariableReliability(_Record):
    """How far the coders agree on one variable."""

    variable: str
    type: Literal["nominal", "ordinal"]
    coders: tuple[str, ...]
    alpha: Agreement
    codebook_hash: str
    reliacode_version: str


def variable_reliability(
    codebook: Codebook,
    codes: Codes,
    variable: str,
    *,
    coders: Sequence[str] | None = None,
) -> VariableReliability:
    """How far the coders agree on ``variable``: Krippendorff's alpha.

    The level of measurement comes from the codebook: nominal, or ordinal with the categories
    ranked by their levels, using Krippendorff's ordinal metric.

    Alpha uses every unit with values from at least two coders, so coders may leave units out
    and any number of coders can be compared. A unit with only one value cannot be paired and
    adds nothing. Missing values are left out, never counted as a category.

    Args:
        codebook: the codebook the units were coded with.
        codes: the codes to compare, from :func:`read_codes`.
        variable: the name of the variable.
        coders: the coders to compare, such as ``["anna", "model"]``. By default, every coder
            with codes for this variable.

    Raises:
        CodesError: a value is not one of the variable's codes.
        KeyError: the codebook has no variable with that name.
        TypeError, ValueError: the other arguments are not as described.
    """
    from reliacode import __version__  # here, because the package imports this module

    if not isinstance(codebook, Codebook):
        raise TypeError(
            'codebook should be a codebook loaded with reliacode.load_codebook("file.yaml").'
        )
    if not isinstance(codes, Codes):
        raise TypeError(
            "codes should be the codes to compare, from reliacode.read_codes(codebook, ...), but "
            f"they are {describe(codes)}."
        )
    data = _prepare(codebook.variable(variable), codes, coders)
    patterns = _Patterns(data)
    return VariableReliability(
        variable=data.variable.name,
        type=data.variable.type,
        coders=data.coders,
        alpha=_alpha_record(data, patterns),
        codebook_hash=codebook.content_hash(),
        reliacode_version=__version__,
    )


class _Data(NamedTuple):
    """One variable's codes, ready to compute on."""

    variable: Variable
    coders: tuple[str, ...]
    #: The codes in the order the metric uses: codebook order, or by level when ordinal.
    order: tuple[str, ...]
    #: For each unit, in the order of their ids, each coder's code (as a position in
    #: ``order``). Missing values are left out.
    units: list[dict[str, int]]


def _prepare(variable: Variable, codes: Codes, coders: Sequence[str] | None) -> _Data:
    chosen = _chosen_coders(variable.name, codes, coders)
    if variable.type == "ordinal":
        categories = sorted(variable.categories, key=lambda category: category.level or 0)
    else:
        categories = list(variable.categories)
    order = tuple(category.code for category in categories)
    position = {code: number for number, code in enumerate(order)}
    by_unit: dict[str, dict[str, int]] = {}
    problems = []
    for item in codes.values:
        if item.variable != variable.name or item.coder not in chosen or item.value is None:
            continue
        if item.value not in position:
            message = _unknown_code(item.value, variable.name, variable.codes)
            problems.append(Problem(message, _place(item, "")))
            continue
        by_unit.setdefault(item.unit, {})[item.coder] = position[item.value]
    if problems:
        raise CodesError(problems)
    units = [by_unit[unit] for unit in sorted(by_unit)]
    return _Data(variable, chosen, order, units)


def _chosen_coders(variable: str, codes: Codes, coders: Sequence[str] | None) -> tuple[str, ...]:
    if coders is None:
        return tuple(sorted({item.coder for item in codes.values if item.variable == variable}))
    names = [coders] if isinstance(coders, str) else list(coders)
    known = codes.coders
    for name in names:
        if name not in known:
            raise ValueError(
                f'There are no codes from the coder "{name}". The coders are: {", ".join(known)}.'
            )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'The coder "{repeated[0]}" is named more than once in coders.')
    if not names:
        raise ValueError("coders is empty. Name at least two, or leave it out to compare them all.")
    return tuple(sorted(names))


class _Patterns:
    """The pairable units of one variable (units with two or more values), grouped by how many
    values of each code they hold.

    Units with the same counts add the same amounts to every total, so each pattern's parts are
    worked out once, and a total is a sum of parts times how often each pattern occurs.
    """

    def __init__(self, data: _Data) -> None:
        size = len(data.order)
        index: dict[tuple[int, ...], int] = {}
        #: Each pattern: the count of each code.
        self.counts: list[tuple[int, ...]] = []
        #: For each pairable unit, in order, its pattern.
        self.of_unit: list[int] = []
        for unit in data.units:
            if len(unit) < 2:
                continue
            tally = [0] * size
            for code in unit.values():
                tally[code] += 1
            key = tuple(tally)
            if key not in index:
                index[key] = len(self.counts)
                self.counts.append(key)
            self.of_unit.append(index[key])
        self.size = size
        #: Each pattern's coincidences of a code with itself, summed over the codes.
        self.same = [_same(counts) for counts in self.counts]
        #: Each pattern's pairs of values, and the pairs that agree.
        self.pairs = [sum(counts) * (sum(counts) - 1) // 2 for counts in self.counts]
        self.agree = [sum(c * (c - 1) // 2 for c in counts) for counts in self.counts]
        #: Ordinal data only: each pattern's coincidences of two different codes c < k.
        self.ordinal = data.variable.type == "ordinal"
        self.mixed = [_mixed(counts) if self.ordinal else [] for counts in self.counts]

    def everyone(self) -> list[int]:
        """How often each pattern occurs among the units."""
        often = [0] * len(self.counts)
        for pattern in self.of_unit:
            often[pattern] += 1
        return often

    def totals(self, often: Sequence[int]) -> "_Totals":
        n = [0] * self.size
        same = 0.0
        pairs = agree = 0
        mixed: dict[tuple[int, int], float] = {}
        for pattern, times in enumerate(often):
            if not times:
                continue
            for code, count in enumerate(self.counts[pattern]):
                n[code] += times * count
            same += times * self.same[pattern]
            pairs += times * self.pairs[pattern]
            agree += times * self.agree[pattern]
            for cell, coincidences in self.mixed[pattern]:
                mixed[cell] = mixed.get(cell, 0.0) + times * coincidences
        return _Totals(n, same, pairs, agree, mixed, self.ordinal)


def _same(counts: tuple[int, ...]) -> float:
    values = sum(counts)
    return sum(c * (c - 1) for c in counts) / (values - 1)


def _mixed(counts: tuple[int, ...]) -> list[tuple[tuple[int, int], float]]:
    """The coincidences o_ck of two different codes c < k in one unit: n_c n_k / (m - 1)."""
    values = sum(counts)
    used = [(code, count) for code, count in enumerate(counts) if count]
    return [
        ((c, k), count_c * count_k / (values - 1))
        for position, (c, count_c) in enumerate(used)
        for k, count_k in used[position + 1 :]
    ]


class _Totals(NamedTuple):
    """Totals over a set of units: the number of pairable values of each code, the
    coincidences of codes with themselves, the pairs of values and the pairs that agree."""

    n: list[int]
    same: float
    pairs: int
    agree: int
    #: Ordinal data only: the coincidences o_ck of codes c < k.
    mixed: dict[tuple[int, int], float]
    ordinal: bool


def _alpha(totals: _Totals) -> float | None:
    """Krippendorff's alpha from the totals, or None when it is undefined: with fewer than two
    pairable values, or with every value the same code.

    Nominal: ((n - 1) sum_c o_cc - sum_c n_c (n_c - 1)) / (n (n - 1) - sum_c n_c (n_c - 1)).
    Ordinal: 1 - (n - 1) sum_{c<k} o_ck d_ck / sum_{c<k} n_c n_k d_ck, where the squared
    difference d_ck = (n_c + ... + n_k - (n_c + n_k) / 2) ** 2 counts the values ranked from c to
    k, so it is worked out from these totals' own n (Krippendorff, 2011).
    """
    n = sum(totals.n)
    if n < 2 or sum(1 for count in totals.n if count) < 2:
        return None
    if not totals.ordinal:
        chance = sum(count * (count - 1) for count in totals.n)
        return ((n - 1) * totals.same - chance) / (n * (n - 1) - chance)
    below = [0]
    for count in totals.n:
        below.append(below[-1] + count)

    def difference(c: int, k: int) -> float:
        return (below[k + 1] - below[c] - (totals.n[c] + totals.n[k]) / 2) ** 2

    observed = sum(o * difference(c, k) for (c, k), o in totals.mixed.items())
    used = [code for code, count in enumerate(totals.n) if count]
    expected = sum(
        totals.n[c] * totals.n[k] * difference(c, k)
        for position, c in enumerate(used)
        for k in used[position + 1 :]
    )
    return 1 - (n - 1) * observed / expected


def _alpha_record(data: _Data, patterns: _Patterns) -> Agreement:
    totals = patterns.totals(patterns.everyone())
    value = _alpha(totals)
    reason = None
    if value is None:
        n = sum(totals.n)
        if n < 2:
            reason = (
                "No alpha: no unit has values from two or more coders, so there are no pairs of "
                "values to compare."
            )
        else:
            only = data.order[next(code for code, count in enumerate(totals.n) if count)]
            reason = (
                f'No alpha: every pairable value is "{only}", so there is no variation. Alpha is '
                "undefined here (the disagreement expected by chance is zero), not 0 or 1."
            )
    pairable = [unit for unit in data.units if len(unit) >= 2]
    counts = _coder_counts(data, pairable)
    return Agreement(
        coefficient="krippendorff_alpha",
        value=value,
        reason=reason,
        units=len(patterns.of_unit),
        pairable_values=sum(totals.n),
        percent_agreement=100 * totals.agree / totals.pairs if totals.pairs else None,
        coder_counts=counts,
        coder_shares=_shares(counts),
    )


def _coder_counts(data: _Data, units: list[dict[str, int]]) -> dict[str, dict[str, int]]:
    """Each coder's count of each code, in codebook order, over the given units. A coder with
    no value in them is left out."""
    tallies: dict[str, Counter[str]] = {}
    for unit in units:
        for coder, code in unit.items():
            tallies.setdefault(coder, Counter())[data.order[code]] += 1
    return {
        coder: {code: tallies[coder][code] for code in data.variable.codes}
        for coder in data.coders
        if coder in tallies
    }


def _shares(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    return {
        coder: {code: count / sum(tally.values()) for code, count in tally.items()}
        for coder, tally in counts.items()
    }

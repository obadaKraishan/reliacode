"""Make the reference values for the reliability tests with other implementations.

reliacode depends on neither package, so this script is run by hand, outside the project, and
its output is kept in ``tests/fixtures/reliability_reference.json``:

    uv run --no-project --with krippendorff --with scikit-learn \\
        python tests/reference/make_reliability_reference.py

It invents fixed tables of codes (seeded, so the same tables every time) and records:

- Krippendorff's alpha from the ``krippendorff`` package, nominal and ordinal, on tables with
  2 to 5 coders and missing values;
- Cohen's kappa from scikit-learn's ``cohen_kappa_score`` on two-coder tables.

The tests check reliacode against these values to within 1e-9.
"""

import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import krippendorff
import numpy as np
from sklearn.metrics import cohen_kappa_score

SEED = 20261124
OUTPUT = Path(__file__).parent.parent / "fixtures" / "reliability_reference.json"


def invented_codes(rng: random.Random, codes: list[str], units: int) -> list[str]:
    return [rng.choice(codes) for _ in range(units)]


def coder_values(
    rng: random.Random, truth: list[str], codes: list[str], skill: float, missing: float
) -> list[str | None]:
    return [
        None if rng.random() < missing else (code if rng.random() < skill else rng.choice(codes))
        for code in truth
    ]


def alpha_tables(rng: random.Random) -> list[dict[str, object]]:
    tables = []
    for number in range(24):
        level = "nominal" if number % 2 == 0 else "ordinal"
        codes = [f"c{n}" for n in range(rng.randint(2, 6))]
        truth = invented_codes(rng, codes, rng.randint(10, 60))
        skill, missing = rng.uniform(0.3, 0.95), rng.uniform(0.0, 0.3)
        matrix = {
            f"coder{c}": coder_values(rng, truth, codes, skill, missing)
            for c in range(rng.randint(2, 5))
        }
        position = {code: float(n) for n, code in enumerate(codes)}
        data = np.array(
            [[np.nan if v is None else position[v] for v in values] for values in matrix.values()]
        )
        value = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement=level,
            value_domain=[float(n) for n in range(len(codes))],
        )
        tables.append({"level": level, "codes": codes, "matrix": matrix, "alpha": float(value)})
    return tables


def kappa_tables(rng: random.Random) -> list[dict[str, object]]:
    tables = []
    for _ in range(12):
        codes = [f"c{n}" for n in range(rng.randint(2, 6))]
        truth = invented_codes(rng, codes, rng.randint(10, 80))
        skill = rng.uniform(0.3, 0.95)
        first = coder_values(rng, truth, codes, skill, 0.0)
        second = coder_values(rng, truth, codes, skill, 0.0)
        value = cohen_kappa_score(first, second)
        tables.append({"codes": codes, "first": first, "second": second, "kappa": float(value)})
    return tables


def main() -> None:
    rng = random.Random(SEED)
    reference = {
        "about": (
            "Reference values from other implementations, made by "
            "tests/reference/make_reliability_reference.py. Do not edit by hand."
        ),
        "made_with": {
            "python": platform.python_version(),
            "krippendorff": version("krippendorff"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "seed": SEED,
        "alpha": alpha_tables(rng),
        "kappa": kappa_tables(rng),
    }
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

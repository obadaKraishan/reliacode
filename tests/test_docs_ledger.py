"""The sample line in docs/ledger.md is a valid ledger line, written in the order reliacode
writes its fields."""

import json
import re
from pathlib import Path

from reliacode import LedgerEntry
from reliacode.ledger import LEDGER_VERSION

DOC = (Path(__file__).parent.parent / "docs" / "ledger.md").read_text(encoding="utf-8")


def test_the_sample_line_is_a_current_ledger_line() -> None:
    match = re.search(r"```json\n(\{.*?\})\n```", DOC, flags=re.DOTALL)
    assert match is not None
    data = json.loads(match.group(1))
    entry = LedgerEntry.model_validate(data)
    assert entry.schema_version == LEDGER_VERSION
    assert list(entry.model_dump(mode="json")) == list(data)
    assert entry.parameters is not None
    assert list(entry.parameters.model_dump()) == list(data["parameters"])

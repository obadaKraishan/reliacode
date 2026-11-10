"""Shared test helpers: paths and small YAML snippets."""

import textwrap
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples" / "codebooks"


def yaml_text(text: str) -> str:
    """Dedent a triple-quoted YAML snippet so tests can indent it naturally."""
    return textwrap.dedent(text).lstrip("\n")


MINIMAL = yaml_text(
    """
    name: tone-demo
    variables:
      - name: tone
        definition: The overall tone of the text toward its main subject.
        categories:
          - code: positive
            definition: The text mostly praises or supports its subject.
          - code: negative
            definition: The text mostly criticises or opposes its subject.
    """
)

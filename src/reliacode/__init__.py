"""reliacode: validated LLM content analysis."""

from importlib.metadata import PackageNotFoundError, version

from reliacode.codebook import (
    Category,
    Codebook,
    Example,
    Variable,
    load_codebook,
    parse_codebook,
)
from reliacode.errors import CodebookError, Problem

try:
    __version__ = version("reliacode")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "Category",
    "Codebook",
    "CodebookError",
    "Example",
    "Problem",
    "Variable",
    "__version__",
    "load_codebook",
    "parse_codebook",
]

"""reliacode: validated LLM content analysis."""

from importlib.metadata import PackageNotFoundError, version

from reliacode.backends import (
    Backend,
    BackendError,
    FakeBackend,
    OpenAICompatibleBackend,
    RequestParameters,
    Response,
)
from reliacode.codebook import (
    Category,
    Codebook,
    Example,
    Variable,
    load_codebook,
    parse_codebook,
)
from reliacode.coding import CodingResult, CodingRun, Rule, code
from reliacode.errors import CodebookError, Problem
from reliacode.ledger import (
    Budget,
    BudgetError,
    Ledger,
    LedgerEntry,
    LedgerError,
    LedgerSummary,
    ModelPrice,
    PriceTable,
    read_ledger,
    summarize,
)
from reliacode.reliability import (
    CodebookReliability,
    Codes,
    CodesError,
    VariableReliability,
    codebook_reliability,
    read_codes,
    variable_reliability,
)

try:
    __version__ = version("reliacode")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "Backend",
    "BackendError",
    "Budget",
    "BudgetError",
    "Category",
    "Codebook",
    "CodebookError",
    "CodebookReliability",
    "Codes",
    "CodesError",
    "CodingResult",
    "CodingRun",
    "Example",
    "FakeBackend",
    "Ledger",
    "LedgerEntry",
    "LedgerError",
    "LedgerSummary",
    "ModelPrice",
    "OpenAICompatibleBackend",
    "PriceTable",
    "Problem",
    "RequestParameters",
    "Response",
    "Rule",
    "Variable",
    "VariableReliability",
    "__version__",
    "code",
    "codebook_reliability",
    "load_codebook",
    "parse_codebook",
    "read_codes",
    "read_ledger",
    "summarize",
    "variable_reliability",
]

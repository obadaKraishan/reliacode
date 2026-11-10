import pytest

from helpers import MINIMAL


@pytest.fixture
def minimal_yaml() -> str:
    return MINIMAL

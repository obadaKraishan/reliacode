import socket

import pytest

from helpers import MINIMAL, NetworkBlockedError


def _blocked(*args: object, **kwargs: object) -> None:
    raise NetworkBlockedError("Tests must not use the network. Answer with a mock transport.")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs with sockets blocked, so no test can reach a real server."""
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture
def minimal_yaml() -> str:
    return MINIMAL

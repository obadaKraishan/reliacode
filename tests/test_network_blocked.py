"""The test suite cannot reach the network: every test runs with sockets blocked."""

import socket

import pytest

from helpers import NetworkBlockedError


def test_a_socket_cannot_connect() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex(("127.0.0.1", 9))


def test_a_connection_cannot_be_opened() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=1)


def test_names_cannot_be_looked_up() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.org", 443)


def test_the_block_is_not_an_os_error() -> None:
    # Client code turns OSError into "no server at the address"; the block must not look like that.
    assert not issubclass(NetworkBlockedError, OSError)

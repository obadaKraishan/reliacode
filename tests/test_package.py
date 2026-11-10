import reliacode


def test_version_is_a_development_version() -> None:
    assert reliacode.__version__.startswith("0.1.0")

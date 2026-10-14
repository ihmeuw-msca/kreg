from importlib.metadata import version

import kreg


def test_version() -> None:
    assert kreg.__version__ == version("kreg")


def test_exports() -> None:
    for name in kreg.__all__:
        assert hasattr(kreg, name)

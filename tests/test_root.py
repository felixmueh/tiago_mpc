"""The package root re-exports what __all__ lists."""
import tiago_mpc as tm


def test_all_names_are_exported():
    assert all(hasattr(tm, name) for name in tm.__all__)

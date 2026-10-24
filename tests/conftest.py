"""Shared test setup: each test starts without a stop request."""
import pytest

from tiago_mpc.loops import STOP


@pytest.fixture(autouse=True)
def clear_stop():
    """The stop flag belongs to one launch; a test that sets it leaves no trace."""
    yield
    STOP.clear()

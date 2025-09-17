# tests/test_slow_example.py
import time

import pytest


@pytest.mark.slow
def test_slow():
    time.sleep(2)
    assert True

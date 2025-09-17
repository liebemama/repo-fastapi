# tests/test_dummy_live.py
import os

import pytest
import requests


BASE_URL = os.getenv("NEUROSERVE_URL", "http://127.0.0.1:8000")


@pytest.mark.integration
def test_dummy_ping_live():
    """Integration test: requires a running NeuroServe at BASE_URL."""
    try:
        r = requests.post(f"{BASE_URL}/plugins/dummy/ping", json={"hello": "world"}, timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"NeuroServe not running at {BASE_URL}")
    assert r.status_code == 200

    data = r.json()
    assert data["plugin"] == "dummy"
    assert data["result"]["task"] == "ping"
    assert data["result"]["payload_received"]["hello"] == "world"

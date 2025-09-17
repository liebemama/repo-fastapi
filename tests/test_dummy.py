# tests/test_dummy.py
from fastapi.testclient import TestClient

from app.main import app


def test_dummy_ping_inprocess():
    client = TestClient(app)  # no real server, same process
    resp = client.post("/plugins/dummy/ping", json={"hello": "world"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["plugin"] == "dummy"
    assert data["result"]["task"] == "ping"
    assert data["result"]["payload_received"]["hello"] == "world"

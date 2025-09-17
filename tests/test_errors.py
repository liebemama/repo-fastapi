from __future__ import annotations

from typing import Annotated

import pytest
from fastapi import Body, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.main import app


@pytest.fixture
def client_no_raise():
    # TestClient that does NOT raise server exceptions, so we can assert 500 responses
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# NOTE: Add test routes once only
if not getattr(app.state, "_test_routes_added", False):

    class Item(BaseModel):
        name: str = Field(..., min_length=2)
        qty: int = Field(..., ge=1)
        note: str | None = None

    @app.get("/_boom")
    def _boom():
        """Simulates a 500 Internal Server Error."""
        raise RuntimeError("Kaboom")

    @app.post("/_validate")
    def _validate(item: Item):
        """Validates input; returns 422 if invalid."""
        return {"ok": True, "item": item.model_dump()}

    @app.get("/_unauth")
    def _unauth():
        """Simulates a 401 Unauthorized error."""
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    @app.get("/_forbidden")
    def _forbidden():
        """Simulates a 403 Forbidden error."""
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    @app.get("/_timeout")
    def _timeout():
        """Simulates a 408 Request Timeout error."""
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Timeout")

    @app.post("/_payload-too-large")
    def _payload_too_large(payload: Annotated[dict, Body(...)]):
        """Simulates a 413 Payload Too Large error."""
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Too large",
        )

    @app.post("/_unsupported-media")
    def _unsupported_media():
        """Simulates a 415 Unsupported Media Type error."""
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported")

    @app.get("/_too-many")
    def _too_many():
        """Simulates a 429 Too Many Requests error."""
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many")

    @app.get("/_not-impl")
    def _not_impl():
        """Simulates a 501 Not Implemented error."""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

    @app.get("/_unavailable")
    def _unavailable():
        """Simulates a 503 Service Unavailable error."""
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unavailable")

    app.state._test_routes_added = True

client = TestClient(app)


# --------- Basic 404 / 405 Tests ---------
def test_404_json():
    r = client.get("/__nope__", headers={"Accept": "application/json"})
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == 404 and body["message"]
    assert body["path"] == "/__nope__"
    assert body["method"] == "GET"
    assert "x-request-id" in {k.lower(): v for k, v in r.headers.items()}


def test_404_html():
    r = client.get("/__nope__?format=html", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"].lower()
    assert "<html" in r.text.lower()
    assert "x-request-id" in {k.lower(): v for k, v in r.headers.items()}


def test_405_method_not_allowed():
    r = client.post("/health", headers={"Accept": "application/json"})
    assert r.status_code == 405
    body = r.json()
    assert body["code"] == 405 and ("Method" in body["message"] or body["message"])
    assert body["method"] == "POST"


# --------- Authentication/Authorization ---------
def test_401_unauthorized():
    r = client.get("/_unauth", headers={"Accept": "application/json"})
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == 401 and body["message"]


def test_403_forbidden():
    r = client.get("/_forbidden", headers={"Accept": "application/json"})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == 403 and body["message"]


# --------- Payloads/Media/Rate Limits ---------
def test_413_payload_too_large():
    r = client.post("/_payload-too-large", json={"x": "y"})
    assert r.status_code == 413
    body = r.json()
    assert body["code"] == 413 and body["message"]


def test_415_unsupported_media_type():
    r = client.post("/_unsupported-media", content=b"abc", headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    body = r.json()
    assert body["code"] == 415 and body["message"]


def test_429_too_many_requests():
    r = client.get("/_too-many", headers={"Accept": "application/json"})
    assert r.status_code == 429
    body = r.json()
    assert body["code"] == 429 and body["message"]


# --------- 422 Validation Errors ---------
def test_422_validation_error_json():
    r = client.post("/_validate", json={"name": "A"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == 422 and body["message"] == "Validation error"
    assert isinstance(body["details"], list) and body["details"]


def test_422_validation_error_html():
    r = client.post("/_validate?format=html", json={"name": ""})
    assert r.status_code == 422
    assert "text/html" in r.headers["content-type"].lower()
    assert "<html" in r.text.lower()


# --------- 500/501/503/408 Errors ---------
def test_500_internal_server_error(client_no_raise):
    r = client_no_raise.get("/_boom", headers={"Accept": "application/json"})
    assert r.status_code == 500


def test_501_not_implemented():
    r = client.get("/_not-impl", headers={"Accept": "application/json"})
    assert r.status_code == 501
    body = r.json()
    assert body["code"] == 501


def test_503_service_unavailable():
    r = client.get("/_unavailable", headers={"Accept": "application/json"})
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == 503


def test_408_request_timeout():
    r = client.get("/_timeout", headers={"Accept": "application/json"})
    assert r.status_code == 408
    body = r.json()
    assert body["code"] == 408


# --------- HTML/JSON Negotiation + Headers ---------
def test_html_negotiation_by_accept_header():
    r = client.get("/_forbidden", headers={"Accept": "text/html"})
    assert r.status_code == 403
    assert "text/html" in r.headers["content-type"].lower()


def test_request_id_present_on_success_and_errors():
    r1 = client.get("/health")
    r2 = client.get("/__nope__")
    for r in (r1, r2):
        assert "x-request-id" in {k.lower(): v for k, v in r.headers.items()}

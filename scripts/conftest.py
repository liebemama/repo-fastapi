# scripts/conftest.py
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Shared FastAPI TestClient for all tests."""
    return TestClient(app)

import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def test_client():
    """
    Provides a test client for the FastAPI application.
    """
    with TestClient(app) as client:
        yield client


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip gpu_cuda/gpu_mps tests if the hardware is not available.
    """
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    mps_available = bool(torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_mps = pytest.mark.skip(reason="MPS not available")

    for item in items:
        if "gpu_cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)
        if "gpu_mps" in item.keywords and not mps_available:
            item.add_marker(skip_mps)


def pytest_addoption(parser):
    """Add custom command-line option to enable running slow tests."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


@pytest.fixture(autouse=True)
def _skip_slow(request):
    """Automatically skip tests marked as 'slow' unless --run-slow is specified."""
    if request.node.get_closest_marker("slow") and not request.config.getoption("--run-slow"):
        pytest.skip("use --run-slow to run this test")

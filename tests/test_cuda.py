# tests/test_cuda.py
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.gpu_cuda]


def test_cuda_tensor():
    import torch

    x = torch.ones(2, device="cuda")
    assert x.is_cuda

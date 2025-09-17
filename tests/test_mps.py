# tests/test_mps.py
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.gpu_mps]


def test_mps_tensor():
    import torch

    x = torch.ones(2, device="mps")
    assert x.device.type == "mps"

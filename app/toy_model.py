from __future__ import annotations

import torch
import torch.nn as nn

from .runtime import pick_device


class TinyNet(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.

    Args:
        in_features (int): Number of input features. Default is 512.
        hidden (int): Number of hidden units. Default is 1024.
        out_features (int): Number of output features. Default is 10.
    """

    def __init__(self, in_features: int = 512, hidden: int = 1024, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.net(x)


def load_model() -> tuple[TinyNet, torch.device]:
    """
    Instantiate and prepare the TinyNet model for inference.

    Returns:
        tuple[TinyNet, torch.device]: The initialized model and the device it resides on.
    """
    dev = pick_device()
    model = TinyNet().to(dev).eval()

    # Warm-up forward pass to initialize weights on device
    with torch.no_grad():
        _ = model(torch.randn(1, 512, device=dev))

    return model, dev

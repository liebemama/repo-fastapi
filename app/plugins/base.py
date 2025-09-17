from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AIPlugin(ABC):
    """
    Standard interface for any AI model plugin.

    Attributes:
        name (str): Name of the provider (usually matches the folder name).
        tasks (List[str]): List of supported tasks such as "infer", "embed", "classify-image".
    """

    name: str = "unknown"
    tasks: list[str] = []

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and required resources into memory (executed once).
        """
        ...

    @abstractmethod
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        General inference entry point. The plugin determines how to interpret the payload.

        Args:
            payload (Dict[str, Any]): Input data for inference.

        Returns:
            Dict[str, Any]: Inference result.
        """
        ...

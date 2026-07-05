"""Base class for model downloaders."""

from abc import ABC, abstractmethod
from pathlib import Path

from models.download_progress import ProgressCallback


class BaseDownloader(ABC):
    """Abstract base class for engine-specific model downloaders."""

    @abstractmethod
    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """
        Download a model.

        Args:
            model_id: ID of the model to download.
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to the downloaded model.
        """
        pass

    @abstractmethod
    def get_model_size(self, model_id: str) -> int:
        """
        Get the size of a model in bytes.

        Args:
            model_id: ID of the model.

        Returns:
            Size in bytes, or 0 if unknown.
        """
        pass

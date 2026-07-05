"""Download progress tracking."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass
class DownloadProgress:
    """Progress information for model downloads."""

    downloaded_bytes: int
    """Number of bytes downloaded so far."""

    total_bytes: int
    """Total size in bytes (0 if unknown)."""

    speed_bytes_per_sec: float
    """Current download speed in bytes per second."""

    eta_seconds: float
    """Estimated time remaining in seconds (-1 if unknown)."""

    model_id: str = ""
    """ID of the model being downloaded."""

    status: str = "downloading"
    """Current status: 'downloading', 'extracting', 'completed', 'error'."""

    @property
    def percentage(self) -> float:
        """Get download percentage (0-100)."""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)

    @property
    def speed_mb_per_sec(self) -> float:
        """Get download speed in MB/s."""
        return self.speed_bytes_per_sec / (1024 * 1024)

    @property
    def downloaded_mb(self) -> float:
        """Get downloaded size in MB."""
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        """Get total size in MB."""
        return self.total_bytes / (1024 * 1024)

    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        if self.eta_seconds < 0:
            return "unknown"
        if self.eta_seconds < 60:
            return f"{int(self.eta_seconds)}s"
        if self.eta_seconds < 3600:
            mins = int(self.eta_seconds // 60)
            secs = int(self.eta_seconds % 60)
            return f"{mins}m {secs}s"
        hours = int(self.eta_seconds // 3600)
        mins = int((self.eta_seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# Type alias for progress callback function
ProgressCallback = Callable[[DownloadProgress], None]


class ProgressReporter(Protocol):
    """Protocol for progress reporting."""

    def report(self, progress: DownloadProgress) -> None:
        """Report download progress."""
        ...

    def complete(self, model_id: str) -> None:
        """Mark download as complete."""
        ...

    def error(self, model_id: str, message: str) -> None:
        """Report download error."""
        ...

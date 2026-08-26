"""Download progress tracking."""

import time
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


class HuggingFaceProgressCallback:
    """Progress callback wrapper for HuggingFace downloads."""

    def __init__(
        self,
        model_id: str,
        total_size: int,
        callback: ProgressCallback,
        start_time: float,
    ):
        self.model_id = model_id
        self.total_size = total_size
        self.callback = callback
        self.start_time = start_time
        self.last_update = 0

    def __call__(self, current: int, total: int):
        """Called during download."""
        # Throttle updates
        current_time = time.time()
        if current_time - self.last_update < 0.1 and current < total:
            return
        self.last_update = current_time

        actual_total = total if total > 0 else self.total_size
        elapsed = current_time - self.start_time
        speed = current / elapsed if elapsed > 0 else 0
        remaining = actual_total - current
        eta = remaining / speed if speed > 0 else -1

        self.callback(
            DownloadProgress(
                downloaded_bytes=current,
                total_bytes=actual_total,
                speed_bytes_per_sec=speed,
                eta_seconds=eta,
                model_id=self.model_id,
                status="downloading",
            )
        )


def make_hf_tqdm_class(hf_callback):
    """Build a ``tqdm`` subclass forwarding snapshot_download updates.

    ``huggingface_hub.snapshot_download`` accepts ``tqdm_class``; the returned
    class reports every bar's cumulative bytes through ``hf_callback(current,
    total)`` so streamed progress reaches the user callback. Returns ``None``
    when tqdm is unavailable, degrading to start/end-only progress.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return None

    class _ForwardingTqdm(tqdm):
        """tqdm bridge mirroring each bar's state into the HF callback."""

        def update(self, n=1):
            shown = super().update(n)
            total = int(self.total) if self.total else 0
            hf_callback(int(self.n), total)
            return shown

    return _ForwardingTqdm

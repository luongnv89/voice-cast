"""Chatterbox TTS model downloader with progress reporting."""

import logging
import time
from pathlib import Path

from models.download_progress import DownloadProgress, ProgressCallback
from models.downloaders.base import BaseDownloader
from models.model_registry import get_registry

logger = logging.getLogger("voice_cloner.models.chatterbox")


class ChatterboxDownloader(BaseDownloader):
    """Downloader for Chatterbox TTS models."""

    # HuggingFace repo IDs
    MODEL_REPOS = {
        "chatterbox-turbo": "ResembleAI/chatterbox",
        "chatterbox-standard": "ResembleAI/chatterbox",
    }

    # Approximate sizes in bytes
    MODEL_SIZES = {
        "chatterbox-turbo": 350 * 1024 * 1024,  # ~350 MB
        "chatterbox-standard": 500 * 1024 * 1024,  # ~500 MB
    }

    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Download a Chatterbox model from HuggingFace."""
        if model_id not in self.MODEL_REPOS:
            raise ValueError(f"Unknown Chatterbox model: {model_id}")

        repo_id = self.MODEL_REPOS[model_id]
        total_size = self.get_model_size(model_id)

        # Report starting
        if progress_callback:
            progress_callback(
                DownloadProgress(
                    downloaded_bytes=0,
                    total_bytes=total_size,
                    speed_bytes_per_sec=0,
                    eta_seconds=-1,
                    model_id=model_id,
                    status="downloading",
                )
            )

        start_time = time.time()

        try:
            # Use huggingface_hub for downloading with progress
            from huggingface_hub import snapshot_download

            # Download the model into the same cache location the registry checks.
            cache_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(get_registry().get_cache_dir("chatterbox")),
            )

            # Calculate final progress
            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        downloaded_bytes=total_size,
                        total_bytes=total_size,
                        speed_bytes_per_sec=total_size / elapsed if elapsed > 0 else 0,
                        eta_seconds=0,
                        model_id=model_id,
                        status="completed",
                    )
                )

            return Path(cache_dir)

        except ImportError as e:
            # If huggingface_hub not available, try via chatterbox directly
            logger.warning("huggingface_hub not available, trying direct chatterbox import")
            return self._download_via_chatterbox(model_id, progress_callback, start_time, total_size)

    def _download_via_chatterbox(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None,
        start_time: float,
        total_size: int,
    ) -> Path:
        """Download by importing chatterbox which triggers auto-download."""
        try:
            from chatterbox.tts import ChatterboxTTS

            # This will trigger download if not cached
            # We can't get progress from this method unfortunately
            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        downloaded_bytes=0,
                        total_bytes=total_size,
                        speed_bytes_per_sec=0,
                        eta_seconds=-1,
                        model_id=model_id,
                        status="downloading",
                    )
                )

            # Load the model (this downloads if needed)
            _ = ChatterboxTTS.from_pretrained(device="cpu")

            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        downloaded_bytes=total_size,
                        total_bytes=total_size,
                        speed_bytes_per_sec=total_size / elapsed if elapsed > 0 else 0,
                        eta_seconds=0,
                        model_id=model_id,
                        status="completed",
                    )
                )

            # Return the HF cache path
            return get_registry().get_cache_dir("chatterbox") / "models--ResembleAI--chatterbox"

        except ImportError as e:
            logger.error("chatterbox-tts package not installed. Install with: pip install chatterbox-tts")
            raise ImportError("chatterbox-tts package required. Install with: pip install chatterbox-tts") from e

    def get_model_size(self, model_id: str) -> int:
        """Get approximate model size in bytes."""
        return self.MODEL_SIZES.get(model_id, 0)


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

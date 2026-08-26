"""Chatterbox TTS model downloader with progress reporting."""

import logging
import time
from pathlib import Path

from models.download_progress import (
    DownloadProgress,
    HuggingFaceProgressCallback,
    ProgressCallback,
    make_hf_tqdm_class,
)
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

    # Pinned Hugging Face revisions for safe, reproducible downloads.
    MODEL_REVISIONS = {
        "chatterbox-turbo": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        "chatterbox-standard": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
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

            download_kwargs: dict = {
                "repo_id": repo_id,
                "revision": self.MODEL_REVISIONS[model_id],
                "cache_dir": str(get_registry().get_cache_dir("chatterbox")),
            }
            if progress_callback:
                hf_cb = HuggingFaceProgressCallback(model_id, total_size, progress_callback, start_time)
                tqdm_class = make_hf_tqdm_class(hf_cb)
                if tqdm_class is not None:
                    download_kwargs["tqdm_class"] = tqdm_class

            # Download the model into the same cache location the registry checks.
            cache_dir = snapshot_download(**download_kwargs)

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

        except ImportError:
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

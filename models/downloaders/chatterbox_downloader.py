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

# Variant/model-id → backend (module, class). Turbo (350M) and Standard (500M)
# ship as separate checkpoints loaded by separate backend classes; this map is
# the single source of truth shared by the engine loader and download fallback.
CHATTERBOX_VARIANT_BACKENDS = {
    "chatterbox-turbo": ("chatterbox.tts_turbo", "ChatterboxTurboTTS"),
    "chatterbox-standard": ("chatterbox.tts", "ChatterboxTTS"),
}


class ChatterboxDownloader(BaseDownloader):
    """Downloader for Chatterbox TTS models."""

    # HuggingFace repo IDs — one distinct repo per variant.
    MODEL_REPOS = {
        "chatterbox-turbo": "ResembleAI/chatterbox-turbo",
        "chatterbox-standard": "ResembleAI/chatterbox",
    }

    # Pinned Hugging Face revisions for safe, reproducible downloads.
    MODEL_REVISIONS = {
        "chatterbox-turbo": "749d1c1a46eb10492095d68fbcf55691ccf137cd",
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

            extra_kwargs: dict = {}
            if progress_callback:
                hf_cb = HuggingFaceProgressCallback(model_id, total_size, progress_callback, start_time)
                tqdm_class = make_hf_tqdm_class(hf_cb)
                if tqdm_class is not None:
                    extra_kwargs["tqdm_class"] = tqdm_class

            # Download the model into the same cache location the registry checks.
            cache_dir = snapshot_download(
                repo_id=repo_id,
                revision=self.MODEL_REVISIONS[model_id],
                cache_dir=str(get_registry().get_cache_dir("chatterbox")),
                **extra_kwargs,
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
            import importlib

            module_name, class_name = CHATTERBOX_VARIANT_BACKENDS[model_id]
            backend_cls = getattr(importlib.import_module(module_name), class_name)

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
            _ = backend_cls.from_pretrained(device="cpu")

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
            repo_id = self.MODEL_REPOS[model_id]
            return get_registry().get_cache_dir("chatterbox") / f"models--{repo_id.replace('/', '--')}"

        except ImportError as e:
            logger.error("chatterbox-tts package not installed. Install with: pip install chatterbox-tts")
            raise ImportError("chatterbox-tts package required. Install with: pip install chatterbox-tts") from e

    def get_model_size(self, model_id: str) -> int:
        """Get approximate model size in bytes."""
        return self.MODEL_SIZES.get(model_id, 0)

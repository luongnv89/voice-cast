"""Audio8 TTS model downloader with progress reporting."""

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

logger = logging.getLogger("voice_cloner.models.audio8")


class Audio8Downloader(BaseDownloader):
    """Downloader for Audio8 TTS ONNX model from Hugging Face."""

    MODEL_REPOS = {
        "audio8-tts": "audio8/audio8-TTS-0-1B-ONNX-INT8",
    }

    MODEL_REVISIONS = {
        "audio8-tts": None,  # Use latest
    }

    MODEL_SIZES = {
        "audio8-tts": 2000 * 1024 * 1024,  # ~2 GB
    }

    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Download an Audio8 TTS model from Hugging Face."""
        if model_id not in self.MODEL_REPOS:
            raise ValueError(f"Unknown Audio8 model: {model_id}")

        repo_id = self.MODEL_REPOS[model_id]
        total_size = self.get_model_size(model_id)
        cache_dir = get_registry().get_cache_dir("audio8-onnx")
        start_time = time.time()

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

        try:
            from huggingface_hub import snapshot_download

            extra_kwargs: dict = {}
            if progress_callback:
                hf_cb = HuggingFaceProgressCallback(model_id, total_size, progress_callback, start_time)
                tqdm_class = make_hf_tqdm_class(hf_cb)
                if tqdm_class is not None:
                    extra_kwargs["tqdm_class"] = tqdm_class

            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                **extra_kwargs,
            )
        except ImportError as e:
            logger.error("huggingface_hub package not installed. Install with: pip install huggingface_hub")
            raise ImportError("huggingface_hub package required. Install with: pip install huggingface_hub") from e

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

        return Path(path)

    def get_model_size(self, model_id: str) -> int:
        """Get approximate model size in bytes."""
        return self.MODEL_SIZES.get(model_id, 0)

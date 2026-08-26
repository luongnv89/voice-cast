"""MLX Audio model downloader with progress reporting."""

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

logger = logging.getLogger("voice_cloner.models.mlx")


class MlxDownloader(BaseDownloader):
    """Downloader for MLX Audio models hosted on Hugging Face."""

    MODEL_REPOS = {
        "mlx-kokoro": "mlx-community/Kokoro-82M-bf16",
        "mlx-csm": "mlx-community/csm-1b",
    }

    MODEL_REVISIONS = {
        "mlx-kokoro": "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
        "mlx-csm": "5bf5ec118cf45fecc7b51198fd9f1a20a5aab65a",
    }

    MODEL_SIZES = {
        "mlx-kokoro": 164 * 1024 * 1024,
        "mlx-csm": 2000 * 1024 * 1024,
    }

    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Download an MLX Audio model from Hugging Face."""
        if model_id not in self.MODEL_REPOS:
            raise ValueError(f"Unknown MLX Audio model: {model_id}")

        repo_id = self.MODEL_REPOS[model_id]
        total_size = self.get_model_size(model_id)
        cache_dir = get_registry().get_cache_dir("mlx-audio")
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
                revision=self.MODEL_REVISIONS[model_id],
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

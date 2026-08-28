"""Coqui TTS model downloader with progress reporting."""

import logging
import threading
import time
from pathlib import Path

from models.download_progress import DownloadProgress, ProgressCallback
from models.downloaders.base import BaseDownloader
from models.model_registry import get_registry

logger = logging.getLogger("voice_cloner.models.coqui")

# TTS manage.py streams files in 1 KiB chunks (iter_content(1024)); its
# module-global ``tqdm`` is the only injection point for transfer progress.
_TQDM_BLOCK_SIZE = 1024
_TQDM_PATCH_LOCK = threading.Lock()


def _make_forwarding_tqdm(base_tqdm, reporthook):
    """Wrap TTS's tqdm so bar updates feed a urllib-style reporthook."""

    class _ForwardingTqdm(base_tqdm):
        def update(self, n=1):
            shown = super().update(n)
            total = int(self.total) if self.total else 0
            reporthook(int(self.n) // _TQDM_BLOCK_SIZE, _TQDM_BLOCK_SIZE, total)
            return shown

    return _ForwardingTqdm


def _download_model_with_progress(
    model_name: str, cache_dir: Path, total_size: int, model_id: str, progress_callback: ProgressCallback
) -> tuple:
    """Run ModelManager.download_model, streaming progress when possible.

    Patches the ``tqdm`` symbol inside ``TTS.utils.manage`` for the duration
    of the call (restored in ``finally``) and constructs the manager with
    ``progress_bar=True`` so upstream reports each chunk through it. Falls
    back to a plain call when no callback is given or the seam is unusable.
    """
    from TTS.utils import manage as tts_manage
    from TTS.utils.manage import ModelManager

    def build_manager(progress_bar):
        try:
            return ModelManager(output_prefix=str(cache_dir), progress_bar=progress_bar)
        except TypeError:
            try:
                return ModelManager(output_prefix=str(cache_dir))
            except TypeError:
                return ModelManager()

    base_tqdm = getattr(tts_manage, "tqdm", None)
    if progress_callback is None or not isinstance(base_tqdm, type):
        manager = build_manager(False)
        return manager.download_model(model_name)

    coqui_cb = CoquiProgressCallback(model_id, total_size, progress_callback)
    forwarding = _make_forwarding_tqdm(base_tqdm, coqui_cb)
    with _TQDM_PATCH_LOCK:
        original = tts_manage.tqdm
        tts_manage.tqdm = forwarding
        try:
            manager = build_manager(True)
            return manager.download_model(model_name)
        finally:
            tts_manage.tqdm = original


class CoquiDownloader(BaseDownloader):
    """Downloader for Coqui TTS models."""

    # Model name mapping
    MODEL_NAMES = {
        "coqui-xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    }

    # Approximate sizes in bytes
    MODEL_SIZES = {
        "coqui-xtts-v2": 1800 * 1024 * 1024,  # ~1.8 GB
    }

    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Download a Coqui TTS model."""
        if model_id not in self.MODEL_NAMES:
            raise ValueError(f"Unknown Coqui model: {model_id}")

        model_name = self.MODEL_NAMES[model_id]
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
            # Import TTS lazily; a missing package surfaces as ImportError
            # and is reported with install guidance by the handler below.
            # Use the same provider-native cache root the registry checks.
            cache_dir = get_registry().get_cache_dir("coqui")

            model_path, _config_path, _model_item = _download_model_with_progress(
                model_name, cache_dir, total_size, model_id, progress_callback
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

            # Return the model directory
            return Path(model_path).parent

        except ImportError as e:
            install_command = 'pip install -e ".[coqui]"'
            logger.error("coqui-tts package not installed. Install with: %s", install_command)
            raise ImportError(f"coqui-tts package required. Install with: {install_command}") from e

    def get_model_size(self, model_id: str) -> int:
        """Get approximate model size in bytes."""
        return self.MODEL_SIZES.get(model_id, 0)


class CoquiProgressCallback:
    """Wrapper to convert Coqui's progress to our format."""

    def __init__(self, model_id: str, total_size: int, callback: ProgressCallback):
        self.model_id = model_id
        self.total_size = total_size
        self.callback = callback
        self.start_time = time.time()
        self.last_update = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        """Called by urllib during download."""
        downloaded = block_num * block_size
        actual_total = total_size if total_size > 0 else self.total_size

        # Throttle updates to avoid overwhelming the UI
        current_time = time.time()
        if current_time - self.last_update < 0.1 and downloaded < actual_total:
            return
        self.last_update = current_time

        elapsed = current_time - self.start_time
        speed = downloaded / elapsed if elapsed > 0 else 0
        remaining = actual_total - downloaded
        eta = remaining / speed if speed > 0 else -1

        self.callback(
            DownloadProgress(
                downloaded_bytes=downloaded,
                total_bytes=actual_total,
                speed_bytes_per_sec=speed,
                eta_seconds=eta,
                model_id=self.model_id,
                status="downloading",
            )
        )

"""Model download orchestration."""

import logging
from pathlib import Path

from models.download_progress import DownloadProgress, ProgressCallback
from models.exceptions import ModelDownloadError, ModelNotFoundError
from models.model_registry import ModelRegistry, get_registry

logger = logging.getLogger("voice_cloner.models")


class ModelDownloader:
    """Orchestrates model downloads with progress reporting."""

    def __init__(self, registry: ModelRegistry | None = None):
        self._registry = registry or get_registry()
        self._downloaders: dict[str, object] = {}

    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
        force: bool = False,
    ) -> Path:
        """
        Download a model.

        Args:
            model_id: ID of the model to download.
            progress_callback: Optional callback for progress updates.
            force: If True, re-download even if already installed.

        Returns:
            Path to the downloaded model.

        Raises:
            ModelNotFoundError: If model_id is not in registry.
            ModelDownloadError: If download fails.
        """
        model = self._registry.get_model(model_id)

        # Skip if already installed (unless forced)
        if model.is_installed and not force and model.install_path is not None:
            logger.info(f"Model '{model_id}' is already installed at {model.install_path}")
            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        downloaded_bytes=model.size_mb * 1024 * 1024,
                        total_bytes=model.size_mb * 1024 * 1024,
                        speed_bytes_per_sec=0,
                        eta_seconds=0,
                        model_id=model_id,
                        status="completed",
                    )
                )
            return model.install_path

        # Get the appropriate downloader
        downloader = self._get_downloader(model.engine)

        logger.info(f"Downloading model '{model_id}' (~{model.size_mb} MB)...")

        try:
            install_path = downloader.download(model_id, progress_callback)
            logger.info(f"Model '{model_id}' downloaded successfully to {install_path}")
            return install_path
        except Exception as e:
            raise ModelDownloadError(model_id, str(e)) from e

    def download_engine_models(
        self,
        engine: str,
        progress_callback: ProgressCallback | None = None,
        force: bool = False,
    ) -> list[Path]:
        """
        Download all models for an engine.

        Args:
            engine: Engine name (e.g., 'coqui', 'chatterbox').
            progress_callback: Optional callback for progress updates.
            force: If True, re-download even if already installed.

        Returns:
            List of paths to downloaded models.
        """
        models = self._registry.get_models_for_engine(engine)
        if not models:
            raise ModelNotFoundError(f"engine:{engine}", self._registry.list_model_ids())

        paths = []
        for model in models:
            path = self.download(model.id, progress_callback, force)
            paths.append(path)
        return paths

    def _get_downloader(self, engine: str):
        """Get or create a downloader for the given engine."""
        if engine not in self._downloaders:
            if engine == "coqui":
                from models.downloaders.coqui_downloader import CoquiDownloader

                self._downloaders[engine] = CoquiDownloader()
            elif engine == "chatterbox":
                from models.downloaders.chatterbox_downloader import ChatterboxDownloader

                self._downloaders[engine] = ChatterboxDownloader()
            elif engine == "mlx-audio":
                from models.downloaders.mlx_downloader import MlxDownloader

                self._downloaders[engine] = MlxDownloader()
            elif engine == "audio8-onnx":
                from models.downloaders.audio8_downloader import Audio8Downloader

                self._downloaders[engine] = Audio8Downloader()
            else:
                raise ModelDownloadError(
                    f"engine:{engine}",
                    f"No explicit downloader is available for engine '{engine}'. "
                    "Use an installed local cache or choose a supported model downloader.",
                )

        return self._downloaders[engine]


# Convenience functions
def download_model(
    model_id: str,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """Download a single model."""
    downloader = ModelDownloader()
    return downloader.download(model_id, progress_callback, force)


def download_engine_models(
    engine: str,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
) -> list[Path]:
    """Download all models for an engine."""
    downloader = ModelDownloader()
    return downloader.download_engine_models(engine, progress_callback, force)

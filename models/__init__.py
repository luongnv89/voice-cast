"""Model management module for VoiceCast TTS engines."""

from models.download_progress import DownloadProgress, ProgressCallback
from models.exceptions import ModelDownloadError, ModelNotFoundError, ModelNotInstalledError
from models.model_downloader import ModelDownloader, download_engine_models, download_model
from models.model_info import ModelInfo
from models.model_registry import ModelRegistry, get_registry

__all__ = [
    "ModelInfo",
    "DownloadProgress",
    "ProgressCallback",
    "ModelRegistry",
    "ModelDownloader",
    "ModelNotInstalledError",
    "ModelDownloadError",
    "ModelNotFoundError",
    "get_registry",
    "download_model",
    "download_engine_models",
]

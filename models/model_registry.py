"""Model registry for tracking available TTS models."""

import logging
import os
import platform
from pathlib import Path

from models.exceptions import ModelNotFoundError
from models.model_info import ModelInfo

logger = logging.getLogger("voice_cloner.models")


class ModelRegistry:
    """Registry for managing TTS model metadata and installation status."""

    # Default model definitions
    _DEFAULT_MODELS: list[ModelInfo] = [
        ModelInfo(
            id="coqui-xtts-v2",
            engine="coqui",
            name="Coqui XTTS v2",
            size_mb=1800,
            description="Multilingual voice cloning model supporting 16 languages. High quality, slower inference.",
            model_path_checker="tts/tts_models--multilingual--multi-dataset--xtts_v2",
        ),
        ModelInfo(
            id="chatterbox-turbo",
            engine="chatterbox",
            name="Chatterbox Turbo",
            size_mb=350,
            description="Fast English voice cloning (350M parameters). Supports paralinguistic tags like [laugh].",
            model_path_checker="ResembleAI/chatterbox",
        ),
        ModelInfo(
            id="chatterbox-standard",
            engine="chatterbox",
            name="Chatterbox Standard",
            size_mb=500,
            description="High-quality English voice cloning (500M parameters). Better prosody than Turbo.",
            model_path_checker="ResembleAI/chatterbox",
        ),
        ModelInfo(
            id="mlx-kokoro",
            engine="mlx-audio",
            name="MLX Kokoro",
            size_mb=164,
            description="Fast preset voice TTS optimized for Apple Silicon. 54 voices, multilingual (EN/JP/ZH).",
            model_path_checker="mlx-community/Kokoro-82M-bf16",
        ),
        ModelInfo(
            id="mlx-csm",
            engine="mlx-audio",
            name="MLX CSM",
            size_mb=2000,
            description="Voice cloning model optimized for Apple Silicon. High quality speech synthesis.",
            model_path_checker="mlx-community/csm-1b",
        ),
    ]

    _instance = None

    def __new__(cls):
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._models: dict[str, ModelInfo] = {}
        self._cache_dirs: dict[str, Path] = {}

        # Register default models
        for model in self._DEFAULT_MODELS:
            self._models[model.id] = model

        # Set up cache directories
        self._setup_cache_dirs()
        self._initialized = True

    def _setup_cache_dirs(self):
        """Set up cache directory paths for each engine."""
        # Check for custom model directory
        custom_dir = os.environ.get("VOICECAST_MODELS_DIR")
        if custom_dir:
            base_dir = Path(custom_dir)
            self._cache_dirs["coqui"] = base_dir / "coqui"
            self._cache_dirs["chatterbox"] = base_dir / "chatterbox"
            return

        # Use default cache locations
        home = Path.home()

        # Coqui uses ~/.local/share/tts on Linux, ~/Library/... on macOS
        if os.name == "nt":  # Windows
            coqui_cache = Path(os.environ.get("LOCALAPPDATA", home)) / "tts"
        elif platform.system() == "Darwin":  # macOS
            coqui_cache = home / "Library" / "Application Support" / "tts"
        else:  # Linux and others
            coqui_cache = home / ".local" / "share" / "tts"

        # Chatterbox uses HuggingFace hub cache
        hf_cache = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")) / "hub"

        self._cache_dirs["coqui"] = coqui_cache
        self._cache_dirs["chatterbox"] = hf_cache
        # MLX Audio also uses HuggingFace hub cache
        self._cache_dirs["mlx-audio"] = hf_cache

    def get_cache_dir(self, engine: str) -> Path:
        """Get cache directory for an engine."""
        return self._cache_dirs.get(engine, Path.home() / ".cache" / "voicecast" / engine)

    def list_models(self) -> list[ModelInfo]:
        """List all registered models with current installation status."""
        models = []
        for model_id in self._models:
            models.append(self.get_model(model_id))
        return models

    def get_model(self, model_id: str) -> ModelInfo:
        """Get model info with current installation status."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id, list(self._models.keys()))

        model = self._models[model_id]
        # Update installation status
        is_installed, install_path = self._check_installation(model)
        model.is_installed = is_installed
        model.install_path = install_path
        return model

    def is_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        model = self.get_model(model_id)
        return model.is_installed

    def get_install_path(self, model_id: str) -> Path | None:
        """Get the installation path for a model."""
        model = self.get_model(model_id)
        return model.install_path

    def get_models_for_engine(self, engine: str) -> list[ModelInfo]:
        """Get all models for a specific engine."""
        return [self.get_model(m.id) for m in self._models.values() if m.engine == engine]

    def _check_installation(self, model: ModelInfo) -> tuple[bool, Path | None]:
        """Check if a model is installed and return its path."""
        if not model.model_path_checker:
            return False, None

        cache_dir = self.get_cache_dir(model.engine)

        if model.engine == "coqui":
            # Coqui stores models in specific subdirectories
            model_path = cache_dir / model.model_path_checker
            if model_path.exists():
                return True, model_path

        elif model.engine == "chatterbox":
            # Chatterbox uses HuggingFace hub format: models--org--repo
            # Check for the hub cache format
            hub_path = cache_dir / f"models--{model.model_path_checker.replace('/', '--')}"
            if hub_path.exists():
                return True, hub_path

            # Also check direct path
            direct_path = cache_dir / model.model_path_checker
            if direct_path.exists():
                return True, direct_path

        elif model.engine == "mlx-audio":
            # MLX Audio uses HuggingFace hub format
            hub_path = cache_dir / f"models--{model.model_path_checker.replace('/', '--')}"
            if hub_path.exists():
                return True, hub_path

        return False, None

    def register_model(self, model: ModelInfo) -> None:
        """Register a new model or update existing."""
        self._models[model.id] = model
        logger.debug(f"Registered model: {model.id}")

    def get_engine_for_model(self, model_id: str) -> str:
        """Get the engine name for a model."""
        model = self.get_model(model_id)
        return model.engine


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

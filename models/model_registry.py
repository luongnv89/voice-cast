"""Model registry for tracking available TTS models."""

import logging
import os
import platform
from pathlib import Path

from models.exceptions import ModelNotFoundError
from models.model_info import ModelInfo

logger = logging.getLogger("voice_cloner.models")


def _coqui_default_cache_dir(home: Path) -> Path:
    """Mirror the Coqui backend's own default cache dir.

    Reproduces ``TTS.utils.generic_utils.get_user_data_dir("tts")`` from the
    pinned TTS 0.22.0 release so the registry checks the same location the
    engine lazy-loader resolves, including its ``TTS_HOME`` and
    ``XDG_DATA_HOME`` overrides.
    """
    tts_home = os.environ.get("TTS_HOME")
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if tts_home:
        base = Path(tts_home).expanduser().resolve(strict=False)
    elif xdg_data_home:
        base = Path(xdg_data_home).expanduser().resolve(strict=False)
    elif os.name == "nt":  # Windows: upstream reads the Local AppData shell folder
        base = Path(os.environ.get("LOCALAPPDATA", home))
    elif platform.system() == "Darwin":
        base = home / "Library" / "Application Support"
    else:  # Linux and others
        base = home / ".local" / "share"
    return base / "tts"


class ModelRegistry:
    """Registry for managing TTS model metadata and installation status.

    No longer a singleton — each instance owns its own model state.
    Use the module-level :func:`get_registry` for the default instance in
    production code, or construct isolated instances in tests.
    """

    # Public engine names used by the factory/CLI mapped to registry model IDs.
    _ENGINE_MODEL_IDS: dict[str, str] = {
        "coqui": "coqui-xtts-v2",
        "chatterbox-turbo": "chatterbox-turbo",
        "chatterbox-standard": "chatterbox-standard",
        "mlx-kokoro": "mlx-kokoro",
        "mlx-csm": "mlx-csm",
        "audio8-onnx": "audio8-tts",
    }

    # Group aliases accepted by CLI/API model-management commands.
    _ENGINE_GROUP_ALIASES: dict[str, str] = {
        "chatterbox": "chatterbox",
        "mlx": "mlx-audio",
        "mlx-audio": "mlx-audio",
        "audio8": "audio8-onnx",
        "audio8-onnx": "audio8-onnx",
    }

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
            model_path_checker="ResembleAI/chatterbox-turbo",
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
        ModelInfo(
            id="audio8-tts",
            engine="audio8-onnx",
            name="Audio8 TTS (1B)",
            size_mb=2000,
            description="High-quality voice cloning with Audio8 ONNX model. Supports reference audio voice cloning.",
            model_path_checker="audio8/audio8-TTS-0-1B-ONNX-INT8",
        ),
    ]

    def __init__(self):
        """Initialize the registry with default models."""
        self._models: dict[str, ModelInfo] = {}
        self._cache_dirs: dict[str, Path] = {}

        # Register default models
        for model in self._DEFAULT_MODELS:
            self._models[model.id] = model

        # Set up cache directories
        self._setup_cache_dirs()

    def _setup_cache_dirs(self):
        """Set up cache directory paths for each engine."""
        # Use provider-native cache locations. Keeping registry checks aligned
        # with backend loader defaults prevents generation-time cache misses.
        home = Path.home()

        # Coqui resolves its data dir via get_user_data_dir("tts"), honoring
        # TTS_HOME and XDG_DATA_HOME — derive ours from the same contract.
        coqui_cache = _coqui_default_cache_dir(home)

        # Chatterbox uses HuggingFace hub cache
        hf_cache = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")) / "hub"

        self._cache_dirs["coqui"] = coqui_cache
        self._cache_dirs["chatterbox"] = hf_cache
        # MLX Audio also uses HuggingFace hub cache
        self._cache_dirs["mlx-audio"] = hf_cache
        # Audio8 also uses HuggingFace hub cache
        self._cache_dirs["audio8-onnx"] = hf_cache

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

    def list_model_ids(self) -> list[str]:
        """List registered model IDs."""
        return list(self._models.keys())

    def get_model_id_for_engine(self, engine_name: str) -> str:
        """Map a public engine name to its default model ID."""
        if engine_name in self._ENGINE_MODEL_IDS:
            return self._ENGINE_MODEL_IDS[engine_name]
        if engine_name in self._models:
            return engine_name
        raise ModelNotFoundError(engine_name, self.list_model_ids())

    def get_models_for_engine(self, engine: str) -> list[ModelInfo]:
        """Get all models for a public engine name or engine group."""
        if engine in self._ENGINE_MODEL_IDS:
            return [self.get_model(self._ENGINE_MODEL_IDS[engine])]

        engine_group = self._ENGINE_GROUP_ALIASES.get(engine, engine)
        return [self.get_model(m.id) for m in self._models.values() if m.engine == engine_group]

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

        elif model.engine == "audio8-onnx":
            # Audio8 uses HuggingFace hub format
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

    def reset(self):
        """Reset the registry to its default state.

        Useful for test isolation — clears all registered models and
        cache directories, then reloads defaults.
        """
        self._models.clear()
        self._cache_dirs.clear()
        for model in self._DEFAULT_MODELS:
            self._models[model.id] = model
        self._setup_cache_dirs()


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

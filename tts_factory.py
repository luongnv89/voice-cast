import logging
from typing import Any

from tts_engine_base import TTSEngineBase
from utils.platform_utils import is_apple_silicon

logger = logging.getLogger("voice_cloner.factory")


class TTSFactory:
    """Factory for creating TTS engine instances."""

    # Engine registry: name -> (engine_class, variant_kwargs)
    _registry: dict[str, tuple] = {}

    # Engine display names for UI
    _display_names: dict[str, str] = {}

    # Engine metadata: name -> {requires_reference_audio, supports_preset_voices, ...}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        engine_class: type[TTSEngineBase],
        display_name: str,
        requires_reference_audio: bool = True,
        supports_preset_voices: bool = False,
        platform_restriction: str | None = None,
        **default_kwargs,
    ):
        """
        Register an engine class.

        Args:
            name: Unique identifier for the engine (e.g., "chatterbox-turbo")
            engine_class: The engine class to instantiate
            display_name: Human-readable name for UI
            requires_reference_audio: Whether this engine needs a voice reference file
            supports_preset_voices: Whether this engine has built-in voice presets
            platform_restriction: Platform requirement (e.g., "apple_silicon")
            **default_kwargs: Default kwargs passed to engine constructor
        """
        cls._registry[name] = (engine_class, default_kwargs)
        cls._display_names[name] = display_name
        cls._metadata[name] = {
            "requires_reference_audio": requires_reference_audio,
            "supports_preset_voices": supports_preset_voices,
            "platform_restriction": platform_restriction,
        }
        logger.debug(f"Registered TTS engine: {name}")

    @classmethod
    def create(cls, engine_name: str, speaker_wav: str, device: str | None = None, **engine_kwargs) -> TTSEngineBase:
        """
        Create an engine instance.

        Args:
            engine_name: Registered engine name
            speaker_wav: Path to speaker reference audio
            device: Device to use ("cuda" or "cpu")
            **engine_kwargs: Additional engine-specific parameters

        Returns:
            Configured TTSEngineBase instance
        """
        if engine_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown engine: '{engine_name}'. Available engines: {available}")

        engine_class, default_kwargs = cls._registry[engine_name]

        # Merge default kwargs with provided kwargs
        merged_kwargs = {**default_kwargs, **engine_kwargs}

        logger.info(f"Creating TTS engine: {engine_name}")
        return engine_class(speaker_wav=speaker_wav, device=device, **merged_kwargs)

    @classmethod
    def available_engines(cls) -> list[str]:
        """Get list of registered engine names."""
        return list(cls._registry.keys())

    @classmethod
    def get_display_name(cls, engine_name: str) -> str:
        """Get human-readable display name for an engine."""
        return cls._display_names.get(engine_name, engine_name)

    @classmethod
    def get_engine_info(cls) -> dict[str, str]:
        """Get dict of engine_name -> display_name for all engines."""
        return cls._display_names.copy()

    @classmethod
    def is_available(cls, engine_name: str) -> bool:
        """Check if an engine's dependencies are installed and platform is compatible."""
        if engine_name not in cls._registry:
            return False

        # Check platform restriction
        metadata = cls._metadata.get(engine_name, {})
        platform_restriction = metadata.get("platform_restriction")
        if platform_restriction == "apple_silicon" and not is_apple_silicon():
            return False

        # Check for required imports
        try:
            if "coqui" in engine_name.lower():
                import TTS  # noqa: F401
            elif "chatterbox" in engine_name.lower():
                import chatterbox  # noqa: F401
            elif "mlx" in engine_name.lower():
                import mlx_audio  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def get_available_engines(cls) -> list[str]:
        """Get list of engines that are available on this platform with dependencies satisfied."""
        return [name for name in cls._registry if cls.is_available(name)]

    @classmethod
    def get_engine_metadata(cls, engine_name: str) -> dict[str, Any]:
        """
        Get metadata for an engine.

        Args:
            engine_name: Engine identifier

        Returns:
            Dictionary with display_name, requires_reference_audio, supports_preset_voices, platform_restriction
        """
        if engine_name not in cls._registry:
            raise ValueError(f"Unknown engine: '{engine_name}'")

        return {
            "display_name": cls._display_names.get(engine_name, engine_name),
            **cls._metadata.get(engine_name, {}),
        }

    @classmethod
    def get_default_engine(cls) -> str:
        """
        Get the recommended default engine for the current platform.

        On Apple Silicon: prefers mlx-kokoro if available
        Otherwise: uses first available engine (typically coqui)

        Returns:
            Engine name string
        """
        available = cls.get_available_engines()
        if not available:
            raise RuntimeError("No TTS engines available. Please install dependencies.")

        # On Apple Silicon, prefer MLX Kokoro
        if is_apple_silicon() and "mlx-kokoro" in available:
            return "mlx-kokoro"

        # Default to first available engine
        return available[0]


def _register_default_engines():
    """Register the default TTS engines."""
    # Register Coqui engine
    try:
        from engines.coqui_engine import CoquiEngine

        TTSFactory.register(
            name="coqui",
            engine_class=CoquiEngine,
            display_name="Coqui XTTS v2",
            requires_reference_audio=True,
            supports_preset_voices=False,
        )
    except ImportError as e:
        logger.warning(f"Coqui engine not available: {e}")

    # Register Chatterbox engines
    try:
        from engines.chatterbox_engine import ChatterboxEngine

        # Chatterbox Turbo (fast, supports paralinguistic tags)
        TTSFactory.register(
            name="chatterbox-turbo",
            engine_class=ChatterboxEngine,
            display_name="Chatterbox Turbo (350M)",
            requires_reference_audio=True,
            supports_preset_voices=False,
            variant="turbo",
        )

        # Chatterbox Standard (higher quality)
        TTSFactory.register(
            name="chatterbox-standard",
            engine_class=ChatterboxEngine,
            display_name="Chatterbox Standard (500M)",
            requires_reference_audio=True,
            supports_preset_voices=False,
            variant="standard",
        )
    except ImportError as e:
        logger.warning(f"Chatterbox engines not available: {e}")

    # Register MLX Audio engines (Apple Silicon only)
    try:
        from engines.mlx_audio_engine import MlxAudioEngine

        # MLX Kokoro (preset voices, no reference audio needed)
        TTSFactory.register(
            name="mlx-kokoro",
            engine_class=MlxAudioEngine,
            display_name="MLX Kokoro (Preset Voices)",
            requires_reference_audio=False,
            supports_preset_voices=True,
            platform_restriction="apple_silicon",
            variant="kokoro",
        )

        # MLX CSM (voice cloning)
        TTSFactory.register(
            name="mlx-csm",
            engine_class=MlxAudioEngine,
            display_name="MLX CSM (Voice Cloning)",
            requires_reference_audio=True,
            supports_preset_voices=False,
            platform_restriction="apple_silicon",
            variant="csm",
        )
    except ImportError as e:
        logger.warning(f"MLX Audio engines not available: {e}")


# Auto-register engines on module import
_register_default_engines()

import logging
from dataclasses import dataclass, field
from typing import Any

from tts_engine_base import TTSEngineBase
from utils.platform_utils import is_apple_silicon

logger = logging.getLogger("voice_cloner.factory")


@dataclass
class EngineDescriptor:
    name: str
    engine_class: type[TTSEngineBase]
    display_name: str
    requires_reference_audio: bool = True
    supports_preset_voices: bool = False
    platform_restriction: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    controls_class: type | None = None

    def is_available_on_platform(self) -> bool:
        return not (self.platform_restriction == "apple_silicon" and not is_apple_silicon())

    def dependencies_installed(self) -> bool:
        try:
            if "coqui" in self.name.lower():
                import TTS  # noqa: F401
            elif "chatterbox" in self.name.lower():
                import chatterbox  # noqa: F401
            elif "mlx" in self.name.lower():
                import mlx_audio  # noqa: F401
            return True
        except ImportError:
            return False


class TTSFactory:
    """Factory for creating TTS engine instances."""

    _registry: dict[str, EngineDescriptor] = {}

    @classmethod
    def register(cls, descriptor: EngineDescriptor):
        """
        Register an engine class using a structured descriptor.

        Args:
            descriptor: EngineDescriptor with all engine metadata
        """
        if not isinstance(descriptor, EngineDescriptor):
            raise TypeError(f"Expected EngineDescriptor, got {type(descriptor).__name__}")
        if descriptor.name in cls._registry:
            raise ValueError(f"Engine '{descriptor.name}' is already registered")
        cls._registry[descriptor.name] = descriptor
        logger.debug(f"Registered TTS engine: {descriptor.name}")

    @classmethod
    def create(cls, engine_name: str, speaker_wav: str, device: str | None = None, **engine_kwargs) -> TTSEngineBase:
        if engine_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown engine: '{engine_name}'. Available engines: {available}")

        descriptor = cls._registry[engine_name]
        merged_kwargs = {**descriptor.default_kwargs, **engine_kwargs}

        logger.info(f"Creating TTS engine: {engine_name}")
        return descriptor.engine_class(speaker_wav=speaker_wav, device=device, **merged_kwargs)

    @classmethod
    def available_engines(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_display_name(cls, engine_name: str) -> str:
        descriptor = cls._registry.get(engine_name)
        return descriptor.display_name if descriptor else engine_name

    @classmethod
    def get_engine_info(cls) -> dict[str, str]:
        return {name: desc.display_name for name, desc in cls._registry.items()}

    @classmethod
    def is_available(cls, engine_name: str) -> bool:
        if engine_name not in cls._registry:
            return False
        descriptor = cls._registry[engine_name]
        if not descriptor.is_available_on_platform():
            return False
        return descriptor.dependencies_installed()

    @classmethod
    def get_available_engines(cls) -> list[str]:
        return [name for name in cls._registry if cls.is_available(name)]

    @classmethod
    def get_engine_metadata(cls, engine_name: str) -> dict[str, Any]:
        if engine_name not in cls._registry:
            raise ValueError(f"Unknown engine: '{engine_name}'")
        descriptor = cls._registry[engine_name]
        return {
            "display_name": descriptor.display_name,
            "requires_reference_audio": descriptor.requires_reference_audio,
            "supports_preset_voices": descriptor.supports_preset_voices,
            "platform_restriction": descriptor.platform_restriction,
        }

    @classmethod
    def get_controls_class(cls, engine_name: str) -> type | None:
        descriptor = cls._registry.get(engine_name)
        return descriptor.controls_class if descriptor else None

    @classmethod
    def get_default_engine(cls) -> str:
        available = cls.get_available_engines()
        if not available:
            raise RuntimeError("No TTS engines available. Please install dependencies.")

        if is_apple_silicon() and "mlx-kokoro" in available:
            return "mlx-kokoro"

        return available[0]


def _register_default_engines():
    try:
        from engines.coqui_engine import CoquiEngine
        from gui.engine_controls import CoquiControls

        TTSFactory.register(
            EngineDescriptor(
                name="coqui",
                engine_class=CoquiEngine,
                display_name="Coqui XTTS v2",
                requires_reference_audio=True,
                supports_preset_voices=False,
                controls_class=CoquiControls,
            )
        )
    except ImportError as e:
        logger.warning(f"Coqui engine not available: {e}")

    try:
        from engines.chatterbox_engine import ChatterboxEngine
        from gui.engine_controls import ChatterboxControls

        TTSFactory.register(
            EngineDescriptor(
                name="chatterbox-turbo",
                engine_class=ChatterboxEngine,
                display_name="Chatterbox Turbo (350M)",
                requires_reference_audio=True,
                supports_preset_voices=False,
                default_kwargs={"variant": "turbo"},
                controls_class=ChatterboxControls,
            )
        )

        TTSFactory.register(
            EngineDescriptor(
                name="chatterbox-standard",
                engine_class=ChatterboxEngine,
                display_name="Chatterbox Standard (500M)",
                requires_reference_audio=True,
                supports_preset_voices=False,
                default_kwargs={"variant": "standard"},
                controls_class=ChatterboxControls,
            )
        )
    except ImportError as e:
        logger.warning(f"Chatterbox engines not available: {e}")

    try:
        from engines.mlx_audio_engine import MlxAudioEngine
        from gui.engine_controls import MlxCsmControls, MlxKokoroControls

        TTSFactory.register(
            EngineDescriptor(
                name="mlx-kokoro",
                engine_class=MlxAudioEngine,
                display_name="MLX Kokoro (Preset Voices)",
                requires_reference_audio=False,
                supports_preset_voices=True,
                platform_restriction="apple_silicon",
                default_kwargs={"variant": "kokoro"},
                controls_class=MlxKokoroControls,
            )
        )

        TTSFactory.register(
            EngineDescriptor(
                name="mlx-csm",
                engine_class=MlxAudioEngine,
                display_name="MLX CSM (Voice Cloning)",
                requires_reference_audio=True,
                supports_preset_voices=False,
                platform_restriction="apple_silicon",
                default_kwargs={"variant": "csm"},
                controls_class=MlxCsmControls,
            )
        )
    except ImportError as e:
        logger.warning(f"MLX Audio engines not available: {e}")


def bootstrap_engines():
    """Initialize and register all available TTS engines.

    Application entry points must call this explicitly before using TTSFactory.
    Importing the module alone does not register any engines.
    """
    _register_default_engines()

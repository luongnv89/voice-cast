from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class TTSEngineBase(ABC):
    """Abstract base class for TTS engines.

    ``MAX_CHUNK_CHARS`` is the default maximum text length that
    :class:`voice_cloner.VoiceCloner` sends to one synthesis call.  Concrete
    engines should override it with a conservative, model-specific character
    limit.  A value of ``0`` means that an engine does not declare a default;
    callers can still opt into chunking by passing ``chunk_size`` explicitly.
    """

    MAX_CHUNK_CHARS: int = 0

    def __init__(self, speaker_wav: str, device: str | None = None):
        self.speaker_wav = speaker_wav
        self.device = device or self._default_device()

    @abstractmethod
    def generate(
        self,
        text: str,
        language: str = "en",
        chunk_size: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Args:
            text: The text to convert to speech.
            language: Language code (e.g., "en", "fr", "de").
            chunk_size: Effective character limit selected by VoiceCloner for
                this synthesis call. The text is already chunked by the
                caller; engines accept this value as part of the shared
                interface.
            **kwargs: Engine-specific parameters.

        Returns:
            Tuple of (audio_data: np.ndarray, sample_rate: int)
        """
        pass

    @abstractmethod
    def get_supported_parameters(self) -> dict[str, dict[str, Any]]:
        """
        Return supported parameters with their metadata.

        Returns:
            Dict mapping param_name -> {"type": type, "default": value, "description": str}
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        pass

    @property
    @abstractmethod
    def supports_languages(self) -> list:
        """List of supported language codes."""
        pass

    @staticmethod
    def _default_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

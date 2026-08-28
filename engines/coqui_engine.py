import logging
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any

import numpy as np
import soundfile as sf
import torch

from models.exceptions import ModelNotInstalledError
from models.model_registry import ModelRegistry, get_registry
from tts_engine_base import TTSEngineBase

logger = logging.getLogger("voice_cloner.coqui")

# TTS 0.22.0 checkpoints contain custom types that PyTorch 2.6+ rejects when
# ``weights_only`` is omitted. Keep the temporary compatibility patch scoped to
# model construction so other callers retain torch.load's safe default.
_COQUI_TORCH_LOAD_LOCK = RLock()
_COQUI_TORCH_LOAD_ACTIVE: ContextVar[bool] = ContextVar("_COQUI_TORCH_LOAD_ACTIVE", default=False)


@contextmanager
def _coqui_torch_load_compatibility():
    """Temporarily allow legacy Coqui checkpoints to be unpickled."""
    with _COQUI_TORCH_LOAD_LOCK:
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if _COQUI_TORCH_LOAD_ACTIVE.get():
                kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        token = _COQUI_TORCH_LOAD_ACTIVE.set(True)
        torch.load = patched_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load
            _COQUI_TORCH_LOAD_ACTIVE.reset(token)


# Model ID for registry lookup
COQUI_MODEL_ID = "coqui-xtts-v2"


def _ensure_mono(audio_data: np.ndarray) -> np.ndarray:
    """Ensure audio is mono (1D array)."""
    if len(audio_data.shape) > 1:
        return audio_data.mean(axis=1)
    return audio_data


class CoquiEngine(TTSEngineBase):
    """TTS engine using Coqui TTS (XTTS v2)."""

    SUPPORTED_LANGUAGES = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh",
        "ja",
        "hu",
        "ko",
    ]

    def __init__(
        self,
        speaker_wav: str,
        device: str | None = None,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        auto_download: bool = False,
        registry: ModelRegistry | None = None,
    ):
        """
        Initialize Coqui TTS engine.

        Args:
            speaker_wav: Path to speaker reference audio.
            device: Device to use ("cuda" or "cpu").
            model_name: Coqui model name.
            auto_download: If True, allow the backend to download a missing model.
                          Defaults to False so generation never downloads models
                          unless the caller explicitly opts in.
            registry: Model registry instance. Uses the global registry when None.
        """
        super().__init__(speaker_wav, device)
        self.model_name = model_name
        self.auto_download = auto_download
        self._tts = None  # Lazy loading
        self._registry = registry or get_registry()

    def _check_model_installed(self) -> bool:
        """Check if the model is installed."""
        return self._registry.is_installed(COQUI_MODEL_ID)

    @property
    def tts(self):
        """Lazy load TTS model on first use."""
        if self._tts is None:
            # Check if model is installed
            if not self._check_model_installed():
                if not self.auto_download:
                    raise ModelNotInstalledError(
                        model_id=COQUI_MODEL_ID,
                        engine="coqui",
                        install_command=f"python vcloner.py --download-models {COQUI_MODEL_ID}",
                    )
                logger.info(f"Model {COQUI_MODEL_ID} not found; auto_download=True so backend download is allowed...")

            from TTS.api import TTS

            logger.info(f"Loading Coqui TTS model: {self.model_name}")
            with _coqui_torch_load_compatibility():
                self._tts = TTS(model_name=self.model_name, progress_bar=True, gpu=(self.device == "cuda"))
            logger.info("Coqui TTS model loaded successfully")
        return self._tts

    @classmethod
    def is_model_installed(cls) -> bool:
        """Check if the Coqui XTTS v2 model is installed."""
        registry = get_registry()
        return registry.is_installed(COQUI_MODEL_ID)

    def generate(
        self, text: str, language: str = "en", temperature: float = 0.7, gpt_cond_len: int = 128, **kwargs
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Coqui TTS.

        Args:
            text: Text to synthesize.
            language: Language code.
            temperature: Sampling temperature (0.1-1.0).
            gpt_cond_len: GPT conditioning length.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Create temp file for output using mkstemp for better cross-platform support
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Close file descriptor immediately

        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.speaker_wav,
                file_path=temp_path,
                language=language,
                gpt_cond_len=gpt_cond_len,
                temperature=temperature,
            )
            audio_data, sample_rate = sf.read(temp_path)
            audio_data = _ensure_mono(audio_data)
            return audio_data.astype(np.float32), sample_rate
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_supported_parameters(self) -> dict[str, dict[str, Any]]:
        return {
            "language": {
                "type": str,
                "default": "en",
                "description": "Language code (en, es, fr, de, etc.)",
                "options": self.SUPPORTED_LANGUAGES,
            },
            "temperature": {
                "type": float,
                "default": 0.7,
                "description": "Sampling temperature (0.1-1.0)",
                "min": 0.1,
                "max": 1.0,
            },
            "gpt_cond_len": {
                "type": int,
                "default": 128,
                "description": "GPT conditioning length",
                "min": 32,
                "max": 256,
            },
        }

    @property
    def name(self) -> str:
        return "Coqui XTTS v2"

    @property
    def supports_languages(self) -> list[str]:
        return self.SUPPORTED_LANGUAGES

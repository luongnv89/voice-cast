"""MLX Audio TTS engine for Apple Silicon.

Supports two variants:
- Kokoro: Fast preset voice TTS (54 voices, multilingual)
- CSM: Voice cloning from reference audio
"""

import logging
import os
from typing import Any, Literal

import numpy as np

from models.exceptions import ModelNotInstalledError
from models.model_registry import get_registry
from tts_engine_base import TTSEngineBase
from utils.platform_utils import is_apple_silicon

logger = logging.getLogger("voice_cloner.mlx_audio")

MlxVariant = Literal["kokoro", "csm"]

# Model IDs for registry lookup
MLX_MODEL_IDS = {
    "kokoro": "mlx-kokoro",
    "csm": "mlx-csm",
}

# Kokoro voice presets organized by language and category
KOKORO_VOICES = {
    "American English": {
        "Female": [
            ("af_heart", "Heart (Warm)"),
            ("af_alloy", "Alloy"),
            ("af_aoede", "Aoede"),
            ("af_bella", "Bella"),
            ("af_jessica", "Jessica"),
            ("af_kore", "Kore"),
            ("af_nicole", "Nicole"),
            ("af_nova", "Nova"),
            ("af_river", "River"),
            ("af_sarah", "Sarah"),
            ("af_sky", "Sky"),
        ],
        "Male": [
            ("am_adam", "Adam"),
            ("am_echo", "Echo"),
            ("am_eric", "Eric"),
            ("am_fenrir", "Fenrir"),
            ("am_liam", "Liam"),
            ("am_michael", "Michael"),
            ("am_onyx", "Onyx"),
            ("am_puck", "Puck"),
            ("am_santa", "Santa"),
        ],
    },
    "British English": {
        "Female": [
            ("bf_alice", "Alice"),
            ("bf_emma", "Emma"),
            ("bf_isabella", "Isabella"),
            ("bf_lily", "Lily"),
        ],
        "Male": [
            ("bm_daniel", "Daniel"),
            ("bm_fable", "Fable"),
            ("bm_george", "George"),
            ("bm_lewis", "Lewis"),
        ],
    },
    "Japanese": {
        "Female": [
            ("jf_alpha", "Alpha"),
            ("jf_gongitsune", "Gongitsune"),
            ("jf_nezumi", "Nezumi"),
            ("jf_tebukuro", "Tebukuro"),
        ],
        "Male": [
            ("jm_kumo", "Kumo"),
        ],
    },
    "Mandarin Chinese": {
        "Female": [
            ("zf_xiaobei", "Xiaobei"),
            ("zf_xiaoni", "Xiaoni"),
            ("zf_xiaoxiao", "Xiaoxiao"),
            ("zf_xiaoyi", "Xiaoyi"),
        ],
        "Male": [
            ("zm_yunjian", "Yunjian"),
            ("zm_yunxi", "Yunxi"),
            ("zm_yunxia", "Yunxia"),
            ("zm_yunyang", "Yunyang"),
        ],
    },
}

# Language code mappings for Kokoro
KOKORO_LANG_CODES = {
    "American English": "a",
    "British English": "b",
    "Japanese": "j",
    "Mandarin Chinese": "z",
}


class MlxAudioEngine(TTSEngineBase):
    """TTS engine using MLX Audio for Apple Silicon."""

    # Supported languages depend on variant
    KOKORO_LANGUAGES = list(KOKORO_VOICES.keys())
    CSM_LANGUAGES = ["en"]  # CSM is English-only

    def __init__(
        self,
        speaker_wav: str,
        device: str | None = None,
        variant: MlxVariant = "kokoro",
        auto_download: bool = False,
    ):
        """
        Initialize MLX Audio TTS engine.

        Args:
            speaker_wav: Path to speaker reference audio (required for CSM, ignored for Kokoro).
            device: Device to use (defaults to "mps" on Apple Silicon).
            variant: "kokoro" (preset voices) or "csm" (voice cloning).
            auto_download: If True, allow the backend to download a missing model.
                          Defaults to False so generation never downloads models
                          unless the caller explicitly opts in.

        Raises:
            RuntimeError: If not running on Apple Silicon.
        """
        if not is_apple_silicon():
            raise RuntimeError(
                "MLX Audio requires Apple Silicon (M1/M2/M3/M4). This engine is not available on your platform."
            )

        super().__init__(speaker_wav, device)
        self.variant = variant
        self.auto_download = auto_download
        self._model = None
        self._sample_rate = 24000  # Kokoro default
        self._registry = get_registry()
        self._model_id = MLX_MODEL_IDS.get(variant, "mlx-kokoro")

    @staticmethod
    def _default_device() -> str:
        """MLX uses Metal Performance Shaders."""
        return "mps"

    def _check_model_installed(self) -> bool:
        """Check if the model is installed."""
        return self._registry.is_installed(self._model_id)

    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            # Check if model is installed
            if not self._check_model_installed():
                if not self.auto_download:
                    raise ModelNotInstalledError(
                        model_id=self._model_id,
                        engine="mlx-audio",
                        install_command=f"python vcloner.py --download-models {self._model_id}",
                    )
                logger.info(f"Model {self._model_id} not found; auto_download=True so backend download is allowed...")

            logger.info(f"Loading MLX Audio {self.variant} model...")
            try:
                from mlx_audio.tts.utils import load_model

                if self.variant == "kokoro":
                    self._model = load_model("mlx-community/Kokoro-82M-bf16")
                else:  # csm
                    self._model = load_model("mlx-community/csm-1b")

                logger.info(f"MLX Audio {self.variant} model loaded successfully")
            except ImportError as e:
                logger.error("mlx-audio package not installed. Install with: pip install mlx-audio")
                raise ImportError("mlx-audio package required. Install with: pip install -e '.[mlx]'") from e
        return self._model

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self._sample_rate

    def generate(
        self,
        text: str,
        language: str = "en",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using MLX Audio.

        Args:
            text: Text to synthesize.
            language: Language name (for UI, maps to lang_code).
            voice: Voice preset ID for Kokoro (e.g., "af_heart").
            speed: Speech speed multiplier (0.5-2.0).
            lang_code: Kokoro language code override (a, b, j, z).
            **kwargs: Additional parameters.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self.variant == "kokoro":
            return self._generate_kokoro(text, voice, speed, lang_code or "a")
        else:
            return self._generate_csm(text, speed)

    def _generate_kokoro(self, text: str, voice: str, speed: float, lang_code: str) -> tuple[np.ndarray, int]:
        """Generate audio using Kokoro preset voices."""
        audio_data = None

        # Kokoro returns a generator, we need the first result
        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
        ):
            audio_data = result.audio
            break

        if audio_data is None:
            raise RuntimeError("Kokoro model failed to generate audio")

        # Convert MLX array to numpy array
        # MLX arrays have a different API than PyTorch tensors
        import mlx.core as mx

        if isinstance(audio_data, mx.array):
            # MLX array - convert to numpy
            audio_data = np.array(audio_data)
        elif hasattr(audio_data, "numpy"):
            # PyTorch tensor or similar
            audio_data = audio_data.numpy()
        elif hasattr(audio_data, "cpu"):
            # PyTorch tensor on GPU
            audio_data = audio_data.cpu().numpy()

        return audio_data.astype(np.float32).squeeze(), self._sample_rate

    def _generate_csm(self, text: str, speed: float) -> tuple[np.ndarray, int]:
        """Generate audio using CSM voice cloning."""
        if not os.path.exists(self.speaker_wav):
            raise FileNotFoundError(f"Speaker reference file not found: {self.speaker_wav}")

        audio_data = None

        # CSM uses ref_audio parameter
        for result in self.model.generate(
            text=text,
            ref_audio=self.speaker_wav,
            speed=speed,
        ):
            audio_data = result.audio
            break

        if audio_data is None:
            raise RuntimeError("CSM model failed to generate audio")

        # Convert MLX array to numpy array
        import mlx.core as mx

        if isinstance(audio_data, mx.array):
            # MLX array - convert to numpy
            audio_data = np.array(audio_data)
        elif hasattr(audio_data, "numpy"):
            # PyTorch tensor or similar
            audio_data = audio_data.numpy()
        elif hasattr(audio_data, "cpu"):
            # PyTorch tensor on GPU
            audio_data = audio_data.cpu().numpy()

        return audio_data.astype(np.float32).squeeze(), self._sample_rate

    def get_supported_parameters(self) -> dict[str, dict[str, Any]]:
        """Return supported parameters with their metadata."""
        params = {
            "speed": {
                "type": float,
                "default": 1.0,
                "description": "Speech speed multiplier (0.5-2.0)",
                "min": 0.5,
                "max": 2.0,
            },
        }

        if self.variant == "kokoro":
            params["voice"] = {
                "type": str,
                "default": "af_heart",
                "description": "Voice preset ID",
                "options": self._get_all_voice_ids(),
            }
            params["lang_code"] = {
                "type": str,
                "default": "a",
                "description": "Language code (a=American, b=British, j=Japanese, z=Chinese)",
                "options": list(KOKORO_LANG_CODES.values()),
            }

        return params

    def _get_all_voice_ids(self) -> list[str]:
        """Get flat list of all Kokoro voice IDs."""
        voice_ids = []
        for lang_voices in KOKORO_VOICES.values():
            for category_voices in lang_voices.values():
                voice_ids.extend([v[0] for v in category_voices])
        return voice_ids

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        variant_names = {
            "kokoro": "MLX Kokoro (Preset Voices)",
            "csm": "MLX CSM (Voice Cloning)",
        }
        return variant_names.get(self.variant, "MLX Audio")

    @property
    def supports_languages(self) -> list[str]:
        """List of supported language codes."""
        if self.variant == "kokoro":
            return self.KOKORO_LANGUAGES
        return self.CSM_LANGUAGES

    @property
    def supports_preset_voices(self) -> bool:
        """Check if this engine supports preset voices (no reference audio needed)."""
        return self.variant == "kokoro"

    @property
    def requires_reference_audio(self) -> bool:
        """Check if this engine requires reference audio for voice cloning."""
        return self.variant == "csm"

    @classmethod
    def get_kokoro_voices(cls) -> dict:
        """Get the Kokoro voice collection organized by language and category."""
        return KOKORO_VOICES

    @classmethod
    def get_kokoro_lang_codes(cls) -> dict[str, str]:
        """Get language name to code mapping for Kokoro."""
        return KOKORO_LANG_CODES

    @classmethod
    def is_available(cls) -> bool:
        """Check if MLX Audio is available on this platform."""
        if not is_apple_silicon():
            return False
        try:
            import mlx_audio  # noqa: F401

            return True
        except ImportError:
            return False

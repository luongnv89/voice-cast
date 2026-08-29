import importlib
import logging
import os
from typing import Any, Literal

import numpy as np

from models.downloaders.chatterbox_downloader import CHATTERBOX_VARIANT_BACKENDS
from models.exceptions import ModelNotInstalledError
from models.model_registry import ModelRegistry, get_registry
from tts_engine_base import TTSEngineBase

logger = logging.getLogger("voice_cloner.chatterbox")

ChatterboxVariant = Literal["turbo", "standard"]

# Model IDs for registry lookup
CHATTERBOX_MODEL_IDS = {
    "turbo": "chatterbox-turbo",
    "standard": "chatterbox-standard",
}


class ChatterboxEngine(TTSEngineBase):
    """TTS engine using Chatterbox by Resemble AI."""

    # Conservative application-level limit for Chatterbox prompts.
    MAX_CHUNK_CHARS = 100

    # Languages supported by multilingual features (when available)
    SUPPORTED_LANGUAGES = ["en"]  # Base Chatterbox is English-focused

    # Paralinguistic tags supported by Turbo variant
    PARALINGUISTIC_TAGS = ["laugh", "chuckle", "cough", "sigh", "gasp", "yawn"]

    def __init__(
        self,
        speaker_wav: str,
        device: str | None = None,
        variant: ChatterboxVariant = "turbo",
        auto_download: bool = False,
        registry: ModelRegistry | None = None,
    ):
        """
        Initialize Chatterbox TTS engine.

        Args:
            speaker_wav: Path to speaker reference audio (~10 seconds recommended).
            device: Device to use ("cuda" or "cpu").
            variant: "turbo" (fast, 350M) or "standard" (higher quality, 500M).
            auto_download: If True, allow the backend to download a missing model.
                          Defaults to False so generation never downloads models
                          unless the caller explicitly opts in.
            registry: Model registry instance. Uses the global registry when None.
        """
        super().__init__(speaker_wav, device)
        self.variant = variant
        self.auto_download = auto_download
        self._model = None  # Lazy loading
        self._sample_rate = None
        self._registry = registry or get_registry()
        self._model_id = CHATTERBOX_MODEL_IDS.get(variant, "chatterbox-turbo")

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
                        engine="chatterbox",
                        install_command=f"python vcloner.py --download-models {self._model_id}",
                    )
                logger.info(f"Model {self._model_id} not found; auto_download=True so backend download is allowed...")

            logger.info(f"Loading Chatterbox {self.variant} model...")
            try:
                module_name, class_name = CHATTERBOX_VARIANT_BACKENDS[self._model_id]
                backend_cls = getattr(importlib.import_module(module_name), class_name)

                self._model = backend_cls.from_pretrained(device=self.device)
                self._sample_rate = self._model.sr
                logger.info(f"Chatterbox {self.variant} model loaded successfully")
            except ImportError as e:
                logger.error("chatterbox-tts package not installed. Install with: pip install chatterbox-tts")
                raise ImportError(
                    f"chatterbox-tts package required for {self._model_id}. Install with: pip install chatterbox-tts"
                ) from e
        return self._model

    @classmethod
    def is_model_installed(cls, variant: ChatterboxVariant = "turbo") -> bool:
        """Check if a Chatterbox model variant is installed."""
        registry = get_registry()
        model_id = CHATTERBOX_MODEL_IDS.get(variant, "chatterbox-turbo")
        return registry.is_installed(model_id)

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self._sample_rate is None:
            _ = self.model  # Trigger lazy load
        return self._sample_rate

    def generate(
        self,
        text: str,
        language: str = "en",
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        chunk_size: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Chatterbox.

        Args:
            text: Text to synthesize. For Turbo, can include tags like [laugh].
            language: Language code (primarily "en" for base models).
            cfg_weight: CFG weight (0.0-1.0). Lower values = better pacing for fast speakers.
            exaggeration: Expressiveness (0.0-1.5). Higher = more dramatic.
            chunk_size: Effective character limit selected by VoiceCloner.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Validate reference audio exists
        if not os.path.exists(self.speaker_wav):
            raise FileNotFoundError(f"Speaker reference file not found: {self.speaker_wav}")

        # Generate audio
        wav_tensor = self.model.generate(
            text,
            audio_prompt_path=self.speaker_wav,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )

        # Convert tensor to numpy array
        audio_data = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

        return audio_data, self.sample_rate

    def get_supported_parameters(self) -> dict[str, dict[str, Any]]:
        params = {
            "cfg_weight": {
                "type": float,
                "default": 0.5,
                "description": "CFG weight - controls text adherence (0.0-1.0). Lower for fast speakers.",
                "min": 0.0,
                "max": 1.0,
            },
            "exaggeration": {
                "type": float,
                "default": 0.5,
                "description": "Expressiveness level (0.0-1.5). Higher = more dramatic.",
                "min": 0.0,
                "max": 1.5,
            },
        }
        return params

    @property
    def name(self) -> str:
        variant_names = {"turbo": "Chatterbox Turbo (350M)", "standard": "Chatterbox Standard (500M)"}
        return variant_names.get(self.variant, "Chatterbox")

    @property
    def supports_languages(self) -> list[str]:
        return self.SUPPORTED_LANGUAGES

    @property
    def supports_paralinguistic_tags(self) -> bool:
        """Check if this variant supports paralinguistic tags."""
        return self.variant == "turbo"

    def get_paralinguistic_tags(self) -> list[str]:
        """Get list of supported paralinguistic tags for Turbo variant."""
        if self.supports_paralinguistic_tags:
            return self.PARALINGUISTIC_TAGS
        return []

    def validate_text(self, text: str) -> tuple[bool, str]:
        """
        Validate text for this engine variant.

        Returns:
            Tuple of (is_valid, message)
        """
        import re

        if not text.strip():
            return False, "Text cannot be empty"

        # Check for paralinguistic tags in non-Turbo variants
        tags_found = re.findall(r"\[(\w+)\]", text)
        if tags_found and not self.supports_paralinguistic_tags:
            return False, (
                f"Paralinguistic tags {tags_found} are only supported in Turbo variant. "
                "Switch to Chatterbox Turbo or remove the tags."
            )

        # Validate tags are recognized
        if tags_found and self.supports_paralinguistic_tags:
            invalid_tags = [t for t in tags_found if t not in self.PARALINGUISTIC_TAGS]
            if invalid_tags:
                return False, (f"Unknown tags: {invalid_tags}. Supported: {self.PARALINGUISTIC_TAGS}")

        return True, "OK"

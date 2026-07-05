import logging
import os
import warnings
from datetime import datetime

import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler
from transformers import logging as transformers_logging

from models import ModelDownloader, ModelInfo, ModelRegistry
from models.download_progress import ProgressCallback
from tts_engine_base import TTSEngineBase
from tts_factory import TTSFactory

# Suppress warnings globally
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress specific warnings from Hugging Face Transformers
transformers_logging.set_verbosity_error()

# Set up logging with RichHandler
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger("voice_cloner")
console = Console()


class VoiceCloner:
    """
    Voice cloning interface supporting multiple TTS engines.

    Supports:
    - Coqui XTTS v2 (default)
    - Chatterbox Turbo (fast, with paralinguistic tags)
    - Chatterbox Standard (higher quality)
    """

    def __init__(
        self,
        speaker_wav: str,
        engine: str | TTSEngineBase | None = None,
        device: str | None = None,
        auto_download: bool = False,
        **engine_kwargs,
    ):
        """
        Initialize the VoiceCloner.

        Args:
            speaker_wav: Path to speaker reference audio file.
            engine: Either an engine name (str) or a TTSEngineBase instance.
                   Defaults to "coqui" if not specified.
            device: Device to use ("cuda" or "cpu"). Auto-detected if None.
            auto_download: Explicit opt-in for backend model downloads. Defaults
                to False so model files are never downloaded during generation.
            **engine_kwargs: Additional parameters passed to engine constructor.
        """
        self.speaker_wav = speaker_wav
        self.device = device
        self.auto_download = auto_download

        # Create or use provided engine
        if engine is None:
            engine = "coqui"

        # Determine if this engine requires a speaker reference file
        requires_voice = True
        if isinstance(engine, str):
            try:
                metadata = TTSFactory.get_engine_metadata(engine)
                requires_voice = metadata.get("requires_reference_audio", True)
            except ValueError:
                pass  # Unknown engine, assume it needs voice file

        # Ensure the speaker reference file exists if required
        if requires_voice and self.speaker_wav:
            if not os.path.exists(self.speaker_wav):
                logger.error(f"Speaker reference file not found: {self.speaker_wav}")
                raise FileNotFoundError(f"Speaker reference file not found: {self.speaker_wav}")

        if isinstance(engine, str):
            engine_kwargs.setdefault("auto_download", auto_download)
            self.engine = TTSFactory.create(engine_name=engine, speaker_wav=speaker_wav, device=device, **engine_kwargs)
            self.engine_name = engine
        else:
            self.engine = engine
            self.engine_name = "custom"

        logger.info(f"VoiceCloner initialized with engine: {self.engine.name}")

    @classmethod
    def from_coqui(
        cls,
        speaker_wav: str,
        device: str | None = None,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    ) -> "VoiceCloner":
        """
        Create a VoiceCloner using Coqui TTS (backward compatible factory method).

        Args:
            speaker_wav: Path to speaker reference audio.
            device: Device to use.
            model_name: Coqui model name.

        Returns:
            VoiceCloner instance configured with Coqui engine.
        """
        return cls(speaker_wav=speaker_wav, engine="coqui", device=device, model_name=model_name)

    @classmethod
    def from_chatterbox(cls, speaker_wav: str, variant: str = "turbo", device: str | None = None) -> "VoiceCloner":
        """
        Create a VoiceCloner using Chatterbox TTS.

        Args:
            speaker_wav: Path to speaker reference audio (~10 seconds recommended).
            variant: "turbo" (fast, 350M) or "standard" (higher quality, 500M).
            device: Device to use.

        Returns:
            VoiceCloner instance configured with Chatterbox engine.
        """
        engine_name = f"chatterbox-{variant}"
        return cls(speaker_wav=speaker_wav, engine=engine_name, device=device)

    def say(
        self,
        text_to_voice: str,
        language: str = "en",
        play_audio: bool = True,
        save_audio: bool = False,
        output_file: str | None = None,
        speed: float = 1.0,
        **kwargs,
    ):
        """
        Convert text to speech using the configured engine.

        Args:
            text_to_voice: Text to synthesize.
            language: Language code (e.g., "en", "fr").
            play_audio: Whether to play the audio.
            save_audio: Whether to save to file.
            output_file: Output file path (auto-generated if not provided).
            speed: Playback speed multiplier.
            **kwargs: Engine-specific parameters (e.g., cfg_weight for Chatterbox).
        """
        logger.info(f"Generating speech for: '{text_to_voice[:50]}...' [{language}]")

        # Determine output file
        if save_audio and not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"generated_audio_{timestamp}.wav"

        if output_file:
            output_dir = os.path.dirname(output_file) or "."
            os.makedirs(output_dir, exist_ok=True)

        with console.status(f"[bold cyan]Generating audio with {self.engine.name}...[/bold cyan]"):
            try:
                # Generate audio using the engine
                audio_data, sample_rate = self.engine.generate(text=text_to_voice, language=language, **kwargs)

                # Save if requested
                if save_audio and output_file:
                    sf.write(output_file, audio_data, sample_rate)
                    logger.info(f"Audio saved to {output_file}")

                # Play if requested
                if play_audio:
                    self._play_audio(audio_data, sample_rate, speed)

            except Exception as e:
                logger.error(f"Error during TTS generation: {e}")
                raise

    def _play_audio(self, audio_data, sample_rate: int, speed: float = 1.0):
        """
        Play the generated audio.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Audio sample rate.
            speed: Playback speed multiplier.
        """
        try:
            adjusted_sample_rate = int(sample_rate * speed)
            sd.play(audio_data, adjusted_sample_rate)
            sd.wait()
            logger.info("Audio playback finished.")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def get_engine_parameters(self):
        """Get supported parameters for the current engine."""
        return self.engine.get_supported_parameters()

    def get_supported_languages(self):
        """Get supported languages for the current engine."""
        return self.engine.supports_languages

    @staticmethod
    def available_engines():
        """Get list of available TTS engines."""
        return TTSFactory.get_engine_info()

    @staticmethod
    def list_models() -> list[ModelInfo]:
        """List available models with current cache status."""
        return ModelRegistry().list_models()

    @staticmethod
    def is_model_installed(model_id: str) -> bool:
        """Return whether a model is present in the local cache."""
        return ModelRegistry().is_installed(model_id)

    @staticmethod
    def download_model(model_id: str, progress_callback: ProgressCallback | None = None):
        """Explicitly download a model and return its local path."""
        return ModelDownloader().download(model_id, progress_callback=progress_callback)

    @staticmethod
    def get_model_id_for_engine(engine_name: str) -> str:
        """Return the default model ID for a public engine name."""
        return ModelRegistry().get_model_id_for_engine(engine_name)

    def switch_engine(self, engine: str, **engine_kwargs) -> None:
        """Switch this cloner to another engine/model without implicit downloads."""
        engine_kwargs.setdefault("auto_download", self.auto_download)
        self.engine = TTSFactory.create(
            engine_name=engine,
            speaker_wav=self.speaker_wav,
            device=self.device,
            **engine_kwargs,
        )
        self.engine_name = engine
        logger.info(f"VoiceCloner switched to engine: {self.engine.name}")

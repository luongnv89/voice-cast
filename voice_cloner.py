import contextlib
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.logging import RichHandler

from models import ModelDownloader, ModelInfo
from models.download_progress import ProgressCallback
from models.model_registry import ModelRegistry, get_registry
from tts_engine_base import TTSEngineBase
from tts_factory import TTSFactory, bootstrap_engines
from utils import split_into_chunks

# Suppress warnings globally
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _configure_transformers_logging():
    """Silence Transformers warnings when an engine extra is installed."""
    try:
        from transformers import logging as transformers_logging
    except ImportError:
        return
    transformers_logging.set_verbosity_error()


# Suppress specific warnings from Hugging Face Transformers when available.
_configure_transformers_logging()

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
        registry: ModelRegistry | None = None,
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
            registry: Model registry instance. Uses the global registry when None.
            **engine_kwargs: Additional parameters passed to engine constructor.
        """
        self.speaker_wav = speaker_wav
        self.device = device
        self.auto_download = auto_download
        self._registry = registry or get_registry()

        # Auto-bootstrap engines so the documented README snippet
        # ``VoiceCloner(speaker_wav=..., engine=...)`` works without an
        # explicit ``bootstrap_engines()`` call. CLI already bootstraps
        # via vcloner.main(); this keeps API and CLI consistent.
        if not TTSFactory.available_engines():
            with contextlib.suppress(Exception):
                bootstrap_engines()

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
                # Unknown engine – it may be known after bootstrap; try once more
                if not TTSFactory.available_engines():
                    with contextlib.suppress(Exception):
                        bootstrap_engines()
                    with contextlib.suppress(ValueError):
                        metadata = TTSFactory.get_engine_metadata(engine)
                        requires_voice = metadata.get("requires_reference_audio", True)
                pass  # Unknown engine, assume it needs voice file
        else:
            requires_voice = getattr(engine, "requires_reference_audio", True)

        # Fail fast when reference audio is required but no path was provided,
        # instead of deferring the failure into engine.generate().
        if requires_voice and not self.speaker_wav:
            raise ValueError("A speaker reference audio path is required by this engine but was not provided.")

        # Ensure the speaker reference file exists if required
        if requires_voice and not os.path.exists(self.speaker_wav):
            logger.error(f"Speaker reference file not found: {self.speaker_wav}")
            raise FileNotFoundError(f"Speaker reference file not found: {self.speaker_wav}")

        if isinstance(engine, str):
            engine_kwargs.setdefault("auto_download", auto_download)
            engine_kwargs.setdefault("registry", self._registry)
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

    @staticmethod
    def _validate_chunking_parameters(chunk_size: int | None, silence_duration: int) -> None:
        """Validate the public chunking controls before generation starts."""
        if chunk_size is not None:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
                raise TypeError("chunk_size must be a positive integer or None")
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

        if isinstance(silence_duration, bool) or not isinstance(silence_duration, int):
            raise TypeError("silence_duration must be a non-negative integer")
        if silence_duration < 0:
            raise ValueError(f"silence_duration must be a non-negative integer, got {silence_duration}")

    @staticmethod
    def _resolve_chunk_size(engine: TTSEngineBase, chunk_size: int | None) -> int | None:
        """Resolve an explicit chunk size or the engine's declared default."""
        if chunk_size is not None:
            return chunk_size

        default = getattr(engine, "MAX_CHUNK_CHARS", 0)
        if isinstance(default, bool) or not isinstance(default, int) or default <= 0:
            return None
        return default

    def _generate_engine_audio(
        self,
        text: str,
        language: str,
        chunk_size: int | None,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Call the engine with the effective chunking contract."""
        if chunk_size is None:
            return self.engine.generate(text=text, language=language, **kwargs)
        return self.engine.generate(text=text, language=language, chunk_size=chunk_size, **kwargs)

    def generate(
        self,
        text: str,
        language: str = "en",
        chunk_size: int | None = None,
        silence_duration: int = 200,
        output_file: str | None = None,
        **kwargs,
    ) -> str:
        """Synthesize text, save the WAV file, and return its path.

        This convenience API always saves the generated audio and never plays
        it. Chunking uses the same controls as :meth:`say`; when ``chunk_size``
        is omitted, the configured engine's ``MAX_CHUNK_CHARS`` is used.
        """
        output_path = self.say(
            text,
            language=language,
            play_audio=False,
            save_audio=True,
            output_file=output_file,
            chunk_size=chunk_size,
            silence_duration=silence_duration,
            **kwargs,
        )
        if output_path is None:  # pragma: no cover - save_audio=True guarantees a path
            raise RuntimeError("Audio generation did not produce an output path")
        return output_path

    def say(
        self,
        text_to_voice: str,
        language: str = "en",
        play_audio: bool = True,
        save_audio: bool = False,
        output_file: str | None = None,
        chunk_size: int | None = None,
        silence_duration: int = 200,
        **kwargs,
    ) -> str | None:
        """
        Convert text to speech using the configured engine.

        Args:
            text_to_voice: Text to synthesize.
            language: Language code (e.g., "en", "fr").
            play_audio: Whether to play the audio.
            save_audio: Whether to save to file.
            output_file: Output file path (auto-generated if not provided).
            chunk_size: Maximum characters per synthesis chunk. When omitted,
                the engine's positive ``MAX_CHUNK_CHARS`` default is used. A
                value of ``0`` on the base class means no automatic limit. An
                explicit value overrides the engine default for this call. When
                the text is not longer than the effective limit, it is sent in
                a single engine call; otherwise it is split on sentence
                boundaries and each chunk is synthesized separately.
            silence_duration: Silence inserted between consecutive chunks, in
                **milliseconds** (default 200). Only takes effect on the
                chunked path; 0 inserts no silence.
            **kwargs: Engine-specific parameters (e.g., cfg_weight for Chatterbox).

        Returns:
            The path to the written WAV file when ``save_audio`` is True,
            otherwise None.
        """
        self._validate_chunking_parameters(chunk_size, silence_duration)
        effective_chunk_size = self._resolve_chunk_size(self.engine, chunk_size)
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
                # Generate audio using the engine. Only engage the chunked path
                # when the text is actually longer than the effective limit, so
                # short texts reach the engine unmodified.
                if effective_chunk_size is not None and len(text_to_voice) > effective_chunk_size:
                    audio_data, sample_rate = self._synthesize_chunked(
                        text_to_voice,
                        chunk_size=effective_chunk_size,
                        silence_duration=silence_duration,
                        language=language,
                        **kwargs,
                    )
                else:
                    audio_data, sample_rate = self._generate_engine_audio(
                        text_to_voice,
                        language=language,
                        chunk_size=effective_chunk_size,
                        **kwargs,
                    )

                # Save if requested
                if save_audio and output_file:
                    sf.write(output_file, audio_data, sample_rate)
                    logger.info(f"Audio saved to {output_file}")

                # Play if requested
                if play_audio:
                    self._play_audio(audio_data, sample_rate)

            except Exception as e:
                logger.error(f"Error during TTS generation: {e}")
                raise

        return output_file if save_audio else None

    def _synthesize_chunked(
        self,
        text: str,
        chunk_size: int,
        silence_duration: int,
        language: str,
        **kwargs,
    ):
        """
        Synthesize long text chunk by chunk and concatenate the result.

        Args:
            text: Full text to synthesize.
            chunk_size: Maximum characters per chunk.
            silence_duration: Silence between consecutive chunks, in milliseconds.
            language: Language code passed to the engine.
            **kwargs: Engine-specific parameters.

        Returns:
            Tuple of (concatenated audio samples, sample rate).

        Raises:
            RuntimeError: If the engine returns different sample rates for
                different chunks, which would garble the concatenated audio.
        """
        chunks = split_into_chunks(text, chunk_size)
        if not chunks:
            # Empty or whitespace-only text: fall back to a single engine call
            # so the caller still gets a defined (audio, sample_rate) pair.
            return self._generate_engine_audio(text, language, chunk_size, **kwargs)

        logger.info(f"Synthesizing {len(chunks)} chunks (chunk_size={chunk_size})")

        audio_chunks: list[np.ndarray] = []
        sample_rate: int | None = None
        for index, chunk in enumerate(chunks):
            chunk_audio, chunk_rate = self._generate_engine_audio(
                chunk,
                language=language,
                chunk_size=chunk_size,
                **kwargs,
            )
            chunk_audio = np.asarray(chunk_audio)
            if chunk_audio.ndim == 0:
                raise ValueError(f"Engine returned scalar audio for chunk {index}; expected an array of samples.")
            if sample_rate is None:
                sample_rate = chunk_rate
            elif chunk_rate != sample_rate:
                raise RuntimeError(
                    f"Engine returned inconsistent sample rates across chunks: "
                    f"chunk 0 produced {sample_rate} Hz but chunk {index} produced "
                    f"{chunk_rate} Hz. Refusing to concatenate mismatched audio."
                )
            if audio_chunks and chunk_audio.shape[1:] != audio_chunks[0].shape[1:]:
                raise RuntimeError(
                    f"Engine returned inconsistent audio shapes across chunks: "
                    f"chunk 0 has trailing shape {audio_chunks[0].shape[1:]} but chunk {index} "
                    f"has {chunk_audio.shape[1:]}. Refusing to concatenate mismatched audio."
                )
            audio_chunks.append(chunk_audio)

        if silence_duration > 0 and len(audio_chunks) > 1:
            silence_samples = int(sample_rate * silence_duration / 1000)
            silence_shape = (silence_samples, *audio_chunks[0].shape[1:])
            silence = np.zeros(silence_shape, dtype=audio_chunks[0].dtype)
            padded = []
            for index, chunk_audio in enumerate(audio_chunks):
                if index:
                    padded.append(silence)
                padded.append(chunk_audio)
            audio_chunks = padded

        return np.concatenate(audio_chunks, axis=0), sample_rate

    def _play_audio(self, audio_data, sample_rate: int):
        """
        Play the generated audio.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Audio sample rate.
        """
        try:
            sd.play(audio_data, sample_rate)
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
    def list_models(registry: ModelRegistry | None = None) -> list[ModelInfo]:
        """List available models with current cache status."""
        r = registry or get_registry()
        return r.list_models()

    @staticmethod
    def is_model_installed(model_id: str, registry: ModelRegistry | None = None) -> bool:
        """Return whether a model is present in the local cache."""
        r = registry or get_registry()
        return r.is_installed(model_id)

    @staticmethod
    def download_model(
        model_id: str,
        progress_callback: ProgressCallback | None = None,
        registry: ModelRegistry | None = None,
    ):
        """Explicitly download a model and return its local path."""
        return ModelDownloader(registry=registry).download(model_id, progress_callback=progress_callback)

    @staticmethod
    def get_model_id_for_engine(engine_name: str, registry: ModelRegistry | None = None) -> str:
        """Return the default model ID for a public engine name."""
        r = registry or get_registry()
        return r.get_model_id_for_engine(engine_name)

    def switch_engine(self, engine: str, **engine_kwargs) -> None:
        """Switch this cloner to another engine/model without implicit downloads."""
        if not TTSFactory.available_engines():
            with contextlib.suppress(Exception):
                bootstrap_engines()
        engine_kwargs.setdefault("auto_download", self.auto_download)
        engine_kwargs.setdefault("registry", self._registry)
        self.engine = TTSFactory.create(
            engine_name=engine,
            speaker_wav=self.speaker_wav,
            device=self.device,
            **engine_kwargs,
        )
        self.engine_name = engine
        logger.info(f"VoiceCloner switched to engine: {self.engine.name}")

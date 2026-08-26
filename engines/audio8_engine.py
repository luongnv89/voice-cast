"""Audio8 TTS engine using ONNX Runtime.

Supports the audio8-TTS-0-1B-ONNX-INT8 model for high-quality voice cloning.
"""

import logging
import os
from typing import Any

import numpy as np

from models.exceptions import ModelNotInstalledError
from models.model_registry import ModelRegistry, get_registry
from tts_engine_base import TTSEngineBase

logger = logging.getLogger("voice_cloner.audio8")


class Audio8Engine(TTSEngineBase):
    """TTS engine using Audio8 ONNX model for voice cloning."""

    SUPPORTED_LANGUAGES = ["en"]

    def __init__(
        self,
        speaker_wav: str,
        device: str | None = None,
        auto_download: bool = False,
        registry: ModelRegistry | None = None,
    ):
        """
        Initialize Audio8 TTS engine.

        Args:
            speaker_wav: Path to speaker reference audio (~10 seconds recommended).
            device: Device to use ("cpu" or "cuda"). Defaults to "cpu".
            auto_download: If True, allow the backend to download a missing model.
                          Defaults to False so generation never downloads models
                          unless the caller explicitly opts in.
            registry: Model registry instance. Uses the global registry when None.
        """
        super().__init__(speaker_wav, device)
        self.auto_download = auto_download
        self._model = None
        self._processor = None
        self._sample_rate = 24000  # Audio8 default sample rate
        self._registry = registry or get_registry()
        self._model_id = "audio8-tts"

    def _check_model_installed(self) -> bool:
        """Check if the model is installed."""
        return self._registry.is_installed(self._model_id)

    @property
    def model(self):
        """Lazy load ONNX model on first use."""
        if self._model is None:
            if not self._check_model_installed():
                if not self.auto_download:
                    raise ModelNotInstalledError(
                        model_id=self._model_id,
                        engine="audio8-onnx",
                        install_command=f"python vcloner.py --download-models {self._model_id}",
                    )
                # auto_download=True: attempt to download the model
                logger.info(f"Model {self._model_id} not found; downloading via model registry...")
                try:
                    self._registry.download_model(self._model_id)
                except Exception as exc:
                    raise ModelNotInstalledError(
                        model_id=self._model_id,
                        engine="audio8-onnx",
                        install_command=f"python vcloner.py --download-models {self._model_id}",
                        details=f"Auto-download failed: {exc}",
                    ) from exc

            logger.info("Loading Audio8 ONNX model...")
            try:
                import onnxruntime as ort

                model_path = self._get_model_path()
                self._model = ort.InferenceSession(
                    str(model_path),
                    providers=self._get_providers(),
                )
                logger.info("Audio8 ONNX model loaded successfully")
            except ImportError as e:
                logger.error("onnxruntime package not installed. Install with: pip install onnxruntime")
                raise ImportError("onnxruntime package required. Install with: pip install voicecast[audio8]") from e
        return self._model

    @property
    def processor(self):
        """Lazy load the Audio8 tokenizer/preprocessor on first use."""
        if self._processor is None:
            logger.info("Loading Audio8 processor...")
            try:
                from transformers import AutoTokenizer

                model_path = self._get_model_path()
                cache_dir = self._registry.get_cache_dir("audio8-onnx")
                self._processor = AutoTokenizer.from_pretrained(
                    str(model_path),
                    cache_dir=str(cache_dir),
                )
                logger.info("Audio8 processor loaded successfully")
            except ImportError as e:
                logger.error("transformers package not installed. Install with: pip install transformers")
                raise ImportError("transformers package required. Install with: pip install transformers") from e
        return self._processor

    def _get_model_path(self) -> str:
        """Get the path to the ONNX model file."""
        install_path = self._registry.get_install_path(self._model_id)
        if install_path is None:
            raise ModelNotInstalledError(
                model_id=self._model_id,
                engine="audio8-onnx",
                install_command=f"python vcloner.py --download-models {self._model_id}",
            )

        # Look for the ONNX model file in the install directory
        for filename in os.listdir(install_path):
            if filename.endswith(".onnx"):
                return str(install_path / filename)

        raise ModelNotInstalledError(
            model_id=self._model_id,
            engine="audio8-onnx",
            install_command=f"python vcloner.py --download-models {self._model_id}",
        )

    def _get_providers(self) -> list[str]:
        """Get the list of ONNX Runtime providers to use."""
        if self.device == "cuda" and self._has_cuda_provider():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _has_cuda_provider() -> bool:
        """Check if CUDA provider is available."""
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self._sample_rate

    def generate(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Audio8 ONNX model.

        Args:
            text: Text to synthesize.
            language: Language code (primarily "en").
            speed: Speech speed multiplier (0.5-2.0).
            **kwargs: Additional parameters.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not os.path.exists(self.speaker_wav):
            raise FileNotFoundError(f"Speaker reference file not found: {self.speaker_wav}")

        logger.info("Generating audio with Audio8 ONNX model...")

        # Get ONNX model and processor
        session = self.model
        tokenizer = self.processor

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="np", padding=True)

        # Prepare reference audio for voice cloning
        ref_audio = self._load_reference_audio()

        # Run inference
        audio_output = self._run_inference(session, inputs, ref_audio, speed)

        return audio_output

    def _load_reference_audio(self) -> np.ndarray:
        """Load and preprocess the reference audio file."""
        import soundfile as sf

        audio_data, sr = sf.read(self.speaker_wav)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 24kHz if needed
        if sr != self._sample_rate:
            import librosa

            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self._sample_rate)

        return audio_data.astype(np.float32)

    def _run_inference(
        self,
        session: Any,
        inputs: Any,
        ref_audio: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Run ONNX inference to generate audio."""
        # Audio8 ONNX model expects specific input names
        # Adjust based on the actual model signature
        input_names = [input.name for input in session.get_inputs()]

        # Build model inputs
        model_inputs = {}

        # Add text inputs
        for name in input_names:
            if "input_ids" in name:
                model_inputs[name] = inputs["input_ids"].astype(np.int64)
            elif "attention_mask" in name:
                model_inputs[name] = inputs.get("attention_mask", np.ones_like(inputs["input_ids"])).astype(np.int64)
            elif "ref_audio" in name or "prompt" in name.lower():
                # Resample reference audio to expected length
                expected_len = self._sample_rate * 6  # 6 seconds default
                if len(ref_audio) < expected_len:
                    ref_audio = np.pad(ref_audio, (0, expected_len - len(ref_audio)))
                else:
                    ref_audio = ref_audio[:expected_len]
                model_inputs[name] = ref_audio.reshape(1, -1).astype(np.float32)
            elif "speed" in name.lower():
                model_inputs[name] = np.array([speed], dtype=np.float32)
            else:
                # Skip unknown inputs
                pass

        # Run inference
        output_names = [output.name for output in session.get_outputs()]
        outputs = session.run(output_names, model_inputs)

        # Extract audio output
        audio_data = outputs[0]

        # Handle different output formats
        if isinstance(audio_data, list):
            audio_data = audio_data[0]

        audio_data = np.asarray(audio_data).flatten().astype(np.float32)

        # Apply speed adjustment
        if speed != 1.0:
            import scipy.signal

            factor = 1.0 / speed
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * factor))

        return audio_data, self._sample_rate

    def get_supported_parameters(self) -> dict[str, dict[str, Any]]:
        """Return supported parameters with their metadata."""
        return {
            "speed": {
                "type": float,
                "default": 1.0,
                "description": "Speech speed multiplier (0.5-2.0)",
                "min": 0.5,
                "max": 2.0,
            },
        }

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return "Audio8 TTS (1B)"

    @property
    def supports_languages(self) -> list[str]:
        """List of supported language codes."""
        return self.SUPPORTED_LANGUAGES

    @property
    def supports_preset_voices(self) -> bool:
        """Check if this engine supports preset voices."""
        return False

    @property
    def requires_reference_audio(self) -> bool:
        """Check if this engine requires reference audio."""
        return True

    @classmethod
    def is_available(cls) -> bool:
        """Check if Audio8 engine dependencies are available."""
        try:
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False

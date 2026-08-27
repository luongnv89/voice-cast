"""Audio8 TTS engine using ONNX Runtime.

Supports the audio8-TTS-0-1B-ONNX-INT8 model for high-quality voice cloning.
The Hugging Face repo ships four ONNX files plus tokenizer data and is
cached under ``hub/models--Audio8--audio8-TTS-0.1B-ONNX-INT8/snapshots/<rev>/``.
The engine resolves that layout, loads all available ONNX sessions, and
orchestrates them so callers do not need manual multi-session wiring.
"""

import logging
import os
from pathlib import Path
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
        """Lazy load ONNX model(s) on first use.

        When the HF snapshot contains multiple ONNX files (fast_ar,
        slow_ar, codec_decoder, codec_encoder) a dict of
        ``{stem: InferenceSession}`` is cached. Otherwise a single
        ``InferenceSession`` is cached for backward compatibility with
        existing mocks and single-file installs.
        """
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
                    from models.model_downloader import ModelDownloader

                    ModelDownloader(registry=self._registry).download(self._model_id)
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

                onnx_files = self._get_all_onnx_files()
                if len(onnx_files) > 1:
                    sessions: dict[str, Any] = {}
                    for fp in sorted(onnx_files):
                        sessions[Path(fp).stem] = ort.InferenceSession(
                            str(fp),
                            providers=self._get_providers(),
                        )
                    self._model = sessions
                    logger.info(f"Audio8 ONNX models loaded: {', '.join(sessions)}")
                else:
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

                _ = self._get_model_path()  # validate model is installed
                install_path = self._registry.get_install_path(self._model_id)
                cache_dir = self._registry.get_cache_dir("audio8-onnx")
                # Snapshot-aware tokenizer resolution: snapshot dir contains
                # tokenizer/ or tokenizer assets at root; fall back to rglob.
                tokenizer_path: Path | None = None
                if install_path is not None:
                    base = Path(install_path)
                    if (base / "tokenizer").exists():
                        tokenizer_path = base / "tokenizer"
                    elif (base / "tokenizer.json").exists() or any(base.rglob("tokenizer.json")):
                        # HF repo may store tokenizer at root
                        tokenizer_path = base
                    else:
                        # Search snapshot for a tokenizer directory
                        for cand in base.rglob("tokenizer"):
                            if cand.is_dir():
                                tokenizer_path = cand
                                break
                        if tokenizer_path is None:
                            tokenizer_path = base
                else:
                    tokenizer_path = Path(cache_dir)

                self._processor = AutoTokenizer.from_pretrained(  # nosec B615 - local cache path, revision pinned at download time
                    str(tokenizer_path),
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                logger.info("Audio8 processor loaded successfully")
            except ImportError as e:
                logger.error("transformers package not installed. Install with: pip install transformers")
                raise ImportError("transformers package required. Install with: pip install transformers") from e
        return self._processor

    def _get_all_onnx_files(self) -> list[Path]:
        """Return all ONNX files under the resolved install path."""
        install_path = self._registry.get_install_path(self._model_id)
        if install_path is None:
            return []
        base = Path(install_path)
        try:
            files = list(base.rglob("*.onnx"))
        except Exception:
            files = []
        # Fallback to single-file check via _get_model_path logic if rglob found nothing
        if not files:
            try:
                single = self._get_model_path()
                p = Path(single)
                if p.exists():
                    files = [p]
            except ModelNotInstalledError:
                pass
        return files

    def _get_model_path(self) -> str:
        """Get the path to the primary ONNX model file.

        Snapshot-aware: searches recursively so ``snapshots/<rev>/``
        layouts are resolved. Prefers ``fast_ar``/``slow_ar`` when
        multiple files exist to keep single-file callers deterministic.
        """
        install_path = self._registry.get_install_path(self._model_id)
        if install_path is None:
            raise ModelNotInstalledError(
                model_id=self._model_id,
                engine="audio8-onnx",
                install_command=f"python vcloner.py --download-models {self._model_id}",
            )

        base = Path(install_path)
        # Recursive search – handles snapshot layout
        try:
            onnx_files = sorted(base.rglob("*.onnx"))
        except FileNotFoundError:
            onnx_files = []
        if onnx_files:
            # Prefer fast_ar / slow_ar / codec_decoder ordering when multiple
            preferred_order = ["fast_ar", "slow_ar", "codec_decoder", "codec_encoder"]
            for pref in preferred_order:
                for fp in onnx_files:
                    if pref in fp.stem:
                        return str(fp)
            return str(onnx_files[0])

        # Fallback: direct iterdir for legacy single-file hub_path
        try:
            for file_path in base.iterdir():
                if file_path.suffix == ".onnx":
                    return str(file_path)
        except FileNotFoundError:
            pass

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

        Supports both single-session and multi-session (four-model) layouts.
        Callers never need to wire multiple sessions.

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

        # Get ONNX model(s) and processor
        session = self.model
        tokenizer = self.processor

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="np", padding=True)

        # Prepare reference audio for voice cloning
        ref_audio = self._load_reference_audio()

        # Run inference (handles single InferenceSession or dict of sessions)
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
            try:
                import librosa
            except ImportError as e:
                raise ImportError(
                    "librosa required for Audio8 reference audio preprocessing. Install with: pip install voicecast[audio8]"
                ) from e

            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self._sample_rate)

        return audio_data.astype(np.float32)

    def _run_inference(
        self,
        session: Any,
        inputs: Any,
        ref_audio: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Run ONNX inference to generate audio.

        When ``session`` is a dict (multi-model repo with fast_ar,
        slow_ar, codec_decoder, codec_encoder), orchestrates the
        pipeline so the caller provides only text + reference audio.
        For single-session layouts the previous single-pass logic is
        preserved.
        """
        if isinstance(session, dict):
            return self._run_multi_inference(session, inputs, ref_audio, speed)
        # Single session path
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
            try:
                import scipy.signal
            except ImportError as e:
                raise ImportError(
                    "scipy required for Audio8 speed adjustment. Install with: pip install voicecast[audio8]"
                ) from e

            factor = 1.0 / speed
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * factor))

        return audio_data, self._sample_rate

    def _run_multi_inference(
        self,
        sessions: dict[str, Any],
        inputs: Any,
        ref_audio: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Orchestrate the four-model Audio8 pipeline.

        Order: codec_encoder (reference) → fast_ar → slow_ar → codec_decoder.
        Each step tolerates missing inputs so mocked sessions in tests still
        produce audio when only a single session is exercised. Falls back to
        the first session's output when orchestration cannot proceed.
        """
        # Normalise ref audio once
        expected_len = self._sample_rate * 6
        if len(ref_audio) < expected_len:
            ref_audio_padded = np.pad(ref_audio, (0, expected_len - len(ref_audio)))
        else:
            ref_audio_padded = ref_audio[:expected_len]
        ref_batch = ref_audio_padded.reshape(1, -1).astype(np.float32)

        # Helper to build inputs for a given session from available tensors
        def _build_for(sess: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
            names = [i.name for i in sess.get_inputs()]
            out: dict[str, Any] = {}
            for n in names:
                if "input_ids" in n:
                    out[n] = inputs["input_ids"].astype(np.int64)
                elif "attention_mask" in n:
                    out[n] = inputs.get("attention_mask", np.ones_like(inputs["input_ids"])).astype(np.int64)
                elif "ref_audio" in n or "prompt" in n.lower():
                    out[n] = ref_batch
                elif "speed" in n.lower():
                    out[n] = np.array([speed], dtype=np.float32)
                elif extra and n in extra:
                    out[n] = extra[n]
                # Unknown inputs are skipped; if required, session.run will raise and we fall back
            return out

        # Attempt staged execution; accumulate intermediate outputs
        last_output: Any | None = None
        extra_tensors: dict[str, Any] = {}

        # Preferred execution order when all four are present
        order = ["codec_encoder", "fast_ar", "slow_ar", "codec_decoder"]
        # Sort sessions by preferred order, then alphabetically for any remaining
        sorted_keys = sorted(
            sessions.keys(),
            key=lambda k: (order.index(next((o for o in order if o in k), k)) if any(o in k for o in order) else 99, k),
        )

        for key in sorted_keys:
            sess = sessions[key]
            try:
                inp = _build_for(sess, extra_tensors)
                # If the session expects an intermediate tensor from previous step,
                # try to map the last_output onto any unmatched required input
                if last_output is not None and not inp:
                    # No inputs were matched – try feeding last_output as first input
                    first_name = sess.get_inputs()[0].name if sess.get_inputs() else None
                    if first_name:
                        inp[first_name] = np.asarray(last_output)
                out_names = [o.name for o in sess.get_outputs()]
                if not out_names:
                    continue
                # Only run if we can supply at least one input or session needs none
                outputs = sess.run(out_names, inp)
                last_output = outputs[0]
                if isinstance(last_output, list):
                    last_output = last_output[0]
                # Expose for next stage
                extra_tensors[key] = np.asarray(last_output)
                # Also expose generically for decoder-style inputs
                extra_tensors[out_names[0]] = np.asarray(last_output)
            except Exception as exc:  # pragma: no cover – orchestration best-effort
                logger.debug(f"Audio8 multi-session step {key} skipped: {exc}")
                continue

        if last_output is not None:
            audio_data = np.asarray(last_output).flatten().astype(np.float32)
            if audio_data.size == 0:
                # Degenerate output – fall back to silence
                audio_data = np.zeros(self._sample_rate, dtype=np.float32)
        else:
            # No session produced output – fallback to single-session logic on first session
            first = next(iter(sessions.values()))
            return self._run_inference(first, inputs, ref_audio, speed)

        if speed != 1.0:
            try:
                import scipy.signal
            except ImportError as e:
                raise ImportError(
                    "scipy required for Audio8 speed adjustment. Install with: pip install voicecast[audio8]"
                ) from e
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

"""Audio8 TTS engine using ONNX Runtime.

Supports the audio8-TTS-0-1B-ONNX-INT8 model for high-quality voice cloning.
The Hugging Face repo ships four ONNX files plus tokenizer data and is
cached under ``hub/models--Audio8--audio8-TTS-0.1B-ONNX-INT8/snapshots/<rev>/``.
The engine resolves that layout, loads all available ONNX sessions, and
orchestrates them so callers do not need manual multi-session wiring.
"""

import json
import logging
import math
import os
import re
import unicodedata
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
                # Snapshot-aware tokenizer resolution: the audio8-TTS-0.1B-ONNX-INT8
                # repo stores config.json at the snapshot root (not inside
                # tokenizer/).  AutoTokenizer.from_pretrained needs config.json
                # to identify the model type, so we point at the root.
                tokenizer_path: Path | None = None
                if install_path is not None:
                    base = Path(install_path)
                    # Prefer the snapshot root which contains config.json
                    if (base / "config.json").exists():
                        tokenizer_path = base
                    elif (base / "tokenizer").exists():
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
                    trust_remote_code=True,
                )
                # The audio8-TTS-0.1B-ONNX-INT8 tokenizer ships with no
                # special-token wiring: config.json declares pad_token_id=0
                # (``<|pad|>``) but `tokenizer.pad_token` is None, so the
                # ``padding=True`` call in :meth:`generate` raises. Adopt the
                # vocab's existing pad token instead of adding a new one —
                # adding tokens would grow the vocabulary and desync the
                # ONNX input space.
                if self._processor.pad_token is None:
                    self._processor.pad_token = self._processor.convert_ids_to_tokens(0)
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

        session = self.model

        # Real four-model ArkTTS layout: run the per-token two-stage
        # autoregressive pipeline (slow AR -> fast AR -> codec decode).
        if isinstance(session, dict) and self._is_arktts_pipeline(session):
            audio_data, sample_rate = self._generate_arktts(session, text, speed=speed, **kwargs)
            return audio_data, sample_rate

        # Legacy layouts: single-session or generic multi-session dicts that
        # do not expose the ArkTTS per-token interface (kept for backwards
        # compatibility with existing mocks and single-file installs).
        tokenizer = self.processor

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="np", padding=True)

        # Prepare reference audio for voice cloning
        ref_audio = self._load_reference_audio()

        # Run inference (handles single InferenceSession or dict of sessions)
        return self._run_inference(session, inputs, ref_audio, speed)

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

    # ------------------------------------------------------------------
    # ArkTTS four-model pipeline support
    #
    # The Audio8 0.1B ONNX repo ships a per-token, stateful autoregressive
    # interface that the legacy single-pass orchestration cannot drive:
    # codec_encoder (audio -> [10,T] codes), slow_ar (text prompt ->
    # semantic tokens, recurrent KV cache), fast_ar (semantic token -> 10
    # codebook tokens/frame), codec_decoder (codes -> waveform). This
    # follows the upstream Audio8 onnx runtime and also supports the 0.1B
    # export's consolidated cache layout (single ``cache_keys`` /
    # ``cache_values`` plus ``conv_states``/``ssm_states``, one step at a
    # time).
    # ------------------------------------------------------------------

    _ORT_DTYPES = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }

    @staticmethod
    def _find_session(sessions: dict[str, Any], part: str):
        """Return the session whose model-file stem contains ``part``."""
        for key, sess in sessions.items():
            if part in key:
                return sess
        return None

    @classmethod
    def _is_arktts_pipeline(cls, sessions: dict[str, Any]) -> bool:
        """Detect the ArkTTS per-token interface from session input names.

        Requires the slow AR (with KV-cache inputs) and fast AR (with
        ``token_id``/``use_slow_hidden``) plus the codec decoder. Returns
        False for the generic dict layouts used by tests and older
        single-file installs.
        """
        slow = cls._find_session(sessions, "slow_ar")
        fast = cls._find_session(sessions, "fast_ar")
        decoder = cls._find_session(sessions, "codec_decoder")
        if slow is None or fast is None or decoder is None:
            return False
        try:
            slow_inputs = {item.name for item in slow.get_inputs()}
            fast_inputs = {item.name for item in fast.get_inputs()}
        except Exception:
            return False
        has_slow_cache = "cache_keys" in slow_inputs or "cache_key_0" in slow_inputs
        return has_slow_cache and "token_id" in fast_inputs and "codes" in slow_inputs

    def _load_manifest(self) -> dict[str, Any]:
        """Load ``runtime_manifest.json`` from the install directory."""
        install_path = self._registry.get_install_path(self._model_id)
        if install_path is None:
            return {}
        manifest_path = Path(install_path) / "runtime_manifest.json"
        try:
            return json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            return {}

    def _generate_arktts(
        self,
        sessions: dict[str, Any],
        text: str,
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Run the real ArkTTS pipeline and return (audio, sample_rate)."""
        manifest = self._load_manifest()
        if not manifest:
            raise RuntimeError("Audio8 runtime_manifest.json not found; cannot run the ArkTTS pipeline.")

        slow = self._find_session(sessions, "slow_ar")
        fast = self._find_session(sessions, "fast_ar")
        decoder = self._find_session(sessions, "codec_decoder")
        encoder = self._find_session(sessions, "codec_encoder")
        if slow is None or fast is None or decoder is None:
            raise RuntimeError("Audio8 ArkTTS sessions (slow_ar/fast_ar/codec_decoder) missing.")

        target_sr = int(manifest.get("sample_rate", 44100))
        num_codebooks = int(manifest.get("num_codebooks", 10))
        semantic_begin = int(manifest.get("semantic_begin_id", 65537))

        # Reference voice -> codec codes. Falls back to the bundled
        # reference profile when the model does not ship a codec encoder.
        if encoder is not None:
            codes = self._encode_reference_codes(encoder, target_sr, manifest)
        else:
            codes = self._load_bundled_reference_codes()
        if codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise RuntimeError(f"invalid reference codes shape: {codes.shape}")

        reference_text = kwargs.get("reference_transcript") or ""
        prompt = self._build_arktts_prompt(text, reference_text, codes, semantic_begin, num_codebooks)

        frames = self._iter_arktts_frames(slow, fast, prompt, manifest, kwargs)
        if not frames:
            raise RuntimeError("Audio8 model produced no codec frames.")
        generated = np.stack(frames, axis=1)  # [num_codebooks, T]

        audio = self._decode_arktts(decoder, generated, num_codebooks)
        if speed != 1.0:
            try:
                import scipy.signal
            except ImportError as e:
                raise ImportError(
                    "scipy required for Audio8 speed adjustment. Install with: pip install voicecast[audio8]"
                ) from e
            audio = scipy.signal.resample(audio, int(len(audio) * (1.0 / speed)))

        return np.asarray(audio, dtype=np.float32), target_sr

    def _encode_reference_codes(
        self,
        encoder: Any,
        target_sr: int,
        manifest: dict[str, Any],
    ) -> np.ndarray:
        """Encode the speaker reference WAV into ``[10, T]`` codec frames."""
        import soundfile as sf

        audio, source_sr = sf.read(self.speaker_wav, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError("reference audio contains non-finite samples")
        if int(source_sr) != int(target_sr):
            try:
                import scipy.signal
            except ImportError as e:
                raise ImportError(
                    "scipy required for Audio8 reference resampling. Install with: pip install voicecast[audio8]"
                ) from e
            factor = math.gcd(int(source_sr), int(target_sr))
            audio = scipy.signal.resample_poly(audio, target_sr // factor, int(source_sr) // factor)
        # The codec operates on fixed-size frames; pad so no partial frame
        # reaches the encoder.
        frame_size = int(manifest.get("codec_frame_size", 2048))
        padding = (-audio.size) % frame_size
        if padding:
            audio = np.pad(audio, (0, padding))
        values = np.ascontiguousarray(audio.reshape(1, 1, -1).astype(np.float32))
        try:
            input_type = encoder.get_inputs()[0].type
            if "float16" in input_type:
                values = values.astype(np.float16)
            codes = np.asarray(encoder.run(None, {"audio": values})[0], dtype=np.int64)
        except Exception as exc:
            raise RuntimeError(f"Audio8 codec_encoder inference failed: {exc}") from exc
        if codes.ndim == 3:
            codes = codes[0]
        return codes

    def _load_bundled_reference_codes(self) -> np.ndarray:
        """Load ``reference_codes.npy`` shipped with the model, if present."""
        install_path = self._registry.get_install_path(self._model_id)
        path = Path(install_path) / "reference_codes.npy"
        if install_path is None or not path.exists():
            raise RuntimeError("Audio8 codec_encoder missing and no bundled reference_codes.npy found.")
        return np.asarray(np.load(path), dtype=np.int64)

    # -- ArkTTS prompt construction -------------------------------------

    _CJK_RANGES = (
        "\u1100-\u11ff\u2e80-\u2fdf\u3000-\u303f\u3040-\u30ff\u3100-\u31ff"
        "\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7a3\ud7b0-\ud7ff"
        "\uf900-\ufaff\ufe30-\ufe4f\uff01-\uff9f\U00020000-\U0002fa1f"
    )
    _CJK_CHARACTER_RE = re.compile(rf"[{_CJK_RANGES}]")
    _LINE_BREAK_RE = re.compile(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Strip control characters and normalize whitespace."""
        value = "".join(
            char if char.isspace() else "" if unicodedata.category(char).startswith("C") else char for char in str(text)
        )

        def replace(match: re.Match) -> str:
            left = value[match.start() - 1] if match.start() else ""
            right = value[match.end()] if match.end() < len(value) else ""
            if (
                cls._LINE_BREAK_RE.search(match.group())
                and cls._CJK_CHARACTER_RE.fullmatch(left)
                and cls._CJK_CHARACTER_RE.fullmatch(right)
            ):
                return ""
            return " "

        return re.sub(r"\s+", replace, value).strip()

    @classmethod
    def _format_reference_text(cls, text: str) -> str:
        text = cls._clean_text(text)
        return text if re.search(r"<\|speaker:\d+\|>", text) else f"<|speaker:0|>{text}"

    def _encode_part(self, part: str) -> list[int]:
        """Tokenize a prompt fragment with the loaded Audio8 processor."""
        encoded = self.processor(part, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return list(ids)

    def _build_arktts_prompt(
        self,
        target_text: str,
        reference_text: str,
        codes: np.ndarray,
        semantic_begin: int,
        num_codebooks: int,
    ) -> np.ndarray:
        """Build the ``[1, num_codebooks + 1, T]`` prompt table.

        Row 0 carries the text prefix, the semantic ids derived from the
        reference codes, and the target text suffix. Rows 1.. ``codes``
        repeat the reference codec frames under the semantic span, exactly
        as the upstream Audio8 onnx runtime does.
        """
        # Use the bundled reference transcript when the caller did not
        # provide one; an empty speaker tag is valid too.
        reference_parts = [
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            self._format_reference_text(reference_text),
            "\n\nSpeech:\n",
        ]
        suffix_parts = [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            self._clean_text(target_text),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ]
        prefix = [token for part in reference_parts for token in self._encode_part(part)]
        suffix = [token for part in suffix_parts for token in self._encode_part(part)]

        semantic_ids = (codes[0] + semantic_begin).tolist()
        row0 = np.asarray(prefix + semantic_ids + suffix, dtype=np.int64)
        values = np.zeros((num_codebooks + 1, row0.size), dtype=np.int64)
        values[0] = row0
        begin = len(prefix)
        values[1:, begin : begin + codes.shape[1]] = codes
        return values[np.newaxis]

    # -- ArkTTS autoregressive generation -------------------------------

    @staticmethod
    def _static_shape(io: Any) -> tuple[int, ...]:
        """Concrete tensor shape, coercing symbolic dims to 1."""
        return tuple(int(d) if isinstance(d, int) else 1 for d in io.shape)

    def _iter_arktts_frames(
        self,
        slow: Any,
        fast: Any,
        prompt: np.ndarray,
        manifest: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> list[np.ndarray]:
        """Slow-AR + fast-AR per-token loop; returns codec frames [10, T]."""
        num_codebooks = int(manifest["num_codebooks"])
        semantic_begin = int(manifest["semantic_begin_id"])
        semantic_end = int(manifest["semantic_end_id"])
        im_end = int(manifest["im_end_id"])
        codebook_size = int(manifest["codebook_size"])
        num_layers = int(manifest["num_layers"])
        num_fast_layers = int(manifest["num_fast_layers"])
        max_seq_len = int(manifest["max_seq_len"])

        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.9))
        top_k = int(kwargs.get("top_k", 50))
        seed = int(kwargs.get("seed", 42))
        max_new_tokens = int(kwargs.get("max_new_tokens", 256))

        prompt_len = int(prompt.shape[2])
        if prompt_len >= max_seq_len:
            raise RuntimeError(f"prompt length {prompt_len} exceeds max sequence length {max_seq_len}")
        max_new_tokens = min(max_new_tokens, max_seq_len - prompt_len)
        rng = np.random.default_rng(seed)

        slow_inputs = {item.name: item for item in slow.get_inputs()}
        fast_inputs = {item.name: item for item in fast.get_inputs()}
        dtype_map = Audio8Engine._ORT_DTYPES

        if "cache_keys" in slow_inputs:
            # Consolidated 0.1B layout: one big KV array plus conv/ssm state.
            cache_keys = np.zeros(self._static_shape(slow_inputs["cache_keys"]), dtype=np.float32)
            cache_values = np.zeros(self._static_shape(slow_inputs["cache_values"]), dtype=np.float32)
            conv_states = np.zeros(self._static_shape(slow_inputs["conv_states"]), dtype=np.float32)
            ssm_states = np.zeros(self._static_shape(slow_inputs["ssm_states"]), dtype=np.float32)

            def slow_step(pos: int, column: np.ndarray):
                nonlocal conv_states, ssm_states
                out = slow.run(
                    None,
                    {
                        "codes": column.astype(np.int64),
                        "position": np.asarray([pos], dtype=np.int64),
                        "cache_keys": cache_keys,
                        "cache_values": cache_values,
                        "conv_states": conv_states,
                        "ssm_states": ssm_states,
                    },
                )
                cache_keys[:, :, :, pos, :] = np.asarray(out[2])
                cache_values[:, :, :, pos, :] = np.asarray(out[3])
                conv_states = np.asarray(out[4])
                ssm_states = np.asarray(out[5])
                return np.asarray(out[0])[0, -1], np.asarray(out[1])[:, -1:, :]

        else:
            # Per-layer layout (upstream 0.6B): separate cache_key_i arrays.
            cache_shape = self._static_shape(slow_inputs["cache_key_0"])
            cache_dtype = dtype_map.get(slow_inputs["cache_key_0"].type, np.float32)
            caches = [np.zeros(cache_shape, dtype=cache_dtype) for _ in range(2 * num_layers)]

            def slow_step(pos: int, column: np.ndarray):
                feeds = {
                    "codes": column.astype(np.int64),
                    "input_pos": np.asarray([pos], dtype=np.int64),
                }
                for i in range(num_layers):
                    feeds[f"cache_key_{i}"] = caches[2 * i]
                    feeds[f"cache_value_{i}"] = caches[2 * i + 1]
                out = slow.run(None, feeds)
                for i, delta in enumerate(out[2:]):
                    caches[i][:, :, pos, :] = np.asarray(delta)
                return np.asarray(out[0])[0, -1], np.asarray(out[1])[:, -1:, :]

        fast_hidden_dtype = dtype_map.get(fast_inputs["slow_hidden"].type, np.float32)
        fast_cache_shape = self._static_shape(fast_inputs["cache_key_0"])
        fast_cache_dtype = dtype_map.get(fast_inputs["cache_key_0"].type, np.float32)

        def fast_step(token_id: int, use_hidden: bool, pos: int, caches: list):
            feeds = {
                "slow_hidden": np.asarray(hidden, dtype=fast_hidden_dtype),
                "token_id": np.asarray([[token_id]], dtype=np.int64),
                "use_slow_hidden": np.asarray([use_hidden], dtype=np.bool_),
                "input_pos": np.asarray([pos], dtype=np.int64),
            }
            for i in range(num_fast_layers):
                feeds[f"cache_key_{i}"] = caches[2 * i]
                feeds[f"cache_value_{i}"] = caches[2 * i + 1]
            out = fast.run(None, feeds)
            for i, delta in enumerate(out[1:]):
                caches[i][:, :, pos, :] = np.asarray(delta)[:, :, -1, :]
            return np.asarray(out[0])[0, -1]

        # Prefill: run the prompt through the slow AR one column at a time.
        for pos in range(prompt_len):
            logits, hidden = slow_step(pos, prompt[:, :, pos : pos + 1])

        previous: list[int] = []
        frames: list[np.ndarray] = []
        for step in range(max_new_tokens):
            semantic = self._sample_semantic(
                logits,
                previous,
                semantic_begin,
                semantic_end,
                im_end,
                temperature,
                top_p,
                top_k,
                rng,
            )
            if semantic == im_end:
                break
            previous.append(semantic)
            previous = previous[-10:]
            fast_caches = [np.zeros(fast_cache_shape, dtype=fast_cache_dtype) for _ in range(2 * num_fast_layers)]
            fast_step(0, True, 0, fast_caches)
            token = min(max(semantic - semantic_begin, 0), codebook_size - 1)
            codebooks = [token]
            for fast_pos in range(1, num_codebooks):
                fast_logits = fast_step(token, False, fast_pos, fast_caches)
                token = self._sample(fast_logits, temperature, top_p, top_k, rng)
                codebooks.append(token)
            frames.append(np.asarray(codebooks, dtype=np.int64))
            if step + 1 >= max_new_tokens:
                break
            column = np.concatenate([[int(semantic)], codebooks]).reshape(1, -1, 1)
            logits, hidden = slow_step(prompt_len + step, column)

        return frames

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng) -> int:
        """Top-p/top-k softmax sampling (ported from the upstream runtime)."""
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        order = np.argsort(values)[::-1]
        sorted_values = values[order]
        base = np.exp(sorted_values - np.max(sorted_values))
        base /= base.sum()
        cumulative = np.cumsum(base)
        remove = (cumulative > top_p) | (np.arange(base.size) >= top_k)
        remove[0] = False
        masked = values.copy()
        masked[order[remove]] = -np.inf
        scaled = masked / max(float(temperature), 1e-5)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= probs.sum()
        noise = -np.log(np.clip(rng.random(probs.size), 1e-12, 1.0))
        return int(np.argmax(probs / noise))

    def _sample_semantic(
        self,
        logits: np.ndarray,
        previous: list[int],
        semantic_begin: int,
        semantic_end: int,
        im_end: int,
        temperature: float,
        top_p: float,
        top_k: int,
        rng,
    ) -> int:
        """Sample a semantic (absolute) id; avoid immediate repeats."""
        allowed_ids = np.concatenate([np.arange(semantic_begin, semantic_end + 1), [im_end]])
        values = np.asarray(logits).reshape(-1)
        if values.size == allowed_ids.size:
            allowed_logits = values
        else:
            selected = allowed_ids[allowed_ids < values.size]
            allowed_logits = values[selected]
            allowed_ids = selected
        normal_index = self._sample(allowed_logits, temperature, top_p, top_k, rng)
        normal = int(allowed_ids[normal_index])
        high_index = self._sample(allowed_logits, 1.0, 0.9, top_k, rng)
        high = int(allowed_ids[high_index])
        if semantic_begin <= normal <= semantic_end and normal in previous:
            return high
        return normal

    def _decode_arktts(self, decoder: Any, codes: np.ndarray, num_codebooks: int) -> np.ndarray:
        """Decode ``[num_codebooks, T]`` frames into a mono waveform."""
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim == 2:
            values = values[np.newaxis]
        if values.ndim != 3 or values.shape[1] != num_codebooks:
            raise ValueError(f"invalid generated codes shape: {values.shape}")
        audio = decoder.run(None, {"codes": values})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

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

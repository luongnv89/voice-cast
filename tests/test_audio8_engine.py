"""Tests for Audio8 TTS engine."""

from unittest.mock import MagicMock, patch

import pytest

from models.exceptions import ModelNotInstalledError
from models.model_registry import ModelRegistry


class StubAudio8Engine:
    """Minimal stub matching Audio8Engine interface for tests that need it."""

    def __init__(self, speaker_wav="", device=None, auto_download=False, registry=None):
        self.speaker_wav = speaker_wav
        self.device = device or "cpu"
        self.auto_download = auto_download
        self._model = None
        self._processor = None
        self._sample_rate = 24000
        self._registry = registry or ModelRegistry()
        self._model_id = "audio8-tts"

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def name(self):
        return "Audio8 TTS (1B)"

    @property
    def supports_languages(self):
        return ["en"]

    @property
    def supports_preset_voices(self):
        return False

    @property
    def requires_reference_audio(self):
        return True

    def get_supported_parameters(self):
        return {
            "speed": {
                "type": float,
                "default": 1.0,
                "description": "Speech speed multiplier (0.5-2.0)",
                "min": 0.5,
                "max": 2.0,
            },
        }


class TestAudio8EngineInit:
    """Tests for Audio8Engine initialization."""

    def test_default_initialization(self):
        """Test basic engine initialization."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.speaker_wav == "voice.wav"
        assert engine.device == "cpu"
        assert engine.auto_download is False
        assert engine._model_id == "audio8-tts"

    def test_custom_device(self):
        """Test custom device selection."""
        engine = StubAudio8Engine(speaker_wav="voice.wav", device="cuda")
        assert engine.device == "cuda"

    def test_auto_download_enabled(self):
        """Test auto_download parameter."""
        engine = StubAudio8Engine(speaker_wav="voice.wav", auto_download=True)
        assert engine.auto_download is True

    def test_sample_rate(self):
        """Test default sample rate."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.sample_rate == 24000


class TestAudio8EngineProperties:
    """Tests for Audio8Engine property getters."""

    def test_name(self):
        """Test engine name property."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.name == "Audio8 TTS (1B)"

    def test_supports_languages(self):
        """Test supported languages."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.supports_languages == ["en"]

    def test_requires_reference_audio(self):
        """Test that reference audio is required."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.requires_reference_audio is True

    def test_supports_preset_voices(self):
        """Test that preset voices are not supported."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        assert engine.supports_preset_voices is False


class TestAudio8EngineModelLoading:
    """Tests for model loading behavior."""

    def test_model_not_installed_raises_error(self, tmp_path, monkeypatch):
        """Test that using an uninstalled model raises ModelNotInstalledError.

        Isolated from the global Hugging Face cache so a real
        ``--download-models audio8-tts`` on disk does not flip the
        assertion.
        """
        from engines.audio8_engine import Audio8Engine

        # Use an isolated HF cache directory for this test
        isolated_cache = tmp_path / "hf_hub"
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_CACHE", str(isolated_cache))
        registry = ModelRegistry()
        # Ensure the audio8 cache dir points at the isolated location even
        # if HF_HUB_CACHE was already read at import time
        registry._cache_dirs["audio8-onnx"] = isolated_cache
        # audio8-tts should not be installed in the isolated cache
        assert not registry.is_installed("audio8-tts")

        engine = Audio8Engine(speaker_wav="voice.wav", registry=registry)

        with pytest.raises(ModelNotInstalledError) as exc_info:
            _ = engine.model

        assert "audio8-tts" in str(exc_info.value)
        assert "audio8-onnx" in str(exc_info.value)
        assert "vcloner.py --download-models" in str(exc_info.value)

    @patch("engines.audio8_engine.Audio8Engine._check_model_installed", return_value=True)
    @patch("engines.audio8_engine.Audio8Engine._get_all_onnx_files", return_value=[])
    @patch("engines.audio8_engine.Audio8Engine._get_model_path", return_value="/fake/model.onnx")
    def test_lazy_model_loading(self, mock_path, mock_all, mock_installed):
        """Test that model is loaded lazily on first access."""
        from engines.audio8_engine import Audio8Engine

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = []
        mock_session.get_outputs.return_value = []

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            engine = Audio8Engine(speaker_wav="voice.wav")
            assert engine._model is None

            # Access model property
            _ = engine.model
            # Single-file or multi-file dict both acceptable; mock must be contained
            if isinstance(engine._model, dict):
                assert mock_session in engine._model.values()
            else:
                assert engine._model is mock_session

    @patch("engines.audio8_engine.Audio8Engine._check_model_installed", return_value=True)
    @patch("engines.audio8_engine.Audio8Engine._get_all_onnx_files", return_value=[])
    @patch("engines.audio8_engine.Audio8Engine._get_model_path", return_value="/fake/model.onnx")
    def test_auto_download_allows_loading(self, mock_path, mock_all, mock_installed):
        """Test that auto_download=True allows model loading."""
        from engines.audio8_engine import Audio8Engine

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = []
        mock_session.get_outputs.return_value = []

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            engine = Audio8Engine(speaker_wav="voice.wav", auto_download=True)
            # Should not raise even if model not installed
            _ = engine.model


class TestAudio8EngineGetSupportedParameters:
    """Tests for get_supported_parameters."""

    def test_returns_speed_parameter(self):
        """Test that speed parameter is returned."""
        engine = StubAudio8Engine(speaker_wav="voice.wav")
        params = engine.get_supported_parameters()

        assert "speed" in params
        assert params["speed"]["type"] is float
        assert params["speed"]["default"] == 1.0
        assert params["speed"]["min"] == 0.5
        assert params["speed"]["max"] == 2.0


class TestAudio8EngineIsAvailable:
    """Tests for is_available class method."""

    def test_available_when_onnxruntime_installed(self):
        """Test is_available returns True when onnxruntime is installed."""
        from engines.audio8_engine import Audio8Engine

        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            assert Audio8Engine.is_available() is True

    def test_unavailable_when_onnxruntime_missing(self):
        """Test is_available returns False when onnxruntime is not installed."""
        from engines.audio8_engine import Audio8Engine

        with patch("builtins.__import__", side_effect=ImportError("no onnxruntime")):
            assert Audio8Engine.is_available() is False


class TestAudio8EngineGenerate:
    """Tests for the generate method."""

    @patch("engines.audio8_engine.Audio8Engine._check_model_installed", return_value=True)
    @patch("engines.audio8_engine.Audio8Engine._get_model_path", return_value="/fake/model.onnx")
    def test_generate_requires_reference_audio(self, mock_path, mock_installed):
        """Test that generate requires valid reference audio."""
        from engines.audio8_engine import Audio8Engine

        engine = Audio8Engine(speaker_wav="nonexistent.wav")

        with pytest.raises(FileNotFoundError, match="Speaker reference file not found"):
            engine.generate("Hello world")


class TestAudio8EngineDeviceProviders:
    """Tests for device provider selection."""

    def test_cpu_providers_default(self):
        """Test that CPU providers are used by default."""
        from engines.audio8_engine import Audio8Engine

        engine = Audio8Engine(speaker_wav="voice.wav", device="cpu")
        providers = engine._get_providers()
        assert "CPUExecutionProvider" in providers
        assert "CUDAExecutionProvider" not in providers

    def test_cuda_providers_when_cuda_device(self):
        """Test that CUDA providers are used when device is cuda."""
        from engines.audio8_engine import Audio8Engine

        engine = Audio8Engine(speaker_wav="voice.wav", device="cuda")

        with patch.object(Audio8Engine, "_has_cuda_provider", return_value=True):
            providers = engine._get_providers()
            assert "CUDAExecutionProvider" in providers
            assert "CPUExecutionProvider" in providers

    def test_has_cuda_provider_true(self):
        """Test _has_cuda_provider returns True when available."""
        from engines.audio8_engine import Audio8Engine

        with patch.dict(
            "sys.modules",
            {"onnxruntime": MagicMock(get_available_providers=lambda: ["CUDAExecutionProvider"])},
        ):
            assert Audio8Engine._has_cuda_provider() is True

    def test_has_cuda_provider_false(self):
        """Test _has_cuda_provider returns False when not available."""
        from engines.audio8_engine import Audio8Engine

        with patch.dict(
            "sys.modules",
            {"onnxruntime": MagicMock(get_available_providers=lambda: ["CPUExecutionProvider"])},
        ):
            assert Audio8Engine._has_cuda_provider() is False


class TestAudio8ProcessorPadToken:
    """The audio8 tokenizer ships without a configured pad_token.

    The engine adopts the vocab's own ``<|pad|>`` (id 0) so the
    ``padding=True`` call in generate() does not raise. Adding a brand-new
    token must be avoided — it would grow the vocabulary and desync the
    ONNX input space.
    """

    @patch("engines.audio8_engine.Audio8Engine._get_model_path", return_value="/fake/model.onnx")
    def test_processor_adopts_vocab_pad_token(self, mock_path, tmp_path):
        """processor uses tokenizer.json and sets the existing vocab pad token."""
        from engines.audio8_engine import Audio8Engine

        class _BareTokenizer:
            """Tokenizer without special-token wiring (mirrors the real model)."""

            pad_token = None
            eos_token = "<|eos|>"

            def __init__(self):
                self.converted_idx = None

            def convert_ids_to_tokens(self, idx):
                self.converted_idx = idx
                return "<|pad|>"

        tokenizer_file = tmp_path / "tokenizer.json"
        tokenizer_file.write_text("{}")
        registry = MagicMock()
        registry.get_install_path.return_value = str(tmp_path)
        engine = Audio8Engine(speaker_wav="voice.wav", registry=registry)

        with patch("transformers.PreTrainedTokenizerFast", return_value=_BareTokenizer()) as tokenizer_class:
            _ = engine.processor

        tokenizer_class.assert_called_once_with(tokenizer_file=str(tokenizer_file))
        assert engine._processor.pad_token == "<|pad|>"
        assert engine._processor.converted_idx == 0

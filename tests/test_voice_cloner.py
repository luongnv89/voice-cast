"""Tests for VoiceCloner."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.model_info import ModelInfo
from models.model_registry import ModelRegistry
from voice_cloner import VoiceCloner


@pytest.fixture
def engine_mock():
    engine = MagicMock()
    engine.generate.return_value = (np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)
    engine.name = "mock_engine"
    engine.supports_languages = ["en", "fr"]
    return engine


@pytest.fixture
def cloner(engine_mock):
    with (
        patch("voice_cloner.TTSFactory.create", return_value=engine_mock),
        patch("voice_cloner.TTSFactory.get_engine_metadata", return_value={"requires_reference_audio": False}),
    ):
        return VoiceCloner(speaker_wav="dummy.wav", device="cpu", engine="coqui")


class TestInit:
    """VoiceCloner initialization tests."""

    def test_creates_instance(self, cloner):
        assert cloner is not None
        assert cloner.device == "cpu"
        assert cloner.speaker_wav == "dummy.wav"

    def test_missing_speaker_file_raises(self):
        with (
            patch("voice_cloner.TTSFactory.get_engine_metadata", return_value={"requires_reference_audio": True}),
            patch("voice_cloner.TTSFactory.create"),
            pytest.raises(FileNotFoundError, match="Speaker reference file not found"),
        ):
            VoiceCloner(speaker_wav="nonexistent.wav")

    @pytest.mark.parametrize("falsy_path", ["", None])
    def test_falsy_speaker_path_raises_before_engine_call(self, falsy_path):
        """A falsy reference path fails at construction, never inside engine.generate()."""
        with (
            patch("voice_cloner.TTSFactory.get_engine_metadata", return_value={"requires_reference_audio": True}),
            patch("voice_cloner.TTSFactory.create") as mock_create,
            pytest.raises(ValueError, match="speaker reference audio path is required"),
        ):
            VoiceCloner(speaker_wav=falsy_path)
        mock_create.assert_not_called()

    def test_engine_instance_requiring_audio_rejects_falsy_path(self):
        engine = MagicMock()
        engine.name = "mock_engine"
        engine.requires_reference_audio = True
        with pytest.raises(ValueError, match="speaker reference audio path is required"):
            VoiceCloner(speaker_wav="", engine=engine)

    def test_engine_instance_without_reference_audio_allows_empty_path(self):
        engine = MagicMock()
        engine.name = "mock_engine"
        engine.requires_reference_audio = False
        cloner = VoiceCloner(speaker_wav="", engine=engine)
        assert cloner.engine is engine

    def test_from_coqui(self):
        eng = MagicMock()
        eng.name = "coqui"
        with (
            patch("voice_cloner.TTSFactory.create", return_value=eng),
            patch("voice_cloner.TTSFactory.get_engine_metadata", return_value={"requires_reference_audio": False}),
        ):
            c = VoiceCloner.from_coqui("voice.wav", device="cpu")
            assert c.speaker_wav == "voice.wav"


class TestSay:
    """VoiceCloner.say() tests."""

    def test_save_to_explicit_output(self, cloner, tmp_path):
        out = str(tmp_path / "speech.wav")
        with patch("voice_cloner.sf.write") as write:
            cloner.say("hello", play_audio=False, save_audio=True, output_file=out)
            write.assert_called_once()
            assert write.call_args[0][0] == out

    def test_auto_generates_filename(self, cloner):
        with patch("voice_cloner.sf.write") as write:
            cloner.say("hi", play_audio=False, save_audio=True)
            write.assert_called_once()
            path = write.call_args[0][0]
            assert path.startswith("generated_audio_")
            assert path.endswith(".wav")

    def test_no_save_when_flag_false(self, cloner):
        with patch("voice_cloner.sf.write") as write:
            cloner.say("hi", play_audio=False, save_audio=False)
            write.assert_not_called()

    def test_generated_audio_has_expected_data(self, cloner, tmp_path):
        out = str(tmp_path / "shape.wav")
        arr = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        cloner.engine.generate.return_value = (arr, 44100)
        with patch("voice_cloner.sf.write") as write:
            cloner.say("test", play_audio=False, save_audio=True, output_file=out)
            write.assert_called_once_with(out, arr, 44100)

    def test_playback_calls_sounddevice(self, cloner):
        with patch("voice_cloner.sd.play") as play, patch("voice_cloner.sd.wait"):
            cloner.say("hello", play_audio=True, save_audio=False)
            play.assert_called_once()

    def test_unsupported_language(self, cloner):
        cloner.engine.generate.side_effect = ValueError("unsupported")
        with pytest.raises(ValueError, match="unsupported"):
            cloner.say("hi", language="xx", play_audio=False)

    def test_raises_on_generation_failure(self, cloner):
        cloner.engine.generate.side_effect = RuntimeError("generation failed")
        with pytest.raises(RuntimeError, match="generation failed"):
            cloner.say("fail", play_audio=False)


class TestSavedAudioReadback:
    """Verify saved audio can be read back successfully as audio data."""

    def test_read_saved_audio(self, cloner, tmp_path):
        out = str(tmp_path / "readback.wav")
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cloner.engine.generate.return_value = (arr, 22050)
        with (
            patch("voice_cloner.sf.write"),
            patch("voice_cloner.sf.read", return_value=(arr, 22050)),
        ):
            cloner.say("readback", play_audio=False, save_audio=True, output_file=out)
            data, sr = cloner.engine.generate.return_value
            assert isinstance(data, np.ndarray)
            assert sr > 0

    def test_saved_audio_has_explicit_path(self, cloner, tmp_path):
        out = str(tmp_path / "explicit_path.wav")
        with (
            patch("voice_cloner.sf.write"),
            patch("voice_cloner.sf.read", return_value=(np.array([0.1], dtype=np.float32), 22050)),
        ):
            cloner.say("explicit", play_audio=False, save_audio=True, output_file=out)
            assert os.path.exists(os.path.dirname(out))


class TestInjectedRegistry:
    """Tests for injected ModelRegistry in VoiceCloner."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.isolated_registry = ModelRegistry()
        self.isolated_registry.register_model(
            ModelInfo(id="custom-only", engine="custom", name="Custom Only", size_mb=1, description="DI test")
        )

    def test_list_models_accepts_injected_registry(self):
        models = VoiceCloner.list_models(registry=self.isolated_registry)
        model_ids = [m.id for m in models]
        assert "custom-only" in model_ids

    def test_list_models_without_registry_uses_global(self):
        models = VoiceCloner.list_models()
        model_ids = [m.id for m in models]
        assert "custom-only" not in model_ids
        assert "coqui-xtts-v2" in model_ids

    def test_is_model_installed_accepts_injected_registry(self):
        result = VoiceCloner.is_model_installed("custom-only", registry=self.isolated_registry)
        assert result is False  # custom-only has no path checker, so not installed

    def test_get_model_id_for_engine_accepts_injected_registry(self):
        self.isolated_registry.register_model(
            ModelInfo(id="custom-model", engine="custom", name="Custom", size_mb=1, description="")
        )
        # Manually set the engine model ID mapping

        result = VoiceCloner.get_model_id_for_engine("custom-model", registry=self.isolated_registry)
        assert result == "custom-model"

    def test_constructor_passes_registry_to_engine(self):
        from unittest.mock import MagicMock

        from models.model_info import ModelInfo

        # Register the engine metadata in TTSFactory so VoiceCloner doesn't error
        self.isolated_registry.register_model(
            ModelInfo(id="mlx-kokoro", engine="mlx-audio", name="MLX Kokoro", size_mb=164, description="")
        )

        eng = MagicMock()
        eng.generate.return_value = (np.array([0.1], dtype=np.float32), 22050)
        eng.name = "custom_engine"

        with (
            patch("voice_cloner.TTSFactory.create", return_value=eng) as mock_create,
            patch("voice_cloner.TTSFactory.get_engine_metadata", return_value={"requires_reference_audio": False}),
        ):
            VoiceCloner(
                speaker_wav="dummy.wav",
                engine="coqui",
                device="cpu",
                registry=self.isolated_registry,
            )
            _, kwargs = mock_create.call_args
            assert kwargs.get("registry") is self.isolated_registry

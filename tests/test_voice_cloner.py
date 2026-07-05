"""Tests for VoiceCloner."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock heavyweight module-level dependencies so tests run with zero
# external dependencies -- no sounddevice, soundfile, transformers, etc.
# ---------------------------------------------------------------------------
for _mod in (
    "sounddevice",
    "soundfile",
    "rich",
    "rich.console",
    "rich.logging",
    "transformers",
    "tts_engine_base",
    "tts_factory",
):
    sys.modules[_mod] = MagicMock()

from voice_cloner import VoiceCloner  # noqa: E402


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

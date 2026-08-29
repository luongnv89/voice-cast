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
    engine.MAX_CHUNK_CHARS = 0
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


class TestSayChunked:
    """VoiceCloner.say() chunked-synthesis tests."""

    @staticmethod
    def _long_text(sentences: int = 3) -> str:
        return " ".join(f"Sentence number {i} is here." for i in range(sentences))

    def test_no_chunk_size_calls_engine_once(self, cloner, tmp_path):
        """Without chunk_size the engine is called once with the original text."""
        out = str(tmp_path / "single.wav")
        text = self._long_text()
        with patch("voice_cloner.sf.write"):
            result = cloner.say(text, play_audio=False, save_audio=True, output_file=out)
        cloner.engine.generate.assert_called_once()
        assert cloner.engine.generate.call_args.kwargs["text"] == text
        assert result == out

    def test_short_text_with_chunk_size_is_unmodified(self, cloner):
        """Text shorter than chunk_size reaches the engine byte-for-byte unchanged."""
        text = "Hello   world.   Spacing  preserved."
        with patch("voice_cloner.sf.write"):
            cloner.say(text, play_audio=False, save_audio=False, chunk_size=500)
        cloner.engine.generate.assert_called_once()
        assert cloner.engine.generate.call_args.kwargs["text"] == text

    def test_long_text_synthesized_per_chunk(self, cloner):
        """Text longer than chunk_size is split and synthesized chunk by chunk."""
        from utils import split_into_chunks

        text = self._long_text()
        expected = split_into_chunks(text, 30)
        assert len(expected) > 1
        cloner.say(text, play_audio=False, save_audio=False, chunk_size=30)
        assert cloner.engine.generate.call_count == len(expected)
        called = [c.kwargs["text"] for c in cloner.engine.generate.call_args_list]
        assert called == expected

    @pytest.mark.parametrize(
        ("sample_rate", "silence_duration"),
        [(22050, 200), (44100, 500)],
    )
    def test_silence_sample_count_derived_from_milliseconds(self, cloner, sample_rate, silence_duration):
        """Inserted silence is silence_duration milliseconds, not raw samples."""
        from utils import split_into_chunks

        chunk_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cloner.engine.generate.return_value = (chunk_audio, sample_rate)
        text = self._long_text()
        n_chunks = len(split_into_chunks(text, 30))
        expected_silence = int(sample_rate * silence_duration / 1000)

        with patch("voice_cloner.sf.write") as write:
            cloner.say(
                text,
                play_audio=False,
                save_audio=True,
                output_file="out.wav",
                chunk_size=30,
                silence_duration=silence_duration,
            )
        written = write.call_args[0][1]
        assert len(written) == n_chunks * len(chunk_audio) + (n_chunks - 1) * expected_silence

    def test_no_trailing_or_leading_silence(self, cloner):
        """Silence is inserted between chunks only, never before or after."""
        chunk_audio = np.array([0.5, 0.5], dtype=np.float32)
        cloner.engine.generate.return_value = (chunk_audio, 1000)
        with patch("voice_cloner.sf.write") as write:
            cloner.say(
                self._long_text(),
                play_audio=False,
                save_audio=True,
                output_file="out.wav",
                chunk_size=30,
                silence_duration=10,
            )
        written = write.call_args[0][1]
        assert written[0] == 0.5
        assert written[-1] == 0.5

    def test_silence_dtype_matches_chunk_dtype(self, cloner):
        """The silence buffer uses the same dtype as the synthesized audio."""
        cloner.engine.generate.return_value = (np.array([0.1, 0.2], dtype=np.float64), 8000)
        with patch("voice_cloner.sf.write") as write:
            cloner.say(
                self._long_text(),
                play_audio=False,
                save_audio=True,
                output_file="out.wav",
                chunk_size=30,
            )
        assert write.call_args[0][1].dtype == np.float64

    def test_chunks_concatenated_in_order(self, cloner):
        """Chunk audio is concatenated in call order along axis 0."""
        cloner.engine.generate.side_effect = [
            (np.array([1.0], dtype=np.float32), 1000),
            (np.array([2.0], dtype=np.float32), 1000),
            (np.array([3.0], dtype=np.float32), 1000),
        ]
        with patch("voice_cloner.sf.write") as write:
            cloner.say(
                self._long_text(),
                play_audio=False,
                save_audio=True,
                output_file="out.wav",
                chunk_size=30,
                silence_duration=0,
            )
        written = write.call_args[0][1]
        assert written.ndim == 1
        assert list(written) == [1.0, 2.0, 3.0]

    def test_zero_silence_duration_adds_no_padding(self, cloner):
        """silence_duration=0 concatenates chunks with no padding at all."""
        from utils import split_into_chunks

        chunk_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        cloner.engine.generate.return_value = (chunk_audio, 22050)
        text = self._long_text()
        n_chunks = len(split_into_chunks(text, 30))
        with patch("voice_cloner.sf.write") as write:
            cloner.say(
                text,
                play_audio=False,
                save_audio=True,
                output_file="out.wav",
                chunk_size=30,
                silence_duration=0,
            )
        assert len(write.call_args[0][1]) == n_chunks * len(chunk_audio)

    def test_sample_rate_mismatch_raises(self, cloner):
        """Differing sample rates across chunks raise instead of concatenating."""
        cloner.engine.generate.side_effect = [
            (np.array([0.1], dtype=np.float32), 22050),
            (np.array([0.2], dtype=np.float32), 44100),
            (np.array([0.3], dtype=np.float32), 22050),
        ]
        with pytest.raises(RuntimeError, match="inconsistent sample rates"):
            cloner.say(self._long_text(), play_audio=False, save_audio=False, chunk_size=30)

    def test_whitespace_only_text_does_not_raise(self, cloner):
        """Whitespace-only text with chunk_size falls back to one engine call."""
        text = "        "
        cloner.say(text, play_audio=False, save_audio=False, chunk_size=2)
        cloner.engine.generate.assert_called_once()
        assert cloner.engine.generate.call_args.kwargs["text"] == text

    def test_returns_output_path_when_saving(self, cloner, tmp_path):
        """say() returns the path of the written file."""
        out = str(tmp_path / "returned.wav")
        with patch("voice_cloner.sf.write"):
            result = cloner.say(
                self._long_text(),
                play_audio=False,
                save_audio=True,
                output_file=out,
                chunk_size=30,
            )
        assert result == out

    def test_returns_none_on_play_only_path(self, cloner):
        """say() returns None when nothing is written, even with an output_file."""
        with patch("voice_cloner.sd.play"), patch("voice_cloner.sd.wait"), patch("voice_cloner.sf.write") as write:
            result = cloner.say("hello", play_audio=True, save_audio=False, output_file="unwritten.wav")
        write.assert_not_called()
        assert result is None

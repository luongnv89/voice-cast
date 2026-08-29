"""Integration tests for VoiceCloner's chunked synthesis contract."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_cloner import VoiceCloner


@pytest.fixture
def mocked_cloner():
    """Build a VoiceCloner with an engine that needs no reference audio."""
    engine = MagicMock()
    engine.name = "mock_engine"
    engine.requires_reference_audio = False
    engine.MAX_CHUNK_CHARS = 0
    engine.generate.return_value = (np.array([0.5], dtype=np.float32), 1000)
    return VoiceCloner(speaker_wav="", engine=engine), engine


class TestVoiceClonerChunking:
    """Verify the public generate() chunking and silence contract."""

    def test_generate_synthesizes_chunks_and_writes_padded_output(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        text = "First source sentence. Second source sentence."
        chunks = ["First source sentence.", "Second source sentence."]
        engine.generate.side_effect = [
            (np.array([1.0, 2.0], dtype=np.float32), 1000),
            (np.array([3.0], dtype=np.float32), 1000),
        ]
        output_file = str(tmp_path / "chunked.wav")

        with (
            patch("voice_cloner.split_into_chunks", return_value=chunks) as split,
            patch("voice_cloner.sf.write") as write,
        ):
            result = cloner.generate(text, chunk_size=32, output_file=output_file)

        split.assert_called_once_with(text, 32)
        assert [call.kwargs["text"] for call in engine.generate.call_args_list] == chunks
        assert all(call.kwargs["chunk_size"] == 32 for call in engine.generate.call_args_list)
        assert all("silence_duration" not in call.kwargs for call in engine.generate.call_args_list)
        expected = np.concatenate(
            [
                np.array([1.0, 2.0], dtype=np.float32),
                np.zeros(200, dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ]
        )
        np.testing.assert_array_equal(write.call_args.args[1], expected)
        assert write.call_args.args[0] == output_file
        assert write.call_args.args[2] == 1000
        assert result == output_file

    def test_generate_reports_each_chunk_before_engine_synthesis(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        chunks = ["One.", "Two.", "Three."]
        events = []

        def report(current, total):
            events.append(("progress", current, total))

        def synthesize(*, text, **_kwargs):
            events.append(("engine", text))
            return np.array([1.0], dtype=np.float32), 1000

        engine.generate.side_effect = synthesize
        with (
            patch("voice_cloner.split_into_chunks", return_value=chunks),
            patch("voice_cloner.sf.write"),
            patch("voice_cloner.console.status") as status,
        ):
            cloner.generate(
                "A long source text",
                chunk_size=4,
                output_file=str(tmp_path / "progress.wav"),
                chunk_progress_callback=report,
            )

        assert events == [
            ("progress", 1, 3),
            ("engine", "One."),
            ("progress", 2, 3),
            ("engine", "Two."),
            ("progress", 3, 3),
            ("engine", "Three."),
        ]
        status.assert_not_called()
        assert all("chunk_progress_callback" not in call.kwargs for call in engine.generate.call_args_list)

    def test_generate_reports_single_short_text_as_one_chunk(self, mocked_cloner, tmp_path):
        cloner, _engine = mocked_cloner
        progress = MagicMock()

        with patch("voice_cloner.sf.write"):
            cloner.generate(
                "Short text.",
                chunk_size=100,
                output_file=str(tmp_path / "single.wav"),
                chunk_progress_callback=progress,
            )

        progress.assert_called_once_with(1, 1)

    def test_generate_stops_progress_when_chunk_synthesis_fails(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        engine.generate.side_effect = [
            (np.array([1.0], dtype=np.float32), 1000),
            RuntimeError("synthesis failed"),
        ]
        progress = MagicMock()

        with (
            patch("voice_cloner.split_into_chunks", return_value=["One.", "Two.", "Three."]),
            patch("voice_cloner.sf.write"),
            pytest.raises(RuntimeError, match="synthesis failed"),
        ):
            cloner.generate(
                "A long source text",
                chunk_size=4,
                output_file=str(tmp_path / "failed.wav"),
                chunk_progress_callback=progress,
            )

        assert [item.args for item in progress.call_args_list] == [(1, 3), (2, 3)]

    def test_generate_keeps_short_text_on_single_engine_call(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        text = "Short text."
        output_file = str(tmp_path / "short.wav")

        with (
            patch("voice_cloner.split_into_chunks") as split,
            patch("voice_cloner.sf.write"),
        ):
            result = cloner.generate(text, chunk_size=100, output_file=output_file)

        split.assert_not_called()
        engine.generate.assert_called_once_with(text=text, language="en", chunk_size=100)
        assert result == output_file

    def test_generate_supports_zero_silence_duration(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        engine.generate.side_effect = [
            (np.array([1.0], dtype=np.float32), 1000),
            (np.array([2.0], dtype=np.float32), 1000),
        ]
        output_file = str(tmp_path / "no-silence.wav")

        with (
            patch("voice_cloner.split_into_chunks", return_value=["One.", "Two."]),
            patch("voice_cloner.sf.write") as write,
        ):
            cloner.generate("A long source text", chunk_size=4, silence_duration=0, output_file=output_file)

        np.testing.assert_array_equal(write.call_args.args[1], np.array([1.0, 2.0], dtype=np.float32))

    @pytest.mark.parametrize(
        ("chunk_size", "expected_exception"),
        [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
    )
    def test_generate_rejects_invalid_chunk_size(self, mocked_cloner, chunk_size, expected_exception):
        cloner, engine = mocked_cloner

        with pytest.raises(expected_exception, match="chunk_size"):
            cloner.generate("text", chunk_size=chunk_size)

        engine.generate.assert_not_called()

    @pytest.mark.parametrize(
        ("silence_duration", "expected_exception"),
        [(-1, ValueError), (True, TypeError), (1.5, TypeError)],
    )
    def test_generate_rejects_invalid_silence_duration(self, mocked_cloner, silence_duration, expected_exception):
        cloner, engine = mocked_cloner

        with pytest.raises(expected_exception, match="silence_duration"):
            cloner.generate("text", silence_duration=silence_duration)

        engine.generate.assert_not_called()

    def test_generate_preserves_stereo_shape_for_silence(self, mocked_cloner, tmp_path):
        cloner, engine = mocked_cloner
        engine.generate.side_effect = [
            (np.array([[1.0, 2.0]], dtype=np.float32), 1000),
            (np.array([[3.0, 4.0]], dtype=np.float32), 1000),
        ]
        output_file = str(tmp_path / "stereo.wav")

        with (
            patch("voice_cloner.split_into_chunks", return_value=["One.", "Two."]),
            patch("voice_cloner.sf.write") as write,
        ):
            cloner.generate("A long source text", chunk_size=4, silence_duration=1, output_file=output_file)

        written = write.call_args.args[1]
        assert written.shape == (3, 2)
        np.testing.assert_array_equal(written[1:2], np.zeros((1, 2), dtype=np.float32))

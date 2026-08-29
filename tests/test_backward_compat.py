"""Backward compatibility: short texts must be bit-identical to pre-chunking behaviour.

Issue #145 requires that texts under the effective chunk limit:
- produce identical samples,
- emit no silence padding,
- and for single-chunk synthesis, progress handling remains correct.

These tests use a deterministic mocked engine so they never need ML deps.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_cloner import VoiceCloner


@pytest.fixture
def mocked_cloner():
    engine = MagicMock()
    engine.name = "mock"
    engine.requires_reference_audio = False
    engine.MAX_CHUNK_CHARS = 0
    engine.generate.return_value = (np.array([0.42, 0.84], dtype=np.float32), 22050)
    return VoiceCloner(speaker_wav="", engine=engine), engine


def test_short_text_single_call_bit_identical(mocked_cloner, tmp_path):
    """Short text reaches the engine in one call and writes identical samples."""
    cloner, engine = mocked_cloner
    short = "Hello world."
    out = str(tmp_path / "short.wav")
    # Deterministic audio
    wanted = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    engine.generate.return_value = (wanted, 44100)

    with patch("voice_cloner.split_into_chunks") as split, patch("voice_cloner.sf.write") as write:
        cloner.generate(short, chunk_size=100, output_file=out)

    split.assert_not_called()
    engine.generate.assert_called_once_with(text=short, language="en", chunk_size=100)
    assert write.call_args[0][0] == out
    np.testing.assert_array_equal(write.call_args[0][1], wanted)
    assert write.call_args[0][2] == 44100


def test_short_text_no_extra_progress_events(mocked_cloner, tmp_path):
    """Single-chunk synthesis reports exactly one progress event (1,1) and no silence."""
    cloner, engine = mocked_cloner
    engine.generate.return_value = (np.array([1.0], dtype=np.float32), 1000)
    progress = MagicMock()

    with patch("voice_cloner.split_into_chunks") as split, patch("voice_cloner.sf.write") as write:
        cloner.generate(
            "Short",
            chunk_size=100,
            output_file=str(tmp_path / "p.wav"),
            chunk_progress_callback=progress,
            silence_duration=500,
        )

    split.assert_not_called()
    progress.assert_called_once_with(1, 1)
    # Silence should not affect single-chunk output
    written = write.call_args[0][1]
    assert len(written) == 1
    assert written[0] == 1.0


def test_no_silence_for_single_chunk_with_nonzero_duration(mocked_cloner, tmp_path):
    """Silence padding is omitted when only one chunk exists."""
    cloner, engine = mocked_cloner
    engine.generate.return_value = (np.array([9.0, 8.0], dtype=np.float32), 8000)

    with patch("voice_cloner.sf.write") as write:
        cloner.generate("Hi", chunk_size=200, silence_duration=1000, output_file=str(tmp_path / "single.wav"))

    # For a single chunk, no silence samples are inserted regardless of duration
    assert len(write.call_args[0][1]) == 2

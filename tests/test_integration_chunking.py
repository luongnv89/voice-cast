"""Integration coverage for engine chunk configuration and Coqui synthesis."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from engines.audio8_engine import Audio8Engine
from engines.chatterbox_engine import ChatterboxEngine
from engines.coqui_engine import CoquiEngine
from engines.mlx_audio_engine import MlxAudioEngine
from utils import split_into_chunks
from voice_cloner import VoiceCloner

ENGINE_LIMITS = [
    (CoquiEngine, 240),
    (ChatterboxEngine, 100),
    (MlxAudioEngine, 200),
    (Audio8Engine, 200),
]


class _FakeSynthesizer:
    output_sample_rate = 16000

    def __init__(self):
        self.paths = []

    def save_wav(self, *, wav, path):
        self.paths.append(path)
        if not isinstance(path, io.BytesIO):
            raise AssertionError("Coqui output must be serialized to an in-memory buffer")
        sf.write(path, np.asarray(wav), self.output_sample_rate, format="WAV", subtype="PCM_16")


class _FakeCoquiTTS:
    def __init__(self):
        self.synthesizer = _FakeSynthesizer()
        self.tts_calls = []
        self.tts_to_file_calls = 0

    def tts(self, *, text, speaker_wav, language, gpt_cond_len, temperature):
        self.tts_calls.append(
            {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "gpt_cond_len": gpt_cond_len,
                "temperature": temperature,
            }
        )
        return np.array([0.1, 0.2], dtype=np.float32)

    def tts_to_file(self, **_kwargs):
        self.tts_to_file_calls += 1
        raise AssertionError("Coqui generation must not use tts_to_file")


@pytest.mark.parametrize(("engine_class", "expected_limit"), ENGINE_LIMITS)
def test_all_concrete_engines_declare_positive_chunk_limits(engine_class, expected_limit):
    assert expected_limit == engine_class.MAX_CHUNK_CHARS
    assert isinstance(engine_class.MAX_CHUNK_CHARS, int)
    assert engine_class.MAX_CHUNK_CHARS > 0


def test_chunk_size_override_runs_coqui_chunking_in_memory(tmp_path):
    speaker_file = tmp_path / "speaker.wav"
    speaker_file.write_bytes(b"mock speaker reference")
    output_file = tmp_path / "chunked.wav"
    fake_tts = _FakeCoquiTTS()
    engine = CoquiEngine(speaker_wav=str(speaker_file), device="cpu", registry=MagicMock())
    engine._tts = fake_tts
    chunks = split_into_chunks("One. Two. Three.", max_chars=8)
    cloner = VoiceCloner(speaker_wav=str(speaker_file), engine=engine)

    with patch.object(engine, "generate", wraps=engine.generate) as generate:
        result = cloner.generate(
            "One. Two. Three.",
            chunk_size=8,
            silence_duration=0,
            output_file=str(output_file),
        )

    assert result == str(output_file)
    assert [call.kwargs["text"] for call in generate.call_args_list] == chunks
    assert [call.kwargs["chunk_size"] for call in generate.call_args_list] == [8] * len(chunks)
    assert [call["text"] for call in fake_tts.tts_calls] == chunks
    assert fake_tts.tts_to_file_calls == 0
    assert len(fake_tts.synthesizer.paths) == len(chunks)
    assert all(isinstance(path, io.BytesIO) for path in fake_tts.synthesizer.paths)

    audio_data, sample_rate = sf.read(output_file)
    assert audio_data.ndim == 1
    assert audio_data.size == len(chunks) * 2
    assert sample_rate == fake_tts.synthesizer.output_sample_rate
    assert {path.name for path in tmp_path.iterdir()} == {speaker_file.name, output_file.name}

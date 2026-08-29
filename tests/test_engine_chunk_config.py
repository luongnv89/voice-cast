"""Tests for engine chunk limits and VoiceCloner propagation."""

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engines.audio8_engine import Audio8Engine
from engines.chatterbox_engine import ChatterboxEngine
from engines.coqui_engine import CoquiEngine
from engines.mlx_audio_engine import MlxAudioEngine
from tts_engine_base import TTSEngineBase
from voice_cloner import VoiceCloner

ENGINE_LIMITS = [
    (CoquiEngine, 240),
    (ChatterboxEngine, 100),
    (MlxAudioEngine, 200),
    (Audio8Engine, 200),
]


class _LegacyEngine:
    name = "legacy"
    requires_reference_audio = False

    def __init__(self):
        self.calls = []

    def generate(self, text, language="en"):
        self.calls.append((text, language))
        return np.array([0.5], dtype=np.float32), 1000


class _KwargsEngine:
    name = "kwargs"
    requires_reference_audio = False

    def __init__(self):
        self.calls = []

    def generate(self, text, language="en", **kwargs):
        self.calls.append((text, language, kwargs))
        return np.array([0.5], dtype=np.float32), 1000


@pytest.fixture
def chunking_cloner():
    engine = MagicMock()
    engine.name = "mock_engine"
    engine.requires_reference_audio = False
    engine.MAX_CHUNK_CHARS = 10
    engine.generate.return_value = (np.array([0.5], dtype=np.float32), 1000)
    return VoiceCloner(speaker_wav="", engine=engine), engine


def test_base_engine_declares_no_default_chunk_limit():
    assert TTSEngineBase.MAX_CHUNK_CHARS == 0


@pytest.mark.parametrize(("engine_class", "expected_limit"), ENGINE_LIMITS)
def test_concrete_engines_declare_model_chunk_limits(engine_class, expected_limit):
    assert expected_limit == engine_class.MAX_CHUNK_CHARS
    assert isinstance(engine_class.MAX_CHUNK_CHARS, int)


@pytest.mark.parametrize(("engine_class", "_expected_limit"), ENGINE_LIMITS)
def test_concrete_generate_methods_accept_chunk_size(engine_class, _expected_limit):
    parameter = inspect.signature(engine_class.generate).parameters["chunk_size"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_explicit_override_preserves_legacy_custom_engine(tmp_path):
    engine = _LegacyEngine()
    cloner = VoiceCloner(speaker_wav="", engine=engine)
    output_file = str(tmp_path / "legacy.wav")

    with (
        patch("voice_cloner.split_into_chunks", return_value=["One.", "Two."]),
        patch("voice_cloner.sf.write"),
    ):
        cloner.generate("One. Two.", chunk_size=5, output_file=output_file)

    assert engine.calls == [("One.", "en"), ("Two.", "en")]


def test_explicit_override_reaches_custom_kwargs_engine(tmp_path):
    engine = _KwargsEngine()
    cloner = VoiceCloner(speaker_wav="", engine=engine)
    output_file = str(tmp_path / "kwargs.wav")

    with (
        patch("voice_cloner.split_into_chunks", return_value=["One.", "Two."]),
        patch("voice_cloner.sf.write"),
    ):
        cloner.generate("One. Two.", chunk_size=5, output_file=output_file)

    assert [call[2]["chunk_size"] for call in engine.calls] == [5, 5]


def test_generate_uses_engine_default_and_forwards_it(chunking_cloner, tmp_path):
    cloner, engine = chunking_cloner
    text = "First sentence. Second sentence."
    chunks = ["First sentence.", "Second sentence."]
    output_file = str(tmp_path / "default-limit.wav")

    with (
        patch("voice_cloner.split_into_chunks", return_value=chunks) as split,
        patch("voice_cloner.sf.write"),
    ):
        result = cloner.generate(text, output_file=output_file)

    split.assert_called_once_with(text, 10)
    assert [call.kwargs["text"] for call in engine.generate.call_args_list] == chunks
    assert all(call.kwargs["chunk_size"] == 10 for call in engine.generate.call_args_list)
    assert result == output_file


def test_generate_override_replaces_engine_default(chunking_cloner, tmp_path):
    cloner, engine = chunking_cloner
    text = "First sentence. Second sentence."
    chunks = ["First sentence.", "Second sentence."]
    output_file = str(tmp_path / "override.wav")

    with (
        patch("voice_cloner.split_into_chunks", return_value=chunks) as split,
        patch("voice_cloner.sf.write"),
    ):
        cloner.generate(text, chunk_size=20, output_file=output_file)

    split.assert_called_once_with(text, 20)
    assert all(call.kwargs["chunk_size"] == 20 for call in engine.generate.call_args_list)


def test_short_text_still_forwards_effective_default(chunking_cloner, tmp_path):
    cloner, engine = chunking_cloner
    output_file = str(tmp_path / "short.wav")

    with (
        patch("voice_cloner.split_into_chunks") as split,
        patch("voice_cloner.sf.write"),
    ):
        cloner.generate("Short", output_file=output_file)

    split.assert_not_called()
    engine.generate.assert_called_once_with(text="Short", language="en", chunk_size=10)


def test_zero_engine_limit_disables_automatic_chunking(chunking_cloner, tmp_path):
    cloner, engine = chunking_cloner
    engine.MAX_CHUNK_CHARS = 0
    text = "A long text that remains one engine call."
    output_file = str(tmp_path / "unlimited.wav")

    with (
        patch("voice_cloner.split_into_chunks") as split,
        patch("voice_cloner.sf.write"),
    ):
        cloner.generate(text, output_file=output_file)

    split.assert_not_called()
    engine.generate.assert_called_once_with(text=text, language="en")

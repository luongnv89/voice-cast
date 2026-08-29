"""Regression tests for Coqui's scoped PyTorch compatibility patch."""

import io
import sys
import types
from threading import Thread
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import soundfile as sf


def _install_fake_tts(monkeypatch, constructor):
    api_module = types.ModuleType("TTS.api")
    api_module.TTS = constructor
    monkeypatch.setitem(sys.modules, "TTS.api", api_module)


def _make_engine(coqui_engine, registry):
    return coqui_engine.CoquiEngine(speaker_wav="voice.wav", device="cpu", registry=registry)


class _FakeSynthesizer:
    output_sample_rate = 16000

    def __init__(self):
        self.paths = []

    def save_wav(self, *, wav, path):
        self.paths.append(path)
        if not isinstance(path, io.BytesIO):
            raise AssertionError("Coqui output must be serialized to an in-memory buffer")
        sf.write(path, np.asarray(wav), self.output_sample_rate, format="WAV", subtype="PCM_16")


class _FakeTTS:
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
        return np.array([[0.25, -0.5], [0.5, -0.25]], dtype=np.float32)

    def tts_to_file(self, **_kwargs):
        self.tts_to_file_calls += 1
        raise AssertionError("Coqui generation must not use tts_to_file")


class TestCoquiInMemorySynthesis:
    def test_generate_returns_valid_audio_without_creating_a_file(self, tmp_path):
        from engines import coqui_engine

        fake_tts = _FakeTTS()
        engine = coqui_engine.CoquiEngine(
            speaker_wav=str(tmp_path / "voice.wav"),
            device="cpu",
            registry=MagicMock(),
        )
        engine._tts = fake_tts

        audio_data, sample_rate = engine.generate(
            "hello",
            language="fr",
            temperature=0.4,
            gpt_cond_len=64,
        )

        assert fake_tts.tts_calls == [
            {
                "text": "hello",
                "speaker_wav": str(tmp_path / "voice.wav"),
                "language": "fr",
                "gpt_cond_len": 64,
                "temperature": 0.4,
            }
        ]
        assert fake_tts.tts_to_file_calls == 0
        assert len(fake_tts.synthesizer.paths) == 1
        assert isinstance(fake_tts.synthesizer.paths[0], io.BytesIO)
        assert audio_data.dtype == np.float32
        assert audio_data.shape == (2,)
        assert sample_rate == fake_tts.synthesizer.output_sample_rate
        np.testing.assert_allclose(audio_data, np.array([-0.125, 0.125], dtype=np.float32), atol=1e-4)
        assert list(tmp_path.iterdir()) == []


class TestCoquiTorchLoadCompatibility:
    def test_scoped_load_overrides_explicit_weights_only(self, monkeypatch):
        from engines import coqui_engine

        original_load = MagicMock(return_value=object())
        monkeypatch.setattr(coqui_engine.torch, "load", original_load)

        with coqui_engine._coqui_torch_load_compatibility():
            unrelated_thread = Thread(target=lambda: coqui_engine.torch.load("unrelated.pth"))
            unrelated_thread.start()
            unrelated_thread.join()

            coqui_engine.torch.load("legacy.pth")
            coqui_engine.torch.load("safe.pth", weights_only=True)

        assert original_load.call_args_list == [
            call("unrelated.pth"),
            call("legacy.pth", weights_only=False),
            call("safe.pth", weights_only=False),
        ]

    def test_torch_load_restored_after_model_construction(self, monkeypatch):
        from engines import coqui_engine

        original_load = MagicMock(return_value=object())
        monkeypatch.setattr(coqui_engine.torch, "load", original_load)

        def fake_tts(**_kwargs):
            assert coqui_engine.torch.load is not original_load
            coqui_engine.torch.load("checkpoint.pth")
            return object()

        _install_fake_tts(monkeypatch, fake_tts)
        registry = MagicMock()
        registry.is_installed.return_value = True

        engine = _make_engine(coqui_engine, registry)
        assert engine.tts is not None
        assert coqui_engine.torch.load is original_load
        original_load.assert_called_once_with("checkpoint.pth", weights_only=False)

    def test_torch_load_restored_when_model_construction_raises(self, monkeypatch):
        from engines import coqui_engine

        original_load = MagicMock(return_value=object())
        monkeypatch.setattr(coqui_engine.torch, "load", original_load)

        def failing_tts(**_kwargs):
            assert coqui_engine.torch.load is not original_load
            coqui_engine.torch.load("checkpoint.pth")
            raise RuntimeError("model construction failed")

        _install_fake_tts(monkeypatch, failing_tts)
        registry = MagicMock()
        registry.is_installed.return_value = True

        engine = _make_engine(coqui_engine, registry)
        with pytest.raises(RuntimeError, match="model construction failed"):
            _ = engine.tts

        assert coqui_engine.torch.load is original_load
        original_load.assert_called_once_with("checkpoint.pth", weights_only=False)

        # The compatibility lock must remain reusable after a failed load.
        with coqui_engine._coqui_torch_load_compatibility():
            coqui_engine.torch.load("after-error.pth")
        assert original_load.call_args_list == [
            call("checkpoint.pth", weights_only=False),
            call("after-error.pth", weights_only=False),
        ]

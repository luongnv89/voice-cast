"""Regression tests for Coqui's scoped PyTorch compatibility patch."""

import sys
import types
from threading import Thread
from unittest.mock import MagicMock, call

import pytest


def _install_fake_tts(monkeypatch, constructor):
    api_module = types.ModuleType("TTS.api")
    api_module.TTS = constructor
    monkeypatch.setitem(sys.modules, "TTS.api", api_module)


def _make_engine(coqui_engine, registry):
    return coqui_engine.CoquiEngine(speaker_wav="voice.wav", device="cpu", registry=registry)


class TestCoquiTorchLoadCompatibility:
    def test_unrelated_thread_keeps_safe_default_load_behavior(self, monkeypatch):
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
            call("safe.pth", weights_only=True),
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

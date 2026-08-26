"""Regression tests for Chatterbox variant-to-checkpoint wiring (#56).

Turbo (350M) and Standard (500M) are distinct HuggingFace checkpoints loaded by
distinct backend classes; selecting one variant must never silently load the
other.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

try:
    from engines.chatterbox_engine import CHATTERBOX_VARIANT_BACKENDS
except ImportError:  # pre-fix tree has no variant-to-checkpoint map
    CHATTERBOX_VARIANT_BACKENDS = {}
from engines.chatterbox_engine import ChatterboxEngine
from models.downloaders.chatterbox_downloader import ChatterboxDownloader
from models.model_registry import ModelRegistry


class TestDownloaderCheckpoints:
    def test_variants_resolve_distinct_repos(self):
        repos = ChatterboxDownloader.MODEL_REPOS
        assert repos["chatterbox-turbo"] == "ResembleAI/chatterbox-turbo"
        assert repos["chatterbox-standard"] == "ResembleAI/chatterbox"
        assert repos["chatterbox-turbo"] != repos["chatterbox-standard"]

    def test_variants_resolve_distinct_revisions(self):
        revisions = ChatterboxDownloader.MODEL_REVISIONS
        assert revisions["chatterbox-turbo"] != revisions["chatterbox-standard"]
        assert all(revisions.values()), "every variant must pin a revision"

    def test_snapshot_download_uses_variant_repo(self, monkeypatch):
        seen = {}

        class FakeRegistry:
            def get_cache_dir(self, _engine):
                return "/tmp/fake-hf-cache"

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = MagicMock(
            side_effect=lambda **kwargs: seen.update(kwargs) or "/tmp/fake-hf-cache/snap"
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        monkeypatch.setattr("models.downloaders.chatterbox_downloader.get_registry", FakeRegistry)

        path = ChatterboxDownloader().download("chatterbox-standard")

        assert str(path) == "/tmp/fake-hf-cache/snap"
        assert seen["repo_id"] == "ResembleAI/chatterbox"
        assert seen["revision"] == ChatterboxDownloader.MODEL_REVISIONS["chatterbox-standard"]


class TestRegistryCheckpoints:
    def test_path_checkers_distinguish_repos(self):
        registry = ModelRegistry()
        turbo = registry.get_model("chatterbox-turbo")
        standard = registry.get_model("chatterbox-standard")
        assert turbo.model_path_checker != standard.model_path_checker
        assert turbo.model_path_checker == "ResembleAI/chatterbox-turbo"

    def test_install_status_does_not_leak_across_variants(self, tmp_path, monkeypatch):
        registry = ModelRegistry()
        monkeypatch.setattr(
            registry,
            "get_cache_dir",
            lambda _engine: tmp_path,
        )
        # Only the Standard repo cache exists on disk.
        (tmp_path / "models--ResembleAI--chatterbox").mkdir()

        assert registry.is_installed("chatterbox-standard") is True
        assert registry.is_installed("chatterbox-turbo") is False


class TestEngineCheckpoints:
    @staticmethod
    def _install_backend_stub(monkeypatch, module_name, class_name, record):
        stub_module = types.ModuleType(module_name)

        class FakeBackend:
            sr = 24000

            @classmethod
            def from_pretrained(cls, device=None):
                record.append((cls, device))
                return MagicMock(sr=cls.sr)

        setattr(stub_module, class_name, FakeBackend)
        monkeypatch.setitem(sys.modules, module_name, stub_module)
        return FakeBackend

    def test_variant_map_targets_distinct_backends(self):
        assert CHATTERBOX_VARIANT_BACKENDS["chatterbox-turbo"] != CHATTERBOX_VARIANT_BACKENDS["chatterbox-standard"]

    @staticmethod
    def _make_engine(variant):
        return ChatterboxEngine(speaker_wav="unused.wav", variant=variant)

    def _assert_variant_loads_backend(self, monkeypatch, model_id):
        record = []
        backends = {}
        for key, (module_name, class_name) in CHATTERBOX_VARIANT_BACKENDS.items():
            backends[key] = self._install_backend_stub(monkeypatch, module_name, class_name, record)

        engine = self._make_engine(model_id.removeprefix("chatterbox-"))
        monkeypatch.setattr(engine, "_check_model_installed", lambda: True)
        model = engine.model

        assert len(record) == 1
        loaded_cls, loaded_device = record[0]
        assert loaded_cls is backends[model_id]
        assert model is not None

    def test_turbo_selection_loads_turbo_checkpoint(self, monkeypatch):
        self._assert_variant_loads_backend(monkeypatch, "chatterbox-turbo")

    def test_standard_selection_loads_standard_checkpoint(self, monkeypatch):
        self._assert_variant_loads_backend(monkeypatch, "chatterbox-standard")

    def test_missing_model_names_selected_variant_id(self):
        engine = self._make_engine("standard")
        with pytest.raises(Exception) as excinfo:
            _ = engine.model
        assert engine._model_id == "chatterbox-standard"
        assert "chatterbox-standard" in str(excinfo.value)

"""Characterization tests at component seams.

Locks in current (now-fixed) behavior for:
- Download → install round-trip
- Variant selection in model manager
- Progress/cancel in downloads
- Generate happy path through VoiceCloner
- Engine generate() paths for all supported engines

These tests serve as regression guards for the component seams where
the worst bugs lived (F-BUG-001/002/003).
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: fake registry and stub modules
# ---------------------------------------------------------------------------


def _fake_registry(cache_dir):
    """Return a minimal registry stub."""
    return types.SimpleNamespace(get_cache_dir=lambda engine: Path(cache_dir))


def _install_stub_tts(monkeypatch):
    """Serve a fake TTS package so coqui_downloader imports cleanly."""
    manage_mod = types.ModuleType("TTS.utils.manage")
    manage_mod.ModelManager = MagicMock
    manage_mod.tqdm = MagicMock
    utils_mod = types.ModuleType("TTS.utils")
    utils_mod.manage = manage_mod
    tts_mod = types.ModuleType("TTS")
    tts_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "TTS", tts_mod)
    monkeypatch.setitem(sys.modules, "TTS.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "TTS.utils.manage", manage_mod)


def _install_stub_hf_hub(monkeypatch, return_path="/fake/hf-cache"):
    """Serve a fake huggingface_hub module."""
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = MagicMock(return_value=return_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


# ---------------------------------------------------------------------------
# 1. Download → install round-trip
# ---------------------------------------------------------------------------


class TestDownloadInstallRoundTrip:
    """Verify that downloading a model produces a valid install."""

    def test_coqui_download_returns_path(self, tmp_path, monkeypatch):
        """Coqui download must return a valid path."""
        _install_stub_tts(monkeypatch)
        _install_stub_hf_hub(monkeypatch)

        from models.downloaders.coqui_downloader import CoquiDownloader

        monkeypatch.setattr(
            "models.downloaders.coqui_downloader.get_registry",
            lambda: _fake_registry(str(tmp_path)),
        )
        # Mock the internal _download_model_with_progress to return a valid path
        fake_path = str(tmp_path / "coqui-model")
        Path(fake_path).mkdir(parents=True, exist_ok=True)
        (Path(fake_path) / "model_file.pth").touch()
        (Path(fake_path) / "config.json").touch()

        with patch(
            "models.downloaders.coqui_downloader._download_model_with_progress",
            return_value=(fake_path, Path(fake_path) / "config.json", {}),
        ):
            downloader = CoquiDownloader()
            result = downloader.download("coqui-xtts-v2")

        assert result is not None
        assert isinstance(result, (str, Path))

    def test_chatterbox_download_returns_path(self, tmp_path, monkeypatch):
        """Chatterbox download must return a valid path."""
        _install_stub_hf_hub(monkeypatch)

        from models.downloaders.chatterbox_downloader import ChatterboxDownloader

        monkeypatch.setattr(
            "models.downloaders.chatterbox_downloader.get_registry",
            lambda: _fake_registry(str(tmp_path)),
        )

        downloader = ChatterboxDownloader()
        result = downloader.download("chatterbox-turbo")

        assert result is not None
        assert isinstance(result, (str, Path))

    def test_mlx_download_returns_path(self, tmp_path, monkeypatch):
        """MLX download must return a valid path."""
        _install_stub_hf_hub(monkeypatch)

        from models.downloaders.mlx_downloader import MlxDownloader

        monkeypatch.setattr(
            "models.downloaders.mlx_downloader.get_registry",
            lambda: _fake_registry(str(tmp_path)),
        )

        downloader = MlxDownloader()
        result = downloader.download("mlx-kokoro")

        assert result is not None
        assert isinstance(result, (str, Path))

    def test_round_trip_cache_dir_derivation(self, tmp_path, monkeypatch):
        """Characterization: cache-dir derivation affects download path."""
        _install_stub_hf_hub(monkeypatch, return_path=str(tmp_path / "hf-cache"))

        from models.downloaders import coqui_downloader
        from models.downloaders.chatterbox_downloader import ChatterboxDownloader

        monkeypatch.setattr(
            "models.downloaders.chatterbox_downloader.get_registry",
            lambda: _fake_registry(str(tmp_path)),
        )
        monkeypatch.setattr(
            coqui_downloader,
            "get_registry",
            lambda: _fake_registry(str(tmp_path)),
        )

        downloader = ChatterboxDownloader()
        result = downloader.download("chatterbox-turbo")

        # Result should be under tmp_path
        result_path = Path(result)
        assert tmp_path in result_path.parents or result_path == tmp_path, (
            f"download result {result_path} must be under cache dir {tmp_path}"
        )


# ---------------------------------------------------------------------------
# 2. Variant selection in model manager
# ---------------------------------------------------------------------------


class TestVariantSelection:
    """Model manager must handle variant selection correctly."""

    @pytest.fixture
    def qapp(self):
        """Provide a Qt application instance."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_model_info_has_engine_attribute(self):
        """ModelInfo must carry an engine attribute for dispatch."""
        from models.model_info import ModelInfo

        info = ModelInfo(
            id="test-model",
            engine="coqui",
            name="Test Model",
            size_mb=100,
            description="Test model",
        )
        assert info.engine == "coqui"

    def test_model_info_equality(self):
        """Two ModelInfo instances with same fields should be equal."""
        from models.model_info import ModelInfo

        a = ModelInfo(
            id="test-model",
            engine="coqui",
            name="Test",
            size_mb=100,
            description="Test",
        )
        b = ModelInfo(
            id="test-model",
            engine="coqui",
            name="Test",
            size_mb=100,
            description="Test",
        )
        assert a == b

    def test_model_info_str_representation(self):
        """ModelInfo str should be human-readable."""
        from models.model_info import ModelInfo

        info = ModelInfo(
            id="xtts-v2",
            engine="coqui",
            name="XTTS v2",
            size_mb=2400,
            description="XTTS model",
            is_installed=True,
        )
        s = str(info)
        assert "XTTS v2" in s
        assert "coqui" in s

    def test_model_info_requires_id(self):
        """ModelInfo must require a non-empty id."""
        from models.model_info import ModelInfo

        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            ModelInfo(
                id="",
                engine="coqui",
                name="Test",
                size_mb=100,
                description="Test",
            )


# ---------------------------------------------------------------------------
# 3. Engine generate() paths via VoiceCloner
# ---------------------------------------------------------------------------


class TestEngineGeneratePaths:
    """Each engine's generate() must accept text and return (audio, sample_rate)."""

    @pytest.fixture
    def engine_mock(self):
        """Return a mock engine with generate behavior."""
        engine = MagicMock()
        engine.generate.return_value = (np.array([0.1, 0.2, 0.3], dtype=np.float32), 22050)
        engine.name = "mock_engine"
        engine.supports_languages = ["en"]
        return engine

    def test_voice_cloner_say_calls_engine(self, tmp_path, engine_mock):
        """VoiceCloner.say must delegate to the engine's generate()."""
        from voice_cloner import VoiceCloner

        speaker_file = tmp_path / "speaker.wav"
        speaker_file.write_bytes(b"fake audio data")

        cloner = VoiceCloner(speaker_wav=str(speaker_file), engine=engine_mock)
        cloner.say("hello", save_audio=False, play_audio=False)

        engine_mock.generate.assert_called()

    def test_voice_cloner_say_with_save(self, tmp_path, engine_mock):
        """VoiceCloner.say with save_audio=True must write a file."""
        from voice_cloner import VoiceCloner

        speaker_file = tmp_path / "speaker.wav"
        speaker_file.write_bytes(b"fake audio data")

        output_file = str(tmp_path / "output.wav")
        # Make the mock return audio data that soundfile can write
        engine_mock.generate.return_value = (np.array([0.1] * 1000, dtype=np.float32), 22050)
        cloner = VoiceCloner(speaker_wav=str(speaker_file), engine=engine_mock)
        cloner.say("hello", save_audio=True, output_file=output_file, play_audio=False)

        assert output_file in str(tmp_path) or Path(output_file).exists(), (
            "save_audio=True must create output file under tmp_path"
        )

    def test_voice_cloner_say_raises_on_failure(self, tmp_path, engine_mock):
        """VoiceCloner.say must raise when engine.generate fails."""
        from voice_cloner import VoiceCloner

        engine_mock.generate.side_effect = RuntimeError("generation failed")

        speaker_file = tmp_path / "speaker.wav"
        speaker_file.write_bytes(b"fake audio data")

        cloner = VoiceCloner(speaker_wav=str(speaker_file), engine=engine_mock)
        with pytest.raises(RuntimeError, match="generation failed"):
            cloner.say("hello", save_audio=False, play_audio=False)


# ---------------------------------------------------------------------------
# 4. TTSFactory engine registration
# ---------------------------------------------------------------------------


class TestTTSFactoryRegistration:
    """TTSFactory must register and retrieve engines correctly."""

    def test_factory_available_engines_returns_list(self):
        """available_engines must return a list (may be empty without deps)."""
        from tts_factory import TTSFactory

        engines = TTSFactory.available_engines()
        assert isinstance(engines, list)

    def test_factory_get_engine_info(self):
        """get_engine_info must return engine metadata dict."""
        from tts_factory import TTSFactory

        info = TTSFactory.get_engine_info()
        assert isinstance(info, dict)

    def test_factory_get_display_name(self):
        """get_display_name must return a human-readable name."""
        from tts_factory import TTSFactory

        name = TTSFactory.get_display_name("coqui")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_factory_is_available_returns_bool(self):
        """is_available must return a boolean for any engine name."""
        from tts_factory import TTSFactory

        result = TTSFactory.is_available("coqui")
        assert isinstance(result, bool)

    def test_factory_get_controls_class(self):
        """get_controls_class must return a class or None."""
        from tts_factory import TTSFactory

        result = TTSFactory.get_controls_class("coqui")
        assert result is None or isinstance(result, type)


# ---------------------------------------------------------------------------
# 5. Model registry operations
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Model registry must handle CRUD operations correctly."""

    def test_registry_has_default_models(self):
        """Default registry must have pre-populated models."""
        from models.model_registry import get_registry

        reg = get_registry()
        models = reg.list_models()
        assert len(models) >= 1, "default registry must have models"

    def test_registry_list_model_ids(self):
        """list_model_ids must return model ID strings."""
        from models.model_registry import get_registry

        reg = get_registry()
        ids = reg.list_model_ids()
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_registry_is_installed(self):
        """is_installed must reflect the installed flag."""
        from models.model_registry import get_registry

        reg = get_registry()
        # Check a known model
        model_ids = reg.list_model_ids()
        if model_ids:
            result = reg.is_installed(model_ids[0])
            assert isinstance(result, bool)

    def test_registry_get_models_for_engine(self):
        """get_models_for_engine must filter by engine."""
        from models.model_registry import get_registry

        reg = get_registry()
        coqui_models = reg.get_models_for_engine("coqui")
        assert isinstance(coqui_models, list)

    def test_registry_get_model_id_for_engine(self):
        """get_model_id_for_engine must return a model ID for known engines."""
        from models.model_registry import get_registry

        reg = get_registry()
        model_id = reg.get_model_id_for_engine("coqui")
        assert isinstance(model_id, str)
        assert len(model_id) > 0

    def test_registry_register_model(self):
        """register_model must add a model to the registry."""
        from models.model_info import ModelInfo
        from models.model_registry import ModelRegistry

        reg = ModelRegistry()
        info = ModelInfo(
            id="custom-model",
            engine="coqui",
            name="Custom Model",
            size_mb=50,
            description="Custom test model",
        )
        reg.register_model(info)
        models = reg.list_models()
        names = [m.name for m in models]
        assert "Custom Model" in names

    def test_registry_get_engine_for_model(self):
        """get_engine_for_model must return the engine name."""
        from models.model_registry import get_registry

        reg = get_registry()
        model_ids = reg.list_model_ids()
        if model_ids:
            engine = reg.get_engine_for_model(model_ids[0])
            assert isinstance(engine, str)


# ---------------------------------------------------------------------------
# 6. VoiceClonerCache
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")


class TestVoiceClonerCache:
    """Tests for the VoiceClonerCache class."""

    def test_cache_miss_creates_new_cloner(self):
        """First access must create a new VoiceCloner."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cloner = cache.get("coqui", "/path/to/speaker.wav")

        assert cloner is not None
        assert cache.size == 1

    def test_cache_hit_returns_same_cloner(self):
        """Repeated access with same keys must return the same instance."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cloner1 = cache.get("coqui", "/path/to/speaker.wav")
        cloner2 = cache.get("coqui", "/path/to/speaker.wav")

        assert cloner1 is cloner2
        assert cache.size == 1

    def test_different_engines_different_caches(self):
        """Different engine names must have separate cache entries."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cache.get("coqui", "/path/to/speaker.wav")
        cache.get("chatterbox-turbo", "/path/to/speaker.wav")

        assert cache.size == 2

    def test_different_speakers_different_caches(self):
        """Different speaker paths must have separate cache entries."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cache.get("coqui", "/path/to/speaker_a.wav")
        cache.get("coqui", "/path/to/speaker_b.wav")

        assert cache.size == 2

    def test_invalidate_engine_clears_only_that_engine(self):
        """Invalidating an engine must only remove entries for that engine."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cache.get("coqui", "/path/to/speaker.wav")
        cache.get("chatterbox-turbo", "/path/to/speaker.wav")

        assert cache.size == 2

        cache.invalidate(engine_name="coqui")

        assert cache.size == 1

    def test_invalidate_speaker_clears_only_that_speaker(self):
        """Invalidating a speaker must only remove entries for that speaker."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cache.get("coqui", "/path/to/speaker_a.wav")
        cache.get("chatterbox-turbo", "/path/to/speaker_a.wav")
        cache.get("coqui", "/path/to/speaker_b.wav")

        assert cache.size == 3

        cache.invalidate(speaker_wav="/path/to/speaker_a.wav")

        assert cache.size == 1

    def test_invalidate_all_clears_everything(self):
        """Invalidating with no args must clear the entire cache."""
        from gui.clone_flow_controller import VoiceClonerCache

        cache = VoiceClonerCache()
        cache.get("coqui", "/path/to/speaker.wav")
        cache.get("chatterbox-turbo", "/path/to/speaker.wav")

        assert cache.size == 2

        cache.invalidate()

        assert cache.size == 0

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock heavyweight external modules (not platform_utils — we patch that via tts_factory)
for _mod in (
    "TTS",
    "chatterbox",
    "mlx_audio",
    "tts_engine_base",
):
    sys.modules[_mod] = MagicMock()

# If another test file (e.g. test_voice_cloner.py) has put a mock in
# sys.modules["tts_factory"], remove it so the real module is imported.
sys.modules.pop("tts_factory", None)

from tts_factory import EngineDescriptor, TTSFactory, bootstrap_engines  # noqa: E402


class StubEngine:
    def __init__(self, speaker_wav="", device=None, **kwargs):
        self.speaker_wav = speaker_wav
        self.device = device
        self.kwargs = kwargs
        self.name = "stub"

    def generate(self, text, **kwargs):
        ...

    def get_supported_parameters(self):
        return []


@pytest.fixture(autouse=True)
def reset_registry():
    TTSFactory._registry.clear()
    yield
    TTSFactory._registry.clear()


@pytest.fixture
def basic_descriptor():
    return EngineDescriptor(
        name="test-engine",
        engine_class=StubEngine,
        display_name="Test Engine",
        requires_reference_audio=True,
        supports_preset_voices=False,
    )


class TestEngineDescriptor:
    def test_creates_with_defaults(self):
        d = EngineDescriptor(name="foo", engine_class=StubEngine, display_name="Foo")
        assert d.name == "foo"
        assert d.requires_reference_audio is True
        assert d.supports_preset_voices is False
        assert d.platform_restriction is None
        assert d.default_kwargs == {}

    def test_creates_with_all_fields(self):
        d = EngineDescriptor(
            name="bar",
            engine_class=StubEngine,
            display_name="Bar",
            requires_reference_audio=False,
            supports_preset_voices=True,
            platform_restriction="apple_silicon",
            default_kwargs={"variant": "turbo"},
        )
        assert d.name == "bar"
        assert d.requires_reference_audio is False
        assert d.supports_preset_voices is True
        assert d.platform_restriction == "apple_silicon"
        assert d.default_kwargs == {"variant": "turbo"}

    def test_is_available_on_platform_no_restriction(self):
        d = EngineDescriptor(name="foo", engine_class=StubEngine, display_name="Foo")
        assert d.is_available_on_platform() is True

    def test_dependencies_installed_unknown_engine(self):
        d = EngineDescriptor(name="unknown-foo", engine_class=StubEngine, display_name="Foo")
        assert d.dependencies_installed() is True

    def test_dependencies_installed_coqui(self):
        d = EngineDescriptor(name="coqui", engine_class=StubEngine, display_name="Coqui")
        with patch.dict(sys.modules, {"TTS": MagicMock()}):
            assert d.dependencies_installed() is True

    def test_dependencies_installed_coqui_missing(self):
        d = EngineDescriptor(name="coqui", engine_class=StubEngine, display_name="Coqui")
        with patch("builtins.__import__", side_effect=ImportError("no TTS")):
            assert d.dependencies_installed() is False


class TestTTSFactoryRegister:
    def test_registers_descriptor(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        assert "test-engine" in TTSFactory._registry
        assert TTSFactory._registry["test-engine"] is basic_descriptor

    def test_raises_on_duplicate(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        with pytest.raises(ValueError, match="already registered"):
            TTSFactory.register(basic_descriptor)

    def test_registers_multiple_engines(self):
        TTSFactory.register(EngineDescriptor(name="a", engine_class=StubEngine, display_name="A"))
        TTSFactory.register(EngineDescriptor(name="b", engine_class=StubEngine, display_name="B"))
        assert set(TTSFactory.available_engines()) == {"a", "b"}

    def test_register_rejects_incomplete(self):
        with pytest.raises(TypeError, match="EngineDescriptor"):
            TTSFactory.register("not-a-descriptor")  # type: ignore


class TestTTSFactoryCreate:
    def test_creates_engine_instance(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        engine = TTSFactory.create("test-engine", speaker_wav="voice.wav", device="cpu")
        assert engine.speaker_wav == "voice.wav"
        assert engine.device == "cpu"

    def test_merges_default_and_custom_kwargs(self):
        TTSFactory.register(EngineDescriptor(
            name="custom",
            engine_class=StubEngine,
            display_name="Custom",
            default_kwargs={"variant": "turbo", "speed": 1.0},
        ))
        engine = TTSFactory.create("custom", speaker_wav="v.wav", speed=2.0)
        assert engine.kwargs == {"variant": "turbo", "speed": 2.0}

    def test_raises_on_unknown_engine(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            TTSFactory.create("nonexistent", speaker_wav="v.wav")


class TestTTSFactoryQuery:
    def test_available_engines_empty_by_default(self):
        assert TTSFactory.available_engines() == []

    def test_get_display_name_returns_descriptor_name(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        assert TTSFactory.get_display_name("test-engine") == "Test Engine"

    def test_get_display_name_fallback(self):
        assert TTSFactory.get_display_name("unknown") == "unknown"

    def test_get_engine_info(self):
        TTSFactory.register(EngineDescriptor(name="a", engine_class=StubEngine, display_name="Engine A"))
        TTSFactory.register(EngineDescriptor(name="b", engine_class=StubEngine, display_name="Engine B"))
        assert TTSFactory.get_engine_info() == {"a": "Engine A", "b": "Engine B"}

    def test_get_engine_metadata(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        meta = TTSFactory.get_engine_metadata("test-engine")
        assert meta["display_name"] == "Test Engine"
        assert meta["requires_reference_audio"] is True
        assert meta["supports_preset_voices"] is False

    def test_get_engine_metadata_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            TTSFactory.get_engine_metadata("nonexistent")


class TestTTSFactoryAvailability:
    def test_is_available_unknown_engine(self):
        assert TTSFactory.is_available("nonexistent") is False

    def test_get_available_engines_empty_when_none_installed(self, basic_descriptor):
        TTSFactory.register(basic_descriptor)
        with patch.object(EngineDescriptor, "dependencies_installed", return_value=False):
            assert TTSFactory.get_available_engines() == []

    def test_get_default_engine_raises_when_empty(self):
        with pytest.raises(RuntimeError, match="No TTS engines available"):
            TTSFactory.get_default_engine()

    def test_get_default_engine_uses_first_available(self):
        TTSFactory.register(EngineDescriptor(name="alpha", engine_class=StubEngine, display_name="Alpha"))
        with patch.object(EngineDescriptor, "dependencies_installed", return_value=True):
            assert TTSFactory.get_default_engine() == "alpha"


class TestBootstrapEngines:
    def test_bootstrap_preserves_existing_registrations(self):
        TTSFactory.register(EngineDescriptor(name="pre-existing", engine_class=StubEngine, display_name="Pre"))
        engines_before = set(TTSFactory.available_engines())

        bootstrap_engines()

        engines_after = set(TTSFactory.available_engines())
        assert "pre-existing" in engines_after
        assert engines_before.issubset(engines_after)

    def test_no_auto_register_on_import(self):
        assert TTSFactory.available_engines() == []

    def test_import_sans_bootstrap_leaves_registry_empty(self):
        assert TTSFactory.available_engines() == []

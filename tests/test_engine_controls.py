"""Tests for engine-specific control widgets."""

import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

# Mock torch before any imports trigger tts_engine_base → torch chain
for _mod in ("torch",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from PySide6.QtCore import QCoreApplication  # noqa: E402

from gui.engine_controls import (  # noqa: E402
    ChatterboxControls,
    CoquiControls,
    EngineControlsBase,
    EngineControlsFactory,
    MlxCsmControls,
    MlxKokoroControls,
)
from tts_factory import EngineDescriptor, TTSFactory  # noqa: E402


class StubEngine:
    def __init__(self, speaker_wav="", device=None, **kwargs):
        self.speaker_wav = speaker_wav
        self.device = device
        self.kwargs = kwargs

    def generate(self, text, **kwargs): ...

    def get_supported_parameters(self):
        return []


@pytest.fixture(autouse=True)
def register_test_engines():
    """Register mock engine descriptors before each test so the
    descriptor-driven EngineControlsFactory can resolve controls_class."""
    TTSFactory._registry.clear()
    TTSFactory.register(
        EngineDescriptor(
            name="coqui",
            engine_class=StubEngine,
            display_name="Coqui XTTS v2",
            requires_reference_audio=True,
            controls_class=CoquiControls,
        )
    )
    TTSFactory.register(
        EngineDescriptor(
            name="chatterbox-turbo",
            engine_class=StubEngine,
            display_name="Chatterbox Turbo (350M)",
            default_kwargs={"variant": "turbo"},
            controls_class=ChatterboxControls,
        )
    )
    TTSFactory.register(
        EngineDescriptor(
            name="chatterbox-standard",
            engine_class=StubEngine,
            display_name="Chatterbox Standard (500M)",
            default_kwargs={"variant": "standard"},
            controls_class=ChatterboxControls,
        )
    )
    TTSFactory.register(
        EngineDescriptor(
            name="mlx-kokoro",
            engine_class=StubEngine,
            display_name="MLX Kokoro (Preset Voices)",
            requires_reference_audio=False,
            supports_preset_voices=True,
            controls_class=MlxKokoroControls,
        )
    )
    TTSFactory.register(
        EngineDescriptor(
            name="mlx-csm",
            engine_class=StubEngine,
            display_name="MLX CSM (Voice Cloning)",
            requires_reference_audio=True,
            controls_class=MlxCsmControls,
        )
    )
    yield
    TTSFactory._registry.clear()


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        from PySide6.QtWidgets import QApplication

        app = QApplication([])
    return app


class TestEngineControlsBase:
    """Tests for EngineControlsBase (fallback for unknown engines)."""

    def test_base_get_parameters_returns_empty_dict(self, qapp):
        controls = EngineControlsBase()
        params = controls.get_parameters()
        assert params == {}

    def test_base_parameters_changed_signal(self, qapp):
        controls = EngineControlsBase()
        emitted = []

        def capture(params):
            emitted.append(params)

        controls.parameters_changed.connect(capture)
        controls.parameters_changed.emit({})
        assert emitted == [{}]

    def test_base_returns_empty_params_for_unknown_engine_via_factory(self, qapp):
        controls = EngineControlsFactory.create("nonexistent-engine")
        assert controls.get_parameters() == {}


class TestEngineControlsFactory:
    """Tests for EngineControlsFactory.create()."""

    def test_create_coqui(self, qapp):
        controls = EngineControlsFactory.create("coqui")
        assert isinstance(controls, CoquiControls)

    def test_create_chatterbox_turbo(self, qapp):
        controls = EngineControlsFactory.create("chatterbox-turbo")
        assert isinstance(controls, ChatterboxControls)
        assert controls.variant == "turbo"

    def test_create_chatterbox_standard(self, qapp):
        controls = EngineControlsFactory.create("chatterbox-standard")
        assert isinstance(controls, ChatterboxControls)
        assert controls.variant == "standard"

    def test_create_mlx_kokoro(self, qapp):
        controls = EngineControlsFactory.create("mlx-kokoro")
        assert isinstance(controls, MlxKokoroControls)

    def test_create_mlx_csm(self, qapp):
        controls = EngineControlsFactory.create("mlx-csm")
        assert isinstance(controls, MlxCsmControls)

    def test_create_unknown_engine_returns_base(self, qapp):
        controls = EngineControlsFactory.create("unknown-engine")
        assert isinstance(controls, EngineControlsBase)

    def test_create_unknown_engine_get_parameters(self, qapp):
        controls = EngineControlsFactory.create("unknown-engine")
        params = controls.get_parameters()
        assert params == {}


class TestKnownEngineControls:
    """Tests that known engine controls return valid parameters."""

    @pytest.mark.parametrize(
        ("engine_name", "expected_type"),
        [
            ("coqui", CoquiControls),
            ("chatterbox-turbo", ChatterboxControls),
            ("chatterbox-standard", ChatterboxControls),
            ("mlx-kokoro", MlxKokoroControls),
            ("mlx-csm", MlxCsmControls),
        ],
    )
    def test_known_engines_return_dict(self, qapp, engine_name, expected_type):
        controls = EngineControlsFactory.create(engine_name)
        assert isinstance(controls, expected_type)
        params = controls.get_parameters()
        assert isinstance(params, dict)


class TestMlxVoiceMetadata:
    """Tests that MLX voice metadata is consolidated (single source of truth)."""

    def test_voice_groups_is_engine_source(self):
        from engines.mlx_audio_engine import KOKORO_VOICES

        assert MlxKokoroControls.VOICE_GROUPS is KOKORO_VOICES

    def test_lang_codes_is_engine_source(self):
        from engines.mlx_audio_engine import KOKORO_LANG_CODES

        assert MlxKokoroControls.LANG_CODES is KOKORO_LANG_CODES

    def test_voice_groups_contains_expected_languages(self):
        assert "American English" in MlxKokoroControls.VOICE_GROUPS
        assert "British English" in MlxKokoroControls.VOICE_GROUPS
        assert "Japanese" in MlxKokoroControls.VOICE_GROUPS
        assert "Mandarin Chinese" in MlxKokoroControls.VOICE_GROUPS

    def test_voice_groups_has_same_structure_as_engine(self):
        from engines.mlx_audio_engine import KOKORO_VOICES

        assert set(MlxKokoroControls.VOICE_GROUPS.keys()) == set(KOKORO_VOICES.keys())

    def test_lang_codes_has_same_mapping(self):
        from engines.mlx_audio_engine import KOKORO_LANG_CODES

        assert MlxKokoroControls.LANG_CODES == KOKORO_LANG_CODES

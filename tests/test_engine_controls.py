"""Tests for engine-specific control widgets."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication  # noqa: E402

from gui.engine_controls import (  # noqa: E402
    Audio8Controls,
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
    TTSFactory.register(
        EngineDescriptor(
            name="audio8-onnx",
            engine_class=StubEngine,
            display_name="Audio8 TTS (1B)",
            requires_reference_audio=True,
            controls_class=Audio8Controls,
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

    def test_base_get_parameters_includes_chunk_silence(self, qapp):
        controls = EngineControlsBase()
        params = controls.get_parameters()
        assert params == {"silence_duration": 200}

    def test_base_parameters_changed_signal(self, qapp):
        controls = EngineControlsBase()
        emitted = []

        def capture(params):
            emitted.append(params)

        controls.parameters_changed.connect(capture)
        controls.parameters_changed.emit({})
        assert emitted == [{}]

    def test_base_returns_chunk_silence_for_unknown_engine_via_factory(self, qapp):
        controls = EngineControlsFactory.create("nonexistent-engine")
        assert controls.get_parameters() == {"silence_duration": 200}


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

    def test_create_audio8(self, qapp):
        controls = EngineControlsFactory.create("audio8-onnx")
        assert isinstance(controls, Audio8Controls)

    def test_create_unknown_engine_returns_base(self, qapp):
        controls = EngineControlsFactory.create("unknown-engine")
        assert isinstance(controls, EngineControlsBase)

    def test_create_unknown_engine_get_parameters(self, qapp):
        controls = EngineControlsFactory.create("unknown-engine")
        params = controls.get_parameters()
        assert params == {"silence_duration": 200}


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
            ("audio8-onnx", Audio8Controls),
        ],
    )
    def test_known_engines_return_dict(self, qapp, engine_name, expected_type):
        controls = EngineControlsFactory.create(engine_name)
        assert isinstance(controls, expected_type)
        params = controls.get_parameters()
        assert isinstance(params, dict)

    @pytest.mark.parametrize(
        "engine_name",
        [
            "coqui",
            "chatterbox-turbo",
            "chatterbox-standard",
            "mlx-kokoro",
            "mlx-csm",
            "audio8-onnx",
        ],
    )
    def test_known_engines_default_to_200ms_chunk_silence(self, qapp, engine_name):
        controls = EngineControlsFactory.create(engine_name)
        assert controls.get_parameters()["silence_duration"] == 200

    def test_silence_change_emits_complete_parameter_payload(self, qapp):
        controls = EngineControlsFactory.create("coqui")
        emitted = []
        controls.parameters_changed.connect(emitted.append)

        controls.silence_duration_spin.setValue(350)

        assert emitted[-1]["language"] == "en"
        assert emitted[-1]["temperature"] == pytest.approx(0.7)
        assert emitted[-1]["silence_duration"] == 350


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

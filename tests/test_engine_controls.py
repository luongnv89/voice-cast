"""Tests for engine-specific control widgets."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication  # noqa: E402

from gui.engine_controls import (  # noqa: E402
    ChatterboxControls,
    CoquiControls,
    EngineControlsBase,
    EngineControlsFactory,
    MlxCsmControls,
    MlxKokoroControls,
)


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

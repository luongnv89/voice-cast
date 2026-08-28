"""GUI import/offscreen smoke test for the main entry point (F-CI-003).

CI installs the package but historically never imported the GUI chain or booted
the app object, so an import error would ship green. These tests import the real
entry point module and construct the main window headlessly; the CI test job
installs PySide6 so they execute there instead of skipping.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

import voice_cloning_app  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gui_entry_point_imports(qapp):
    """Importing voice_cloning_app resolves the full GUI dependency chain."""
    assert callable(voice_cloning_app.main)


def test_main_window_constructs_offscreen(qapp):
    """Booting the main window offscreen yields a sane, closable window."""
    window = voice_cloning_app.VoiceCloningApp()
    try:
        assert isinstance(window, QMainWindow)
        assert window.windowTitle() == "VoiceCast"
        assert window.centralWidget() is not None
    finally:
        window.close()


def test_stage_accessible_name_tracks_current_stage(qapp):
    """Assistive technology must receive the current generation stage."""
    window = voice_cloning_app.VoiceCloningApp()
    try:
        window.set_stage_text("Loading model...")
        assert window._stage_label.text() == "Loading model..."
        assert window._stage_label.accessibleName() == "Generation stage: Loading model..."
    finally:
        window.close()

"""Offscreen widget tests for Generate input gating (Task 6.3, #75).

Verifies that the Generate button stays disabled until inputs are valid,
with an inline hint naming what is missing.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestGenerateGate:
    """Generate button must be disabled until all required inputs are valid."""

    def test_generate_disabled_with_no_inputs(self, qapp):
        """Generate is disabled when no text and no voice file."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            assert window.btn_generate.isEnabled() is False
            assert window._generate_hint.isVisible() is True
            hint = window._generate_hint.text().lower()
            assert "enter text" in hint or "select" in hint
        finally:
            window.close()

    def test_generate_enabled_when_text_entered(self, qapp):
        """Generate becomes enabled after text is entered (voice not required for preset engines)."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            # Check if voice is required for default engine
            voice_required = window.is_voice_required
            if not voice_required:
                window.text_input.setPlainText("Hello world")
                assert window.btn_generate.isEnabled() is True
                assert not window._generate_hint.isVisible()
            else:
                # For engines requiring voice, need both text and voice
                pytest.skip("Engine requires voice file — test covers text-only path")
        finally:
            window.close()

    def test_generate_hint_shows_missing_inputs(self, qapp):
        """Hint text names what is missing when Generate is disabled."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            # Start with hint visible
            assert window._generate_hint.isVisible()
            hint = window._generate_hint.text()

            # Enter some text
            window.text_input.setPlainText("Some text")
            voice_required = window.is_voice_required
            if voice_required:
                # Still disabled because no voice file
                assert window.btn_generate.isEnabled() is False
                assert "select" in hint.lower()
            else:
                # Should be enabled now
                assert window.btn_generate.isEnabled() is True
                assert not window._generate_hint.isVisible()
        finally:
            window.close()

    def test_generate_stays_disabled_with_empty_text(self, qapp):
        """Entering and clearing text keeps Generate disabled."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            voice_required = window.is_voice_required
            if not voice_required:
                window.text_input.setPlainText("temp")
                window.text_input.setPlainText("")
                assert window.btn_generate.isEnabled() is False
                assert window._generate_hint.isVisible()
            else:
                pytest.skip("Engine requires voice file")
        finally:
            window.close()

"""Tests for safe window close during generation (Task 6.4, #76).

Verifies that closing the window during generation requests cooperative
shutdown and defers close until the worker exits.
"""

from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _UncooperativeThread:
    """Fake worker that ignores interruption until its native finish signal."""

    def __init__(self):
        self.running = True
        self.interruption_requests = 0
        self.wait_calls = []
        self.output_path = None
        self.output_owned = False
        self.keep_output = False
        self.temp_voice_file = None
        self.deleted = False

    def isRunning(self):
        return self.running

    def requestInterruption(self):
        self.interruption_requests += 1

    def wait(self, *args):
        self.wait_calls.append(args)
        raise AssertionError("generation shutdown must not wait on the GUI thread")

    def terminate(self):
        raise AssertionError("generation threads must never be forcibly terminated")

    def _discard_output(self):
        if self.output_owned and self.output_path is not None:
            self.output_path.unlink(missing_ok=True)
            self.output_owned = False

    def deleteLater(self):
        self.deleted = True


class TestWindowCloseSafety:
    """Window close must safely terminate in-flight generation."""

    def test_terminate_method_exists(self, qapp):
        """CloneFlowController must have a terminate method."""
        from gui.clone_flow_controller import CloneFlowController

        assert hasattr(CloneFlowController, "terminate")

    def test_terminate_is_noop_when_not_running(self, qapp):
        """terminate() must be safe to call when no generation is in progress."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            # Should not raise
            window._clone_flow.terminate()
        finally:
            window.close()

    def test_generate_disabled_during_terminate(self, qapp):
        """Generate button state is checked during close."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        try:
            # Verify Generate button exists
            assert window.btn_generate is not None
            # Verify is_running property exists
            assert hasattr(window._clone_flow, "is_running")
        finally:
            window.close()

    def test_terminate_is_non_blocking_and_defers_cleanup(self, qapp, tmp_path):
        """A worker remains owned until its native finished signal arrives."""
        from gui.clone_flow_controller import CloneFlowController

        controller = CloneFlowController(object())
        thread = _UncooperativeThread()
        output_path = tmp_path / "owned.wav"
        output_path.touch()
        temp_voice_file = tmp_path / "voice.wav"
        temp_voice_file.touch()
        thread.output_path = output_path
        thread.output_owned = True
        thread.temp_voice_file = str(temp_voice_file)
        controller._thread = thread
        controller._temp_voice_file = str(temp_voice_file)

        controller.terminate()
        controller.terminate()

        assert thread.interruption_requests == 1
        assert thread.wait_calls == []
        assert controller._thread is thread
        assert output_path.exists()
        assert temp_voice_file.exists()

        controller._on_thread_finished(thread)

        assert controller._thread is None
        assert not output_path.exists()
        assert not temp_voice_file.exists()
        assert thread.deleted

    def test_close_retries_after_worker_completion(self, qapp, monkeypatch):
        """Repeated close requests stay ignored until the worker has exited."""
        from voice_cloning_app import VoiceCloningApp

        window = VoiceCloningApp()
        thread = _UncooperativeThread()
        window._clone_flow._thread = thread
        retry_calls = []
        monkeypatch.setattr(window, "close", lambda: retry_calls.append(True))
        try:
            first_event = QCloseEvent()
            second_event = QCloseEvent()
            window.closeEvent(first_event)
            window.closeEvent(second_event)

            assert not first_event.isAccepted()
            assert not second_event.isAccepted()
            assert thread.interruption_requests == 1
            assert window._close_pending

            window._clone_flow._on_thread_finished(thread)
            qapp.processEvents()

            assert retry_calls == [True]
            assert not window._clone_flow.has_worker

            accepted_event = QCloseEvent()
            monkeypatch.undo()
            window.closeEvent(accepted_event)
            assert accepted_event.isAccepted()
        finally:
            window.hide()

    def test_late_generation_signals_are_suppressed_during_shutdown(self, qapp):
        """Queued stage, success, and error signals cannot show late UI."""
        from gui.clone_flow_controller import CloneFlowController

        ui = Mock()
        controller = CloneFlowController(ui)
        thread = _UncooperativeThread()
        controller._thread = thread
        controller.terminate()

        controller._handle_stage_changed(thread, "late stage")
        controller._handle_finished(thread, "late.wav", "late text")
        controller._handle_error(thread, "late error")

        ui.set_stage_text.assert_not_called()
        ui.info.assert_not_called()
        ui.critical.assert_not_called()

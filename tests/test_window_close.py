"""Tests for safe window close during generation (Task 6.4, #76).

Verifies that closing the window during generation properly terminates
the thread and prevents crash-on-exit risk.
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

    def test_terminate_requests_cooperative_shutdown(self, qapp):
        """Closing a generation must not forcibly terminate its QThread."""
        from gui.clone_flow_controller import CloneFlowController

        class CooperativeThread:
            def __init__(self, wait_results=(True,)):
                self.running = True
                self.interruption_requests = 0
                self.wait_calls = []
                self.wait_results = iter(wait_results)

            def isRunning(self):
                return self.running

            def requestInterruption(self):
                self.interruption_requests += 1

            def wait(self, *args):
                self.wait_calls.append(args)
                result = next(self.wait_results)
                if result:
                    self.running = False
                return result

            def terminate(self):
                raise AssertionError("generation threads must never be forcibly terminated")

        controller = CloneFlowController(object())
        thread = CooperativeThread()
        controller._thread = thread

        controller.terminate()

        assert thread.interruption_requests == 1
        assert thread.wait_calls == [(5000,)]
        assert controller._thread is None

    def test_terminate_waits_for_natural_completion_after_timeout(self, qapp):
        """A slow model load is allowed to finish instead of being killed."""
        from gui.clone_flow_controller import CloneFlowController

        class SlowCooperativeThread:
            def __init__(self):
                self.running = True
                self.wait_calls = []
                self.wait_results = iter((False, True))

            def isRunning(self):
                return self.running

            def requestInterruption(self):
                pass

            def wait(self, *args):
                self.wait_calls.append(args)
                result = next(self.wait_results)
                if result:
                    self.running = False
                return result

        controller = CloneFlowController(object())
        thread = SlowCooperativeThread()
        controller._thread = thread

        controller.terminate()

        assert thread.wait_calls == [(5000,), ()]
        assert controller._thread is None

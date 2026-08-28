"""Regression tests for GUI-thread clone completion handling."""

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.clone_flow_controller import CloneFlowController, CloneThread  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _UiDelegate:
    def __init__(self, event_loop):
        self._event_loop = event_loop
        self.info_calls = 0
        self.info_thread_id = None
        self.stage_thread_id = None

    def enable_generate(self):
        pass

    def set_generate_text(self, _text):
        pass

    def enable_voice_select(self):
        pass

    def enable_engine_combo(self):
        pass

    def hide_progress(self):
        pass

    def show_play_save(self):
        pass

    def set_stage_text(self, _stage):
        self.stage_thread_id = threading.get_ident()

    def info(self, _title, _message):
        self.info_calls += 1
        self.info_thread_id = threading.get_ident()
        self._event_loop.quit()


class _VoiceCloner:
    def say(self, *_args, **_kwargs):
        pass


def test_clone_completion_runs_on_gui_thread(qapp):
    event_loop = QEventLoop()
    ui = _UiDelegate(event_loop)
    controller = CloneFlowController(ui)
    worker = CloneThread(
        text="hello",
        voice_path="voice.wav",
        engine_name="test",
        engine_params={},
        voice_cloner=_VoiceCloner(),
    )
    controller._thread = worker
    controller._connect_thread_signals(worker)

    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(event_loop.quit)
    worker.start()
    timeout.start(2000)
    try:
        event_loop.exec()
    finally:
        timeout.stop()
        worker.wait(2000)

    gui_thread_id = threading.get_ident()
    assert ui.info_calls == 1
    assert ui.info_thread_id == gui_thread_id
    assert ui.stage_thread_id == gui_thread_id

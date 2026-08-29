"""Regression tests for GUI-thread clone completion handling."""

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import clone_flow_controller
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
        self.info_message = None
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

    def info(self, _title, message):
        self.info_calls += 1
        self.info_message = message
        self.info_thread_id = threading.get_ident()
        self._event_loop.quit()


class _VoiceCloner:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_clone_thread_forwards_chunking_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_flow_controller.tempfile, "gettempdir", lambda: str(tmp_path))
    cloner = _VoiceCloner()
    worker = CloneThread(
        text="hello",
        voice_path="voice.wav",
        engine_name="test",
        engine_params={"chunk_size": 120, "silence_duration": 350},
        voice_cloner=cloner,
    )

    worker.run()

    assert cloner.calls[0][1]["chunk_size"] == 120
    assert cloner.calls[0][1]["silence_duration"] == 350


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
    assert str(ui.current_audio) in ui.info_message
    assert ui.info_thread_id == gui_thread_id
    assert ui.stage_thread_id == gui_thread_id


def test_output_paths_are_unique_and_owned_cleanup_is_collision_safe(qapp, tmp_path, monkeypatch):
    """A colliding prior output is preserved while the current output is removable."""
    monkeypatch.setattr(clone_flow_controller.tempfile, "gettempdir", lambda: str(tmp_path))
    uuids = iter(
        [
            SimpleNamespace(hex="collision"),
            SimpleNamespace(hex="collision"),
        ]
    )
    monkeypatch.setattr(clone_flow_controller.uuid, "uuid4", lambda: next(uuids))

    first = CloneThread("first", "voice.wav", "test", {}, _VoiceCloner())
    first.output_path.parent.mkdir(parents=True, exist_ok=True)
    first.output_path.touch()
    second = CloneThread("second", "voice.wav", "test", {}, _VoiceCloner())

    assert first.output_path != second.output_path
    assert first.output_path.exists()

    first._discard_output()
    assert first.output_path.exists()

    second.output_path.touch()
    second.output_owned = True
    second._discard_output()

    assert not second.output_path.exists()
    assert first.output_path.exists()

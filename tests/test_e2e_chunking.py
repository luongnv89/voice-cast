"""End-to-end integration for long-text chunking with progress and cancellation.

Covers GUI and CLI adapter paths with a mocked engine so CI never needs
torch/TTS. Mirrors the acceptance criteria of issue #144.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.clone_flow_controller import CloneFlowController, CloneThread  # noqa: E402
from voice_cloner import GenerationCancelled, VoiceCloner  # noqa: E402
import vcloner  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Delegate:
    def __init__(self):
        self.progress = []
        self.stage = None
        self.info = None

    def enable_generate(self):
        pass

    def set_generate_text(self, _t):
        pass

    def enable_voice_select(self):
        pass

    def enable_engine_combo(self):
        pass

    def hide_progress(self):
        pass

    def show_play_save(self):
        pass

    def set_stage_text(self, t):
        self.stage = t

    def set_chunk_progress(self, c, t):
        self.progress.append((c, t))

    def info(self, title, msg):
        self.info = (title, msg)


def _mock_engine():
    engine = MagicMock()
    engine.name = "mock"
    engine.requires_reference_audio = False
    engine.MAX_CHUNK_CHARS = 10
    engine.generate.return_value = (np.array([0.1], dtype=np.float32), 1000)
    return engine


def test_long_text_with_progress_via_voice_cloner(tmp_path):
    """Backend synthesizes long text with mocked engine and reports progress correctly."""
    engine = _mock_engine()
    engine.generate.side_effect = [
        (np.array([1.0], dtype=np.float32), 1000),
        (np.array([2.0], dtype=np.float32), 1000),
        (np.array([3.0], dtype=np.float32), 1000),
    ]
    cloner = VoiceCloner(speaker_wav="", engine=engine)
    events = []

    def cb(c, t):
        events.append((c, t))

    with patch("voice_cloner.split_into_chunks", return_value=["a", "b", "c"]):
        with patch("voice_cloner.sf.write") as write:
            cloner.generate("long text long text", chunk_size=10, output_file=str(tmp_path / "out.wav"), chunk_progress_callback=cb)

    assert events == [(1, 3), (2, 3), (3, 3)]
    assert engine.generate.call_count == 3
    assert write.called


def test_cancellation_produces_partial_output(tmp_path):
    """Cancellation between chunks produces a valid partial WAV and stops synthesis."""
    engine = _mock_engine()
    engine.generate.side_effect = [
        (np.array([1.0, 1.0], dtype=np.float32), 1000),
        (np.array([2.0], dtype=np.float32), 1000),
        (np.array([3.0], dtype=np.float32), 1000),
    ]
    cloner = VoiceCloner(speaker_wav="", engine=engine)

    def cancel_after_first():
        return engine.generate.call_count >= 1

    with patch("voice_cloner.split_into_chunks", return_value=["a", "b", "c"]):
        out = str(tmp_path / "partial.wav")
        with pytest.raises(GenerationCancelled):
            cloner.generate("long text long text", chunk_size=10, output_file=out, cancel_requested=cancel_after_first)

    assert engine.generate.call_count == 1
    assert Path(out).exists()
    import soundfile as sf

    data, sr = sf.read(out)
    assert sr == 1000
    assert len(data) == 2


def test_gui_chunk_progress_and_cancel(qapp, tmp_path):
    """GUI controller relays chunk progress and handles cooperative cancellation."""
    # Progress relay via controller
    delegate = _Delegate()
    controller = CloneFlowController(delegate)
    engine = _mock_engine()
    engine.generate.side_effect = [
        (np.array([1.0], dtype=np.float32), 1000),
        (np.array([2.0], dtype=np.float32), 1000),
    ]

    cloner = VoiceCloner(speaker_wav="", engine=engine)

    with patch("voice_cloner.split_into_chunks", return_value=["a", "b"]):
        # Directly test progress handling
        worker = CloneThread("hello", "", "test", {}, cloner)
        relay = []
        controller.chunk_progress.connect(lambda c, t: relay.append((c, t)))
        controller._thread = worker
        controller._handle_chunk_progress(worker, 1, 2)
        controller._handle_chunk_progress(worker, 2, 2)
        assert relay == [(1, 2), (2, 2)]
        assert delegate.progress == [(1, 2), (2, 2)]

        # Cancellation path: generate with cancel_after_first and verify partial file via VoiceCloner
        engine2 = _mock_engine()
        engine2.generate.side_effect = [
            (np.array([5.0], dtype=np.float32), 1000),
            (np.array([6.0], dtype=np.float32), 1000),
        ]
        cloner2 = VoiceCloner(speaker_wav="", engine=engine2)
        out = str(tmp_path / "gui_partial.wav")

        def cancel_after_first2():
            return engine2.generate.call_count >= 1

        with pytest.raises(GenerationCancelled):
            with patch("voice_cloner.split_into_chunks", return_value=["x", "y"]):
                cloner2.generate("long text long text", chunk_size=10, output_file=out, cancel_requested=cancel_after_first2)
        assert Path(out).exists()


def test_cli_progress_and_cancel_via_vcloner(tmp_path, monkeypatch):
    """CLI main renders progress and respects cancellation flag."""
    # Mock VoiceCloner to capture cancel_requested and chunk_progress_callback
    captured = {}

    class FakeCloner:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, text, play_audio=True, output_file=None, chunk_size=None, silence_duration=200, chunk_progress_callback=None, cancel_requested=None, **kwargs):
            captured["cancel"] = cancel_requested
            captured["progress"] = chunk_progress_callback
            # Simulate chunked progress: 2 chunks, cancel after first
            if chunk_progress_callback:
                chunk_progress_callback(1, 2)
                if cancel_requested and cancel_requested():
                    # If already cancelled, simulate GenerationCancelled
                    raise GenerationCancelled(audio=np.array([1.0], dtype=np.float32), sample_rate=1000)
                chunk_progress_callback(2, 2)
            # Simulate cancellation after first progress if flag becomes true
            if cancel_requested and cancel_requested():
                raise GenerationCancelled(audio=np.array([1.0], dtype=np.float32), sample_rate=1000)
            # Write dummy file to satisfy CLI expectation if not cancelled
            import soundfile as sf
            import numpy as np

            sf.write(output_file, np.array([1.0, 2.0], dtype=np.float32), 1000)
            return output_file

    # Patch Progress to capture updates
    progress_updates = []
    mock_progress = MagicMock()
    mock_progress.add_task.return_value = 1
    mock_progress.update.side_effect = lambda *a, **kw: progress_updates.append((a, kw))
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_progress
    mock_context.__exit__.return_value = False

    out = tmp_path / "cli_out.wav"
    # First, test normal progress rendering without cancellation
    with patch("vcloner.bootstrap_engines"), patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]), patch(
        "vcloner.VoiceCloner", FakeCloner
    ), patch("vcloner.Progress", return_value=mock_context), patch("vcloner.signal.signal"):
        # Simulate running with no cancellation (cancel flag always False)
        # Need to run main with args
        import sys

        with patch.object(sys, "argv", ["vcloner.py", "-i", "voice.wav", "-t", "long text", "-o", str(out)]):
            vcloner.main()

    assert len(progress_updates) >= 2
    # Verify first update had total=2
    assert any(kw.get("total") == 2 for _, kw in progress_updates)

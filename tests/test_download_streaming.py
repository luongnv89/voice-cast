"""Streamed download progress and bounded cancel/close wait coverage.

Regression tests for issue #57: download progress must stream through the
wired callback wrappers (not jump 0→100%) and shutdown waits must be bounded.
"""

import inspect
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from models.download_progress import DownloadProgress

BLOCK = 1024
CHUNKS = 8
FAKE_TOTAL = BLOCK * CHUNKS


class FakeTqdm:
    """Stand-in for ``tqdm`` mirroring TTS manage.py's usage pattern."""

    instances: list["FakeTqdm"] = []

    def __init__(self, *args, total=None, **kwargs):
        self.total = total or 0
        self.n = 0
        self.updates: list[int] = []
        FakeTqdm.instances.append(self)

    def update(self, n=1):
        self.n += n
        self.updates.append(self.n)
        return self.n

    def close(self):
        pass


def _intermediates(callbacks):
    """Return reports strictly between start (0 bytes) and completion."""
    return [c for c in callbacks if c.status == "downloading" and 0 < c.downloaded_bytes < FAKE_TOTAL]


def _fake_registry(cache_dir):
    return types.SimpleNamespace(get_cache_dir=lambda engine: Path(cache_dir))


# ---------------------------------------------------------------------------
# Coqui downloader
# ---------------------------------------------------------------------------


class _ChunkedFakeCoquiModelManager:
    """Reproduce TTS manage.py's chunked transfer through module-global tqdm."""

    def __init__(self, output_prefix=None, progress_bar=False):
        import TTS.utils.manage as manage

        self.output_prefix = Path(output_prefix) / "tts"
        self.progress_bar = progress_bar
        self._manage = manage

    def download_model(self, model_name):
        model_dir = self.output_prefix / model_name.replace("/", "--")
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.progress_bar:
            bar = self._manage.tqdm(total=FAKE_TOTAL)
            for _ in range(CHUNKS):
                bar.update(BLOCK)
        (model_dir / "model_file.pth").touch()
        config_path = model_dir / "config.json"
        config_path.touch()
        return model_dir / "model_file.pth", config_path, {"model_name": model_name}


def _install_fake_coqui_tts(monkeypatch, manager_cls=_ChunkedFakeCoquiModelManager):
    """Serve a fake TTS package whose manage module exposes tqdm."""
    manage_mod = types.ModuleType("TTS.utils.manage")
    manage_mod.ModelManager = manager_cls
    manage_mod.tqdm = FakeTqdm
    utils_mod = types.ModuleType("TTS.utils")
    utils_mod.manage = manage_mod
    tts_mod = types.ModuleType("TTS")
    tts_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "TTS", tts_mod)
    monkeypatch.setitem(sys.modules, "TTS.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "TTS.utils.manage", manage_mod)


class TestCoquiStreamedProgress:
    """Coqui downloads must emit intermediate progress during transfer."""

    @pytest.fixture(autouse=True)
    def _reset_fake_bars(self):
        FakeTqdm.instances = []
        yield
        FakeTqdm.instances = []

    def _download(self, monkeypatch, callback):
        from models.downloaders import coqui_downloader

        monkeypatch.setattr(coqui_downloader, "get_registry", lambda: _fake_registry("/tmp/coqui-cache"))
        _install_fake_coqui_tts(monkeypatch)
        return coqui_downloader.CoquiDownloader().download("coqui-xtts-v2", progress_callback=callback)

    def test_intermediate_progress_fires_during_transfer(self, tmp_path, monkeypatch):
        callbacks: list[DownloadProgress] = []
        self._download(monkeypatch, callbacks.append)

        assert len(callbacks) >= 3, f"expected start + >=1 intermediate + end, got {len(callbacks)} callbacks"
        intermediates = _intermediates(callbacks)
        assert len(intermediates) >= 1, "no intermediate progress between start and completion"
        downloaded = [c.downloaded_bytes for c in intermediates]
        assert downloaded == sorted(downloaded), "intermediate progress must be monotonic"
        assert callbacks[-1].status == "completed"

    def test_cancel_mid_transfer_propagates(self, tmp_path, monkeypatch):
        class CancelledError(Exception):
            pass

        calls = {"n": 0}

        def cancel_after_intermediate(progress: DownloadProgress):
            calls["n"] += 1
            if 0 < progress.downloaded_bytes < FAKE_TOTAL:
                raise CancelledError()

        with pytest.raises(CancelledError):
            self._download(monkeypatch, cancel_after_intermediate)
        # start + first intermediate (the wrapper throttles later ones out)
        assert calls["n"] >= 2, "cancel must be reachable mid-transfer, not only at start/end"


# ---------------------------------------------------------------------------
# Hugging Face downloaders (chatterbox, mlx-audio)
# ---------------------------------------------------------------------------


def _install_fake_hf_hub(monkeypatch, simulate=None):
    """Serve a fake huggingface_hub whose snapshot_download runs ``simulate``."""

    class _FakeHfHubModule(types.ModuleType):
        captured: dict = {}

        def snapshot_download(self, **kwargs):
            type(self).captured = kwargs
            if simulate is not None:
                simulate(kwargs)
            return "/fake/hf-cache"

    mod = _FakeHfHubModule("huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def _chunked_simulation(kwargs):
    """Drive the captured tqdm_class like HF hub would across a transfer."""
    bar_cls = kwargs.get("tqdm_class")
    assert bar_cls is not None, "snapshot_download must receive a tqdm_class bridge"
    bar = bar_cls(total=FAKE_TOTAL)
    for _ in range(CHUNKS):
        bar.update(BLOCK)


class TestHuggingFaceStreamedProgress:
    def test_chatterbox_intermediate_progress_fires(self, monkeypatch):
        from models.downloaders.chatterbox_downloader import ChatterboxDownloader

        _install_fake_hf_hub(monkeypatch, _chunked_simulation)
        monkeypatch.setattr(
            "models.downloaders.chatterbox_downloader.get_registry",
            lambda: _fake_registry("/tmp/chatterbox-cache"),
        )

        callbacks: list[DownloadProgress] = []
        ChatterboxDownloader().download("chatterbox-turbo", progress_callback=callbacks.append)

        intermediates = _intermediates(callbacks)
        assert len(intermediates) >= 1, "no intermediate progress between start and completion"
        downloaded = [c.downloaded_bytes for c in intermediates]
        assert downloaded == sorted(downloaded), "intermediate progress must be monotonic"
        assert callbacks[-1].status == "completed"

    def test_mlx_intermediate_progress_fires(self, monkeypatch):
        from models.downloaders.mlx_downloader import MlxDownloader

        _install_fake_hf_hub(monkeypatch, _chunked_simulation)
        monkeypatch.setattr(
            "models.downloaders.mlx_downloader.get_registry",
            lambda: _fake_registry("/tmp/mlx-cache"),
        )

        callbacks: list[DownloadProgress] = []
        MlxDownloader().download("mlx-kokoro", progress_callback=callbacks.append)

        intermediates = _intermediates(callbacks)
        assert len(intermediates) >= 1, "no intermediate progress between start and completion"
        assert callbacks[-1].status == "completed"

    def test_cancel_mid_transfer_propagates_chatterbox(self, monkeypatch):
        from models.downloaders.chatterbox_downloader import ChatterboxDownloader

        _install_fake_hf_hub(monkeypatch, _chunked_simulation)
        monkeypatch.setattr(
            "models.downloaders.chatterbox_downloader.get_registry",
            lambda: _fake_registry("/tmp/chatterbox-cache"),
        )

        class CancelledError(Exception):
            pass

        calls = {"n": 0}

        def cancel_after_intermediate(progress: DownloadProgress):
            calls["n"] += 1
            if 0 < progress.downloaded_bytes < FAKE_TOTAL:
                raise CancelledError()

        with pytest.raises(CancelledError):
            ChatterboxDownloader().download("chatterbox-turbo", progress_callback=cancel_after_intermediate)
        assert calls["n"] >= 2, "cancel must be reachable mid-transfer"


# ---------------------------------------------------------------------------
# GUI: bounded shutdown wait + reporter protocol adoption
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QThread  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        from PySide6.QtWidgets import QApplication

        app = QApplication([])
    return app


class _NeverEndingDownloadThread(QThread):
    """QThread that ignores interruptions until its stop event is set."""

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(0.02)


class TestBoundedShutdown:
    def test_default_timeout_is_finite(self):
        from gui.model_manager_widget import ModelManagerWidget

        sig = inspect.signature(ModelManagerWidget.shutdown_downloads)
        assert "timeout_ms" in sig.parameters, "shutdown_downloads must take a bounded timeout"
        timeout_ms = sig.parameters["timeout_ms"].default
        assert isinstance(timeout_ms, int) and 0 < timeout_ms < 60_000

    def test_shutdown_returns_within_bound_even_if_thread_ignores_interrupt(self, qapp):
        from gui.model_manager_widget import ModelManagerWidget

        thread = _NeverEndingDownloadThread("m")
        thread.start()
        try:
            stub = types.SimpleNamespace(
                _shutting_down=False,
                _download_threads={"m": thread},
            )
            started = time.monotonic()
            ModelManagerWidget.shutdown_downloads(stub, timeout_ms=250)
            elapsed = time.monotonic() - started
            assert elapsed < 5.0, f"shutdown blocked {elapsed:.1f}s — wait is unbounded"
            assert stub._shutting_down is True
        finally:
            thread._stop_event.set()
            thread.wait(2000)

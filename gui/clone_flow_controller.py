"""Clone flow controller for VoiceCloningApp.

Extracts the voice cloning generation flow from the main application window:
- CloneThread: QThread for running TTS generation without blocking the UI
- CloneFlowController: Manages the clone lifecycle (start, finish, error, reset)
- VoiceClonerCache: Caches VoiceCloner instances per engine to avoid reloading
  multi-GB model weights on every generation.
"""

import contextlib
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from voice_cloner import VoiceCloner


class VoiceClonerCache:
    """Cache VoiceCloner instances per (engine, speaker_wav) to avoid
    reloading multi-GB model weights on every generation.

    The cache is invalidated when the engine name or speaker_wav path changes.
    """

    def __init__(self):
        self._cache: dict[tuple[str, str], VoiceCloner] = {}

    def get(self, engine_name: str, speaker_wav: str) -> VoiceCloner:
        """Get or create a VoiceCloner for the given engine and speaker.

        Args:
            engine_name: Engine identifier (e.g., "coqui", "chatterbox-turbo").
            speaker_wav: Path to speaker reference audio file.

        Returns:
            A VoiceCloner instance (cached if possible).
        """
        key = (engine_name, speaker_wav)
        if key not in self._cache:
            self._cache[key] = VoiceCloner(speaker_wav=speaker_wav, engine=engine_name)
        return self._cache[key]

    def invalidate(self, engine_name: str | None = None, speaker_wav: str | None = None):
        """Invalidate cache entries.

        If engine_name is provided, invalidate all entries for that engine.
        If speaker_wav is provided, invalidate all entries for that speaker.
        If both are None, clear the entire cache.
        """
        if engine_name is None and speaker_wav is None:
            self._cache.clear()
            return

        keys_to_remove = [
            key
            for key in self._cache
            if (engine_name is not None and key[0] == engine_name)
            or (speaker_wav is not None and key[1] == speaker_wav)
        ]
        for key in keys_to_remove:
            del self._cache[key]

    @property
    def size(self) -> int:
        return len(self._cache)


class CloneThread(QThread):
    """Thread for running TTS generation without blocking the UI."""

    finished = Signal(str, str)
    error_occurred = Signal(str)
    stage_changed = Signal(str)

    def __init__(
        self,
        text: str,
        voice_path: str,
        engine_name: str,
        engine_params: dict,
        voice_cloner: VoiceCloner | None = None,
    ):
        super().__init__()
        self.text = text
        self.voice_path = voice_path
        self.engine_name = engine_name
        self.engine_params = engine_params
        self._voice_cloner = voice_cloner
        self.output_path = None

    def run(self):
        try:
            # Create temporary output directory
            output_dir = Path(tempfile.gettempdir()) / "voice_cloning"
            output_dir.mkdir(exist_ok=True)

            # Use engine name + timestamp for readable filenames
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_dir / f"{self.engine_name}_{ts}.wav"

            # Use cached cloner or create a new one
            voice_cloner = self._voice_cloner
            if voice_cloner is None:
                self.stage_changed.emit("Loading model...")
                voice_cloner = VoiceCloner(speaker_wav=self.voice_path, engine=self.engine_name)

            # Generate audio
            self.stage_changed.emit("Synthesizing...")
            voice_cloner.say(
                self.text,
                play_audio=False,
                save_audio=True,
                output_file=str(self.output_path),
                **self.engine_params,
            )
            self.finished.emit(str(self.output_path), self.text)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CloneFlowController(QObject):
    """Manages the clone generation lifecycle.

    Coordinates UI state, validation, thread management, and result handling
    for the voice cloning flow. Delegated to by VoiceCloningApp.
    """

    def __init__(self, ui_delegate):
        """Initialize the controller.

        Args:
            ui_delegate: Object providing UI access with these methods:
                - warning(title, message)
                - critical(title, message)
                - question(title, message, default_button)
                - disable_generate()
                - enable_generate()
                - set_generate_text(text)
                - show_generate()
                - hide_play_save()
                - show_play_save()
                - disable_voice_select()
                - enable_voice_select()
                - disable_engine_combo()
                - enable_engine_combo()
                - show_progress()
                - hide_progress()
                - get_text_input() -> str
                - get_voice_path() -> str | None
                - get_engine_name() -> str
                - get_engine_params() -> dict
                - is_voice_required() -> bool
                - is_model_installed(model_id) -> bool
                - get_model_id_for_engine(engine_name) -> str
                - switch_to_model_manager()
                - is_thread_running() -> bool
        """
        super().__init__()
        self._ui = ui_delegate
        self._thread = None
        self._temp_voice_file = None
        self._cloner_cache = VoiceClonerCache()

    @property
    def is_running(self) -> bool:
        """Whether a generation is currently in progress."""
        return self._thread is not None and self._thread.isRunning()

    def terminate(self):
        """Terminate the generation thread if running.

        Blocks until the thread has finished terminating.
        """
        if self._thread is not None and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait(5000)  # Wait up to 5 seconds
            self._thread = None

    def start(self) -> bool:
        """Start the voice cloning process.

        Returns:
            True if generation started, False if validation failed.
        """
        # Validate voice file requirement
        if self._ui.is_voice_required() and not self._ui.get_voice_path():
            self._ui.warning(
                "Missing Voice Reference",
                "Please select an audio file (.wav, .mp3, .ogg, .flac) as voice reference.",
            )
            return False

        # Validate text input
        text = self._ui.get_text_input().strip()
        if not text:
            self._ui.warning(self, "Missing Text", "Please enter text to generate audio.")
            return False

        # Check if a thread is already running
        if self.is_running:
            self._ui.warning(
                "Generation In Progress",
                "Please wait for the current generation to complete.",
            )
            return False

        # Get engine and model info
        engine_name = self._ui.get_engine_name()
        model_id = self._ui.get_model_id_for_engine(engine_name)

        # Check if model is installed
        if not self._ui.is_model_installed(model_id):
            reply = self._ui.question(
                "Model Not Installed",
                f"The model '{model_id}' is not installed.\n\n"
                "Would you like to go to the Model Manager to download it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._ui.switch_to_model_manager()
            return False

        # Prepare voice file
        temp_voice_path = ""
        if self._ui.is_voice_required():
            try:
                temp_voice = Path(tempfile.gettempdir()) / (
                    f"voice_{uuid.uuid4().hex}{Path(self._ui.get_voice_path()).suffix}"
                )
                shutil.copyfile(self._ui.get_voice_path(), str(temp_voice))
                self._temp_voice_file = str(temp_voice)
                temp_voice_path = str(temp_voice)
            except (OSError, PermissionError, MemoryError) as e:
                self._reset_ui()
                self._ui.critical("File Error", f"Cannot read voice file: {e}")
                return False

        # Disable UI during processing
        self._ui.disable_generate()
        self._ui.set_generate_text("Generating audio...")
        self._ui.hide_play_save()
        self._ui.disable_voice_select()
        self._ui.disable_engine_combo()
        self._ui.show_progress()

        # Get engine parameters and start thread
        engine_params = self._ui.get_engine_params() if hasattr(self._ui, "get_engine_params") else {}

        # Get cached VoiceCloner (or create new one)
        voice_cloner = self._cloner_cache.get(engine_name, temp_voice_path)

        self._thread = CloneThread(
            text=text,
            voice_path=temp_voice_path,
            engine_name=engine_name,
            engine_params=engine_params,
            voice_cloner=voice_cloner,
        )
        self._connect_thread_signals(self._thread)
        self._thread.start()
        return True

    @staticmethod
    def _connect_queued(signal, slot):
        signal.connect(slot, Qt.ConnectionType.QueuedConnection)

    def _connect_thread_signals(self, thread: CloneThread):
        """Connect worker signals to GUI-thread handlers."""
        self._connect_queued(thread.finished, self._on_finished)
        self._connect_queued(thread.error_occurred, self._on_error)
        self._connect_queued(thread.stage_changed, self._on_stage_changed)

    @Slot(str)
    def _on_stage_changed(self, stage: str):
        """Handle stage change during generation."""
        self._ui.set_stage_text(stage)

    @Slot(str, str)
    def _on_finished(self, output_path: str, text: str):
        """Handle successful generation completion."""
        self._ui.current_audio = output_path
        self._cleanup_temp()
        self._reset_ui()
        self._ui.show_play_save()
        self._ui.info(
            "Generation Complete",
            "Audio generation completed.\n\nUse Save Audio to choose where to save it.",
        )

    @Slot(str)
    def _on_error(self, message: str):
        """Handle generation error."""
        self._cleanup_temp()
        self._reset_ui()
        self._ui.critical("Generation Error", f"Failed to generate audio:\n\n{message}")

    def _reset_ui(self):
        """Reset UI to normal state after generation."""
        self._ui.enable_generate()
        self._ui.set_generate_text("Generate Audio")
        self._ui.enable_voice_select()
        self._ui.enable_engine_combo()
        self._ui.hide_progress()

    def _cleanup_temp(self):
        """Clean up temporary voice file."""
        if self._temp_voice_file:
            with contextlib.suppress(OSError):
                Path(self._temp_voice_file).unlink(missing_ok=True)
            self._temp_voice_file = None

    def engine_changed(self, engine_name: str):
        """Invalidate cloner cache when engine changes.

        Args:
            engine_name: The new engine name that was selected.
        """
        self._cloner_cache.invalidate(engine_name=engine_name)

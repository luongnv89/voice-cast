"""
VoiceCast - Voice Cloning Application.

Main application window with voice cloning and model management tabs.
"""

import contextlib
import sys
import tempfile
import uuid
from pathlib import Path

import pygame
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from gui.engine_controls import EngineControlsFactory
from gui.model_manager_widget import ModelManagerWidget
from gui.styled_widgets import (
    StyledButton,
    StyledComboBox,
    StyledGroupBox,
    StyledLabel,
    StyledTabWidget,
    StyledTextEdit,
)
from gui.theme import (
    SPACING,
    ThemeMode,
    apply_theme,
    get_theme_manager,
)
from models.model_registry import get_registry
from tts_factory import TTSFactory
from voice_cloner import VoiceCloner


class CloneThread(QThread):
    """Thread for running TTS generation without blocking the UI."""

    finished = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, text: str, voice_path: str, engine_name: str, engine_params: dict):
        super().__init__()
        self.text = text
        self.voice_path = voice_path
        self.engine_name = engine_name
        self.engine_params = engine_params
        self.output_path = None

    def run(self):
        try:
            # Create temporary output directory
            output_dir = Path(tempfile.gettempdir()) / "voice_cloning"
            output_dir.mkdir(exist_ok=True)

            # Generate unique filename
            self.output_path = output_dir / f"output_{uuid.uuid4().hex}.wav"

            # Create VoiceCloner with selected engine
            voice_cloner = VoiceCloner(speaker_wav=self.voice_path, engine=self.engine_name)

            # Generate audio
            voice_cloner.say(
                self.text, play_audio=False, save_audio=True, output_file=str(self.output_path), **self.engine_params
            )
            self.finished.emit(str(self.output_path), self.text)
        except Exception as e:
            self.error_occurred.emit(str(e))


class VoiceCloningApp(QMainWindow):
    """Main application window for VoiceCast."""

    def __init__(self):
        super().__init__()
        self.current_audio = None
        self.voice_path = None
        self.engine_controls = None
        self.clone_thread = None
        self._temp_voice_file = None
        self._registry = get_registry()
        self._theme_manager = get_theme_manager()
        self._current_engine_requires_voice = True  # Track if current engine needs voice file

        self._setup_menu_bar()
        self._init_ui()
        self.setWindowTitle("VoiceCast")
        self.setMinimumSize(750, 650)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.jpg")))

        # Initialize pygame mixer for audio
        self._audio_available = True
        try:
            pygame.mixer.init()
        except pygame.error as e:
            self._audio_available = False
            print(f"Warning: Audio playback unavailable — pygame.mixer.init() failed: {e}")

        # Connect theme changes
        self._theme_manager.theme_changed.connect(self._on_theme_changed)

    def _setup_menu_bar(self):
        """Set up the application menu bar."""
        menubar = self.menuBar()

        # View menu
        view_menu = menubar.addMenu("View")

        # Theme submenu
        theme_menu = QMenu("Theme", self)
        view_menu.addMenu(theme_menu)

        # Theme options
        self._theme_actions = {}

        light_action = QAction("Light", self)
        light_action.setCheckable(True)
        light_action.triggered.connect(lambda: self._set_theme(ThemeMode.LIGHT))
        theme_menu.addAction(light_action)
        self._theme_actions[ThemeMode.LIGHT] = light_action

        dark_action = QAction("Dark", self)
        dark_action.setCheckable(True)
        dark_action.triggered.connect(lambda: self._set_theme(ThemeMode.DARK))
        theme_menu.addAction(dark_action)
        self._theme_actions[ThemeMode.DARK] = dark_action

        theme_menu.addSeparator()

        system_action = QAction("System", self)
        system_action.setCheckable(True)
        system_action.triggered.connect(lambda: self._set_theme(ThemeMode.SYSTEM))
        theme_menu.addAction(system_action)
        self._theme_actions[ThemeMode.SYSTEM] = system_action

        # Set initial check state
        self._update_theme_menu_state()

    def _set_theme(self, mode: ThemeMode):
        """Set the application theme."""
        self._theme_manager.set_theme(mode)
        self._update_theme_menu_state()
        apply_theme(QApplication.instance())

    def _update_theme_menu_state(self):
        """Update theme menu check states."""
        current = self._theme_manager.current_mode
        for mode, action in self._theme_actions.items():
            action.setChecked(mode == current)

    def _on_theme_changed(self, mode: ThemeMode):
        """Handle theme change."""
        self._update_theme_menu_state()

    def closeEvent(self, event):
        """Clean up resources when window closes."""
        # Stop any playing audio
        if self._audio_available:
            pygame.mixer.music.stop()
            pygame.mixer.quit()

        # Wait for thread to finish
        if self.clone_thread and self.clone_thread.isRunning():
            self.clone_thread.quit()
            self.clone_thread.wait(1000)

        # Wait for model downloads before widgets are destroyed.
        if self.model_manager:
            self.model_manager.shutdown_downloads()

        # Clean up temp files
        self._cleanup_temp_files()

        super().closeEvent(event)

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self._temp_voice_file and Path(self._temp_voice_file).exists():
            with contextlib.suppress(OSError):
                Path(self._temp_voice_file).unlink()
        if self.current_audio and Path(self.current_audio).exists():
            with contextlib.suppress(OSError):
                Path(self.current_audio).unlink()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create tab widget for main content
        self.tab_widget = StyledTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create Voice Cloning tab
        voice_tab = QWidget()
        self._setup_voice_tab(voice_tab)
        self.tab_widget.addTab(voice_tab, "Voice Cloning")

        # Create Model Manager tab
        self.model_manager = ModelManagerWidget()
        self.tab_widget.addTab(self.model_manager, "Model Manager")

    def _setup_voice_tab(self, tab_widget: QWidget):
        """Set up the voice cloning tab."""
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        layout.setSpacing(SPACING.md)

        # Model selection
        model_group = StyledGroupBox("TTS Engine")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(SPACING.sm)

        engine_row = QHBoxLayout()
        engine_label = StyledLabel("Engine:", role="secondary")
        engine_row.addWidget(engine_label)

        self.engine_combo = StyledComboBox()
        # Populate engine list dynamically from factory
        self._populate_engine_list()
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_row.addWidget(self.engine_combo, stretch=1)
        model_layout.addLayout(engine_row)

        # Container for engine-specific controls
        self.controls_container = QVBoxLayout()
        self.controls_container.setContentsMargins(0, SPACING.sm, 0, 0)
        model_layout.addLayout(self.controls_container)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Text input
        text_group = StyledGroupBox("Text to Synthesize")
        text_layout = QVBoxLayout()
        self.text_input = StyledTextEdit()
        self.text_input.setPlaceholderText("Enter text to generate audio...")
        self.text_input.setMinimumHeight(120)
        text_layout.addWidget(self.text_input)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # Voice file selection (hidden for preset voice engines)
        self.voice_group = StyledGroupBox("Voice Reference")
        voice_layout = QHBoxLayout()
        voice_layout.setSpacing(SPACING.md)

        self.btn_select_voice = StyledButton("Select Voice File", variant="secondary")
        self.btn_select_voice.clicked.connect(self.select_voice_file)
        voice_layout.addWidget(self.btn_select_voice)

        self.voice_label = StyledLabel("No voice file selected", role="muted")
        self.voice_label.setWordWrap(True)
        voice_layout.addWidget(self.voice_label, stretch=1)

        self.voice_group.setLayout(voice_layout)
        layout.addWidget(self.voice_group)

        # Initialize with default engine's controls
        default_engine = self._get_default_engine()
        self._update_engine_controls(default_engine)

        # Set the combo box to the default engine
        for i in range(self.engine_combo.count()):
            if self.engine_combo.itemData(i) == default_engine:
                self.engine_combo.setCurrentIndex(i)
                break

        # Generate button
        self.btn_generate = StyledButton("Generate Audio", variant="primary")
        self.btn_generate.setMinimumHeight(48)
        self.btn_generate.clicked.connect(self.start_cloning)
        layout.addWidget(self.btn_generate)

        # Activity indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Result controls
        result_layout = QHBoxLayout()
        result_layout.setSpacing(SPACING.md)

        self.btn_play = StyledButton("Play", variant="secondary")
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_play.hide()

        self.btn_save = StyledButton("Save Audio", variant="secondary")
        self.btn_save.clicked.connect(self.save_audio)
        self.btn_save.hide()

        result_layout.addWidget(self.btn_play)
        result_layout.addWidget(self.btn_save)
        result_layout.addStretch()
        layout.addLayout(result_layout)

        layout.addStretch()

    def _populate_engine_list(self):
        """Populate the engine combo box from factory."""
        # Get all registered engines (not just available ones, for visibility)
        for engine_name in TTSFactory.available_engines():
            try:
                metadata = TTSFactory.get_engine_metadata(engine_name)
                display_name = metadata.get("display_name", engine_name)
                self.engine_combo.addItem(display_name, engine_name)
            except ValueError:
                pass

    def _get_default_engine(self) -> str:
        """Get the default engine for the current platform."""
        try:
            return TTSFactory.get_default_engine()
        except RuntimeError:
            # Fallback to first available if no engines available
            available = TTSFactory.available_engines()
            return available[0] if available else "coqui"

    def _update_engine_controls(self, engine_name: str):
        """Update the engine-specific control widgets."""
        # Remove existing controls
        if self.engine_controls:
            self.controls_container.removeWidget(self.engine_controls)
            self.engine_controls.deleteLater()

        # Create new controls for selected engine
        self.engine_controls = EngineControlsFactory.create(engine_name)
        self.controls_container.addWidget(self.engine_controls)

        # Update voice file visibility based on engine requirements
        try:
            metadata = TTSFactory.get_engine_metadata(engine_name)
            requires_voice = metadata.get("requires_reference_audio", True)
            self._current_engine_requires_voice = requires_voice
            self.voice_group.setVisible(requires_voice)
        except ValueError:
            # Unknown engine, assume it needs voice file
            self._current_engine_requires_voice = True
            self.voice_group.setVisible(True)

    def on_engine_changed(self, index: int):
        """Handle engine selection change."""
        engine_name = self.engine_combo.currentData()
        self._update_engine_controls(engine_name)

    def select_voice_file(self):
        """Open file dialog to select voice reference file."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3 *.ogg *.flac)")
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.voice_path = files[0]
                self.voice_label.setText(Path(self.voice_path).name)
                self.voice_label.set_role("primary")

    def _get_model_id_for_engine(self, engine_name: str) -> str:
        """Get the model ID for an engine name."""
        return self._registry.get_model_id_for_engine(engine_name)

    def start_cloning(self):
        """Start the voice cloning process."""
        # Check if voice file is needed for current engine
        if self._current_engine_requires_voice and not self.voice_path:
            QMessageBox.warning(
                self,
                "Missing Voice Reference",
                "Please select an audio file (.wav, .mp3, .ogg, .flac) as voice reference.",
            )
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing Text", "Please enter text to generate audio.")
            return

        # Check if a thread is already running
        if self.clone_thread and self.clone_thread.isRunning():
            QMessageBox.warning(self, "Generation In Progress", "Please wait for the current generation to complete.")
            return

        # Check if the model is installed
        engine_name = self.engine_combo.currentData()
        model_id = self._get_model_id_for_engine(engine_name)

        if not self._registry.is_installed(model_id):
            reply = QMessageBox.question(
                self,
                "Model Not Installed",
                f"The model '{model_id}' is not installed.\n\n"
                "Would you like to go to the Model Manager to download it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.tab_widget.setCurrentWidget(self.model_manager)
            return

        # Disable UI during processing
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating audio...")
        self.btn_play.hide()
        self.btn_save.hide()
        self.btn_select_voice.setEnabled(False)
        self.engine_combo.setEnabled(False)
        self.progress_bar.show()

        # Get selected engine and parameters
        engine_name = self.engine_combo.currentData()
        engine_params = self.engine_controls.get_parameters() if self.engine_controls else {}

        # Handle voice file based on engine requirements
        temp_voice_path = ""
        if self._current_engine_requires_voice:
            # Create temporary copy of voice file
            try:
                temp_voice = Path(tempfile.gettempdir()) / f"voice_{uuid.uuid4().hex}{Path(self.voice_path).suffix}"
                temp_voice.write_bytes(Path(self.voice_path).read_bytes())
                self._temp_voice_file = str(temp_voice)
                temp_voice_path = str(temp_voice)
            except (OSError, PermissionError, MemoryError) as e:
                self._reset_ui_state()
                QMessageBox.critical(self, "File Error", f"Cannot read voice file: {e}")
                return
        else:
            # For preset voice engines, use a placeholder path
            temp_voice_path = ""

        # Start cloning thread
        self.clone_thread = CloneThread(
            text=text, voice_path=temp_voice_path, engine_name=engine_name, engine_params=engine_params
        )
        self.clone_thread.finished.connect(self.on_cloning_finished)
        self.clone_thread.error_occurred.connect(self.on_cloning_error)
        self.clone_thread.start()

    def on_cloning_finished(self, output_path: str, text: str):
        """Handle successful cloning completion."""
        self.current_audio = output_path
        self._reset_ui_state()
        self.btn_play.show()
        self.btn_save.show()

    @Slot(str)
    def on_cloning_error(self, message: str):
        """Handle cloning error."""
        self._reset_ui_state()
        QMessageBox.critical(self, "Generation Error", f"Failed to generate audio:\n\n{message}")

    def _reset_ui_state(self):
        """Reset UI to normal state after generation."""
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate Audio")
        self.btn_select_voice.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.progress_bar.hide()

    def play_audio(self):
        """Play the generated audio."""
        if not self._audio_available:
            QMessageBox.warning(self, "Playback Unavailable", "Audio playback is not available on this system.")
            return
        if self.current_audio and Path(self.current_audio).exists():
            # Stop any currently playing audio first
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.current_audio)
            pygame.mixer.music.play()

    def save_audio(self):
        """Save the generated audio to a file."""
        if self.current_audio:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio File", f"cloned_voice_{uuid.uuid4().hex[:8]}.wav", "Wave Files (*.wav)"
            )
            if file_path:
                import shutil

                shutil.copy2(self.current_audio, file_path)
                QMessageBox.information(self, "Saved", f"Audio file saved to:\n{file_path}")


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("VoiceCast")
    app.setOrganizationName("VoiceCast")

    # Apply theme
    apply_theme(app)

    # Create and show main window
    window = VoiceCloningApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

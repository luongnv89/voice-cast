"""
VoiceCast - Voice Cloning Application.

Main application window with voice cloning and model management tabs.
"""

import contextlib
import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui.audio_playback import AudioPlaybackController
from gui.clone_flow_controller import CloneFlowController
from gui.engine_controls import EngineControlsFactory
from gui.model_manager_widget import ModelManagerWidget
from gui.styled_widgets import (
    StyledButton,
    StyledComboBox,
    StyledGroupBox,
    StyledLabel,
    StyledProgressBar,
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
from tts_factory import TTSFactory, bootstrap_engines


class VoiceCloningApp(QMainWindow):
    """Main application window for VoiceCast."""

    def __init__(self):
        super().__init__()
        self.voice_path = None
        self.engine_controls = None
        self._generate_enabled = False
        self._registry = get_registry()
        self._theme_manager = get_theme_manager()
        self._current_engine_requires_voice = True

        # Extracted collaborators
        self._audio = AudioPlaybackController()
        self._clone_flow = CloneFlowController(self)
        self._close_pending = False
        self._close_retry_queued = False
        self._clone_flow.worker_finished.connect(self._on_clone_worker_finished)

        self._setup_menu_bar()
        self._init_ui()
        self._setup_status_bar()
        self._setup_keyboard_shortcuts()
        self.setWindowTitle("VoiceCast")
        self.setMinimumSize(750, 650)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.jpg")))

        # Connect theme changes
        self._theme_manager.theme_changed.connect(self._on_theme_changed)

    # ------------------------------------------------------------------ #
    # Menu bar
    # ------------------------------------------------------------------ #

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
        self._apply_stage_label_font()
        self._update_status_bar()

    def _apply_stage_label_font(self):
        """Keep the generation stage label emphasized across theme changes."""
        if not hasattr(self, "_stage_label"):
            return
        font = self._stage_label.font()
        font.setBold(True)
        font.setPixelSize(14)
        self._stage_label.setFont(font)

    def _on_tab_changed(self, index: int):
        """Refresh model status when Model Manager tab is activated."""
        if self.tab_widget.widget(index) == self.model_manager:
            self.model_manager.refresh_models()

    # ------------------------------------------------------------------ #
    # Window lifecycle
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        """Close only after any in-flight generation has exited naturally."""
        if self._clone_flow.has_worker:
            self._close_pending = True
            event.ignore()
            self._clone_flow.terminate()
            return

        self._close_pending = False

        # Stop any playing audio
        if self._audio.audio_available:
            with contextlib.suppress(Exception):
                import pygame

                pygame.mixer.music.stop()
                pygame.mixer.quit()

        # Wait for model downloads before widgets are destroyed.
        if self.model_manager:
            self.model_manager.shutdown_downloads()

        # Clean up temp files
        self._cleanup_temp_files()

        super().closeEvent(event)

    @Slot()
    def _on_clone_worker_finished(self):
        """Retry a close request once the generation worker is fully stopped."""
        if not self._close_pending or self._clone_flow.has_worker or self._close_retry_queued:
            return
        self._close_retry_queued = True
        QTimer.singleShot(0, self._retry_close)

    def _retry_close(self):
        self._close_retry_queued = False
        if self._close_pending and not self._clone_flow.has_worker:
            self.close()

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self._clone_flow._temp_voice_file and Path(self._clone_flow._temp_voice_file).exists():
            with contextlib.suppress(OSError):
                Path(self._clone_flow._temp_voice_file).unlink()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

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
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

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
        self.engine_combo.setToolTip("Choose the TTS engine to use for generation")
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
        self.text_input.setMinimumHeight(160)
        self.text_input.setToolTip("Enter the text you want to synthesize into speech")
        self.text_input.textChanged.connect(self._update_generate_state)
        text_layout.addWidget(self.text_input)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # Voice file selection (hidden for preset voice engines)
        self.voice_group = StyledGroupBox("Voice Reference")
        voice_layout = QHBoxLayout()
        voice_layout.setSpacing(SPACING.md)

        self.btn_select_voice = StyledButton("Select Voice File", variant="secondary")
        self.btn_select_voice.setAccessibleName("Select voice file")
        self.btn_select_voice.setToolTip("Select a reference audio file (WAV, MP3, OGG, FLAC) for voice cloning")
        self.btn_select_voice.clicked.connect(self.select_voice_file)
        self.btn_select_voice.clicked.connect(self._update_generate_state)
        voice_layout.addWidget(self.btn_select_voice)

        self.voice_label = StyledLabel("No voice file selected", role="muted")
        self.voice_label.setToolTip("Selected voice reference file for cloning")
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
        self.btn_generate.setAccessibleName("Generate audio")
        self.btn_generate.setToolTip("Generate audio from the entered text and selected voice (Ctrl+G)")
        self.btn_generate.setMinimumHeight(48)
        self.btn_generate.clicked.connect(self._on_generate_clicked)
        self.btn_generate.setEnabled(False)
        layout.addWidget(self.btn_generate)

        # Inline hint for missing inputs
        self._generate_hint = StyledLabel(
            "Select a voice file and enter text to generate audio",
            role="muted",
        )
        self._generate_hint.setWordWrap(True)
        self._generate_hint.hide()
        layout.addWidget(self._generate_hint)
        self.progress_bar = StyledProgressBar()
        self.progress_bar.setAccessibleName("Generation progress")
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self._stage_label = StyledLabel("", role="info")
        self._stage_label.setAccessibleName("Generation stage")
        self._stage_label.setToolTip("Current generation stage")
        self._stage_label.setWordWrap(True)
        self._apply_stage_label_font()
        self._stage_label.hide()
        layout.addWidget(self._stage_label)

        # Result controls
        result_layout = QHBoxLayout()
        result_layout.setSpacing(SPACING.md)

        self.btn_play = StyledButton("Play", variant="secondary")
        self.btn_play.setAccessibleName("Play audio")
        self.btn_play.setToolTip("Play the generated audio (Ctrl+P)")
        self.btn_play.clicked.connect(self._on_play_clicked)
        self.btn_play.hide()

        self.btn_save = StyledButton("Save Audio", variant="secondary")
        self.btn_save.setAccessibleName("Save audio")
        self.btn_save.setToolTip("Save the generated audio to a file (Ctrl+S)")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_save.hide()

        result_layout.addWidget(self.btn_play)
        result_layout.addWidget(self.btn_save)
        result_layout.addStretch()
        layout.addLayout(result_layout)

        # Initialize generate button state based on current inputs
        self._update_generate_state()

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Status bar
    # ------------------------------------------------------------------ #

    def _setup_status_bar(self):
        """Set up the application status bar."""
        status_bar = QStatusBar()
        status_bar.setContentsMargins(SPACING.sm, 0, SPACING.sm, 0)
        self.setStatusBar(status_bar)

        # Engine status indicator
        self._status_engine = StyledLabel("Engine: —", role="muted")
        status_bar.addWidget(self._status_engine, stretch=1)

        # Model status indicator
        self._status_model = StyledLabel("", role="muted")
        status_bar.addWidget(self._status_model)

        # Theme indicator
        self._status_theme = StyledLabel("", role="muted")
        status_bar.addWidget(self._status_theme)

        self._update_status_bar()

    def _update_status_bar(self):
        """Update status bar with current engine and model info."""
        engine_name = self.engine_combo.currentData()
        if engine_name:
            try:
                metadata = TTSFactory.get_engine_metadata(engine_name)
                display_name = metadata.get("display_name", engine_name)
                self._status_engine.setText(f"Engine: {display_name}")
            except ValueError:
                self._status_engine.setText(f"Engine: {engine_name}")

        if engine_name:
            try:
                model_id = self.get_model_id_for_engine(engine_name)
                if self.is_model_installed(model_id):
                    self._status_model.setText("✓ Model installed")
                    self._status_model.set_role("success")
                else:
                    self._status_model.setText("○ Model not installed")
                    self._status_model.set_role("muted")
            except ValueError:
                self._status_model.setText("")
        else:
            self._status_model.setText("")

        tm = get_theme_manager()
        mode_map = {ThemeMode.LIGHT: "☀ Light", ThemeMode.DARK: "☾ Dark", ThemeMode.SYSTEM: "⚙ System"}
        self._status_theme.setText(mode_map.get(tm.current_mode, ""))

    # ------------------------------------------------------------------ #
    # Keyboard shortcuts
    # ------------------------------------------------------------------ #

    def _setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for primary actions."""
        # Generate: Ctrl+G
        shortcut_generate = QKeySequence("Ctrl+G")
        action_generate = QAction(self)
        action_generate.setShortcut(shortcut_generate)
        action_generate.triggered.connect(self._on_generate_clicked)
        self.addAction(action_generate)

        # Save: Ctrl+S
        shortcut_save = QKeySequence("Ctrl+S")
        action_save = QAction(self)
        action_save.setShortcut(shortcut_save)
        action_save.triggered.connect(self._on_save_clicked)
        self.addAction(action_save)

        # Play: Ctrl+P
        shortcut_play = QKeySequence("Ctrl+P")
        action_play = QAction(self)
        action_play.setShortcut(shortcut_play)
        action_play.triggered.connect(self._on_play_clicked)
        self.addAction(action_play)

        # Tab 1: Ctrl+1
        shortcut_tab1 = QKeySequence("Ctrl+1")
        action_tab1 = QAction(self)
        action_tab1.setShortcut(shortcut_tab1)
        action_tab1.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        self.addAction(action_tab1)

        # Tab 2: Ctrl+2
        shortcut_tab2 = QKeySequence("Ctrl+2")
        action_tab2 = QAction(self)
        action_tab2.setShortcut(shortcut_tab2)
        action_tab2.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.addAction(action_tab2)

    # ------------------------------------------------------------------ #
    # Engine management
    # ------------------------------------------------------------------ #

    def _populate_engine_list(self):
        """Populate the engine combo box from factory."""
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
            available = TTSFactory.available_engines()
            return available[0] if available else "coqui"

    def _update_engine_controls(self, engine_name: str):
        """Update the engine-specific control widgets."""
        if self.engine_controls:
            self.controls_container.removeWidget(self.engine_controls)
            self.engine_controls.deleteLater()

        self.engine_controls = EngineControlsFactory.create(engine_name)
        self.controls_container.addWidget(self.engine_controls)

        try:
            metadata = TTSFactory.get_engine_metadata(engine_name)
            requires_voice = metadata.get("requires_reference_audio", True)
            self._current_engine_requires_voice = requires_voice
            self.voice_group.setVisible(requires_voice)
        except ValueError:
            self._current_engine_requires_voice = True
            self.voice_group.setVisible(True)

    def on_engine_changed(self, index: int):
        """Handle engine selection change."""
        engine_name = self.engine_combo.currentData()
        self._update_engine_controls(engine_name)
        self._clone_flow.engine_changed(engine_name)
        self._update_status_bar()

    # ------------------------------------------------------------------ #
    # Voice file selection
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Clone flow delegate (for CloneFlowController)
    # ------------------------------------------------------------------ #

    def warning(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def critical(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def question(self, title: str, message: str, buttons, default_button):
        return QMessageBox.question(self, title, message, buttons, default_button)

    def info(self, title: str, message: str):
        QMessageBox.information(self, title, message)

    def disable_generate(self):
        self.btn_generate.setEnabled(False)

    def enable_generate(self):
        self.btn_generate.setEnabled(True)

    def set_generate_text(self, text: str):
        self.btn_generate.setText(text)

    def hide_play_save(self):
        self.btn_play.hide()
        self.btn_save.hide()

    def show_play_save(self):
        self.btn_play.show()
        self.btn_save.show()

    def disable_voice_select(self):
        self.btn_select_voice.setEnabled(False)

    def enable_voice_select(self):
        self.btn_select_voice.setEnabled(True)

    def disable_engine_combo(self):
        self.engine_combo.setEnabled(False)

    def enable_engine_combo(self):
        self.engine_combo.setEnabled(True)

    def show_progress(self):
        self.progress_bar.show()

    def hide_progress(self):
        self.progress_bar.hide()
        self._stage_label.hide()

    def set_stage_text(self, text: str):
        """Set the generation stage text and accessible status."""
        self._stage_label.setText(text)
        self._stage_label.setAccessibleName(f"Generation stage: {text}" if text else "Generation stage")
        self._stage_label.show()

    def get_text_input(self) -> str:
        return self.text_input.toPlainText()

    def get_voice_path(self):
        return self.voice_path

    def get_engine_name(self) -> str:
        return self.engine_combo.currentData()

    def get_engine_params(self) -> dict:
        if self.engine_controls:
            return self.engine_controls.get_parameters()
        return {}

    @property
    def is_voice_required(self) -> bool:
        return self._current_engine_requires_voice

    @property
    def is_thread_running(self) -> bool:
        return self._clone_flow.is_running

    def is_model_installed(self, model_id: str) -> bool:
        return self._registry.is_installed(model_id)

    def get_model_id_for_engine(self, engine_name: str) -> str:
        return self._registry.get_model_id_for_engine(engine_name)

    def switch_to_model_manager(self):
        self.tab_widget.setCurrentWidget(self.model_manager)

    @property
    def current_audio(self):
        return self._audio.current_audio

    @current_audio.setter
    def current_audio(self, value):
        self._audio.current_audio = value

    # ------------------------------------------------------------------ #
    # Signal handlers (wire up to collaborators)
    # ------------------------------------------------------------------ #

    @Slot()
    def _update_generate_state(self):
        """Enable/disable Generate button based on input validity.

        Shows an inline hint naming what input is missing when disabled.
        """
        text = self.text_input.toPlainText().strip()
        voice_valid = not self._current_engine_requires_voice or self.voice_path is not None
        valid = bool(text) and voice_valid

        if valid:
            self.btn_generate.setEnabled(True)
            self._generate_hint.hide()
        else:
            self.btn_generate.setEnabled(False)
            parts = []
            if not text:
                parts.append("enter text")
            if not voice_valid:
                parts.append("select a voice file")
            self._generate_hint.setText(f"Click Generate to start — please {' and '.join(parts)}.")
            self._generate_hint.show()

    def _on_generate_clicked(self):
        """Start voice cloning process."""
        self._clone_flow.start()

    @Slot()
    def _on_play_clicked(self):
        """Play the generated audio."""
        self._audio.play(self)

    @Slot()
    def _on_save_clicked(self):
        """Save the generated audio."""
        self._audio.save(self)

    def get_save_filename(self, default_name: str, file_filter: str):
        """Delegate for QFileDialog.getSaveFileName."""
        return QFileDialog.getSaveFileName(self, "Save Audio File", default_name, file_filter)


def main():
    """Application entry point."""
    bootstrap_engines()
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

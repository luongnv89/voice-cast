"""
GUI widget for managing TTS models.

Provides model browsing, download, and status management.
"""

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.styled_widgets import (
    SectionHeader,
    StatusLabel,
    StyledButton,
    StyledCard,
    StyledLabel,
    StyledProgressBar,
)
from gui.theme import (
    SPACING,
    TYPOGRAPHY,
    generate_scrollarea_style,
    get_theme_manager,
)
from models import DownloadProgress, ModelDownloader, ModelRegistry
from models.model_info import ModelInfo


class ModelDownloadThread(QThread):
    """Thread for downloading models without blocking the UI."""

    progress_updated = Signal(str, int, int, float)  # model_id, downloaded, total, speed
    download_complete = Signal(str)  # model_id
    download_error = Signal(str, str)  # model_id, error_message

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        self._downloader = ModelDownloader()

    def run(self):
        try:

            def progress_callback(progress: DownloadProgress):
                if self.isInterruptionRequested():
                    raise RuntimeError("Download cancelled")
                self.progress_updated.emit(
                    progress.model_id or self.model_id,
                    progress.downloaded_bytes,
                    progress.total_bytes,
                    progress.speed_bytes_per_sec,
                )

            self._downloader.download(self.model_id, progress_callback=progress_callback)
            if not self.isInterruptionRequested():
                self.download_complete.emit(self.model_id)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.download_error.emit(self.model_id, str(e))


class ModelCard(StyledCard):
    """Card widget displaying a single model's info and controls."""

    download_requested = Signal(str)  # model_id

    def __init__(self, model: ModelInfo, parent=None):
        super().__init__(parent)
        self.model = model
        self._setup_ui()
        # Apply styles after UI is set up
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.md, SPACING.md, SPACING.md, SPACING.md)
        layout.setSpacing(SPACING.md)

        # Model info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(SPACING.xs)

        # Title row
        title_layout = QHBoxLayout()
        title_layout.setSpacing(SPACING.sm)

        self.name_label = StyledLabel(self.model.name, role="primary")
        self._apply_name_style()
        title_layout.addWidget(self.name_label)

        self.status_label = StatusLabel()
        self._update_status_label()
        title_layout.addWidget(self.status_label)
        title_layout.addStretch()

        info_layout.addLayout(title_layout)

        # Description
        self.desc_label = StyledLabel(self.model.description, role="muted")
        self.desc_label.setWordWrap(True)
        info_layout.addWidget(self.desc_label)

        # Details row
        details = f"Engine: {self.model.engine} | Size: ~{self.model.size_mb} MB"
        self.details_label = StyledLabel(details, role="muted")
        self._apply_details_style()
        info_layout.addWidget(self.details_label)

        layout.addLayout(info_layout, stretch=1)

        # Action section
        action_layout = QVBoxLayout()
        action_layout.setSpacing(SPACING.xs)

        self.download_btn = StyledButton("Download", variant="primary")
        self.download_btn.setFixedWidth(110)
        self.download_btn.clicked.connect(self._on_download_clicked)
        self._update_download_button()
        action_layout.addWidget(self.download_btn)

        self.progress_bar = StyledProgressBar()
        self.progress_bar.setFixedWidth(110)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        self.speed_label = StyledLabel("", role="muted")
        self._apply_speed_style()
        self.speed_label.setVisible(False)
        action_layout.addWidget(self.speed_label)

        layout.addLayout(action_layout)

    def _apply_style(self):
        """Apply theme-aware styling."""
        super()._apply_style()
        self._apply_name_style()
        self._apply_details_style()
        self._apply_speed_style()
        self._update_status_label()
        self._update_download_button()

    def _apply_name_style(self):
        palette = get_theme_manager().palette
        self.name_label.setStyleSheet(f"""
            font-size: {TYPOGRAPHY.size_lg}px;
            font-weight: bold;
            color: {palette.text_primary};
        """)

    def _apply_details_style(self):
        palette = get_theme_manager().palette
        self.details_label.setStyleSheet(f"""
            font-size: {TYPOGRAPHY.size_sm}px;
            color: {palette.text_muted};
        """)

    def _apply_speed_style(self):
        palette = get_theme_manager().palette
        self.speed_label.setStyleSheet(f"""
            font-size: {TYPOGRAPHY.size_xs}px;
            color: {palette.text_muted};
        """)

    def _update_status_label(self):
        if self.model.is_installed:
            self.status_label.set_status("success", "Installed")
        else:
            self.status_label.set_status("warning", "Not installed")

    def _update_download_button(self):
        palette = get_theme_manager().palette
        if self.model.is_installed:
            self.download_btn.setText("Installed")
            self.download_btn.setEnabled(False)
            self.download_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {palette.bg_tertiary};
                    color: {palette.success};
                    border: 2px solid {palette.success};
                    padding: {SPACING.sm}px {SPACING.md}px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
            """)
        else:
            self.download_btn.setText("Download")
            self.download_btn.setEnabled(True)
            self.download_btn.set_variant("primary")

    def _on_download_clicked(self):
        self.download_requested.emit(self.model.id)

    def start_download(self):
        """Show download progress UI."""
        self.download_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.speed_label.setVisible(True)
        self.speed_label.setText("Starting...")

    def update_progress(self, downloaded: int, total: int, speed: float):
        """Update download progress."""
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)

        speed_mb = speed / (1024 * 1024)
        self.speed_label.setText(f"{speed_mb:.1f} MB/s")

    def download_finished(self, success: bool, error_message: str = ""):
        """Handle download completion."""
        self.progress_bar.setVisible(False)
        self.speed_label.setVisible(False)
        self.download_btn.setVisible(True)

        if success:
            self.model.is_installed = True
            self._update_status_label()
            self._update_download_button()
        else:
            self.download_btn.setEnabled(True)
            QMessageBox.critical(self, "Download Error", f"Failed to download model:\n\n{error_message}")

    def refresh_status(self, model: ModelInfo):
        """Refresh the model status."""
        self.model = model
        self._update_status_label()
        self._update_download_button()


class EngineSection(QWidget):
    """Section containing models for a specific engine."""

    download_requested = Signal(str)

    def __init__(self, engine_name: str, models: list[ModelInfo], parent=None):
        super().__init__(parent)
        self._engine_name = engine_name
        self._models = models
        self._cards: dict[str, ModelCard] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, SPACING.md)
        layout.setSpacing(SPACING.sm)

        # Engine header
        palette = get_theme_manager().palette
        header = StyledLabel(self._engine_name.upper(), role="accent")
        header.setStyleSheet(f"""
            font-size: {TYPOGRAPHY.size_sm}px;
            font-weight: bold;
            color: {palette.accent};
            letter-spacing: 1px;
            padding: {SPACING.xs}px 0;
            border-bottom: 1px solid {palette.border_secondary};
        """)
        layout.addWidget(header)

        # Model cards
        for model in self._models:
            card = ModelCard(model)
            card.download_requested.connect(self.download_requested.emit)
            self._cards[model.id] = card
            layout.addWidget(card)

    def get_card(self, model_id: str) -> ModelCard | None:
        """Get a model card by ID."""
        return self._cards.get(model_id)

    def refresh_model(self, model: ModelInfo):
        """Refresh a specific model's status."""
        card = self._cards.get(model.id)
        if card:
            card.refresh_status(model)


class ModelManagerWidget(QWidget):
    """Widget for managing TTS model downloads."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry = ModelRegistry()
        self._download_threads: dict[str, ModelDownloadThread] = {}
        self._engine_sections: dict[str, EngineSection] = {}
        self._shutting_down = False
        self._setup_ui()
        get_theme_manager().theme_changed.connect(self._apply_theme)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(SPACING.lg, SPACING.lg, SPACING.lg, SPACING.lg)
        main_layout.setSpacing(SPACING.md)

        # Header
        header = SectionHeader(
            "Model Manager",
            "Download TTS models before using them. Models are stored locally for offline use.",
        )
        main_layout.addWidget(header)

        # Refresh button
        button_layout = QHBoxLayout()
        button_layout.setSpacing(SPACING.sm)

        self.refresh_btn = StyledButton("Refresh Status", variant="secondary")
        self.refresh_btn.clicked.connect(self._refresh_models)
        button_layout.addWidget(self.refresh_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Scrollable model list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self._apply_scroll_style()

        self.scroll_widget = QWidget()
        self.models_layout = QVBoxLayout(self.scroll_widget)
        self.models_layout.setSpacing(SPACING.lg)
        self.models_layout.setContentsMargins(0, 0, SPACING.sm, 0)

        self.scroll.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll)

        # Load models
        self._load_models()

    def _apply_scroll_style(self):
        self.scroll.setStyleSheet(generate_scrollarea_style())

    def _apply_theme(self):
        """Apply theme updates."""
        self._apply_scroll_style()

    def _load_models(self):
        """Load and display all models."""
        # Clear existing sections
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._engine_sections.clear()

        # Group models by engine
        models = self._registry.list_models()
        engines: dict[str, list[ModelInfo]] = {}
        for model in models:
            if model.engine not in engines:
                engines[model.engine] = []
            engines[model.engine].append(model)

        # Create sections for each engine
        for engine_name, engine_models in engines.items():
            section = EngineSection(engine_name, engine_models)
            section.download_requested.connect(self._start_download)
            self._engine_sections[engine_name] = section
            self.models_layout.addWidget(section)

        self.models_layout.addStretch()

    def _refresh_models(self):
        """Refresh model status from registry."""
        for engine_name, section in self._engine_sections.items():
            for model in self._registry.list_models():
                if model.engine == engine_name:
                    section.refresh_model(model)

    def _get_card(self, model_id: str) -> ModelCard | None:
        """Find a model card by ID across all sections."""
        for section in self._engine_sections.values():
            card = section.get_card(model_id)
            if card:
                return card
        return None

    @Slot(str)
    def _start_download(self, model_id: str):
        """Start downloading a model."""
        if model_id in self._download_threads:
            return  # Already downloading

        card = self._get_card(model_id)
        if card:
            card.start_download()

        thread = ModelDownloadThread(model_id)
        thread.progress_updated.connect(self._on_progress_updated)
        thread.download_complete.connect(self._on_download_complete)
        thread.download_error.connect(self._on_download_error)
        thread.finished.connect(lambda model_id=model_id: self._download_threads.pop(model_id, None))
        self._download_threads[model_id] = thread
        thread.start()

    @Slot(str, int, int, float)
    def _on_progress_updated(self, model_id: str, downloaded: int, total: int, speed: float):
        """Handle progress update."""
        card = self._get_card(model_id)
        if card:
            card.update_progress(downloaded, total, speed)

    @Slot(str)
    def _on_download_complete(self, model_id: str):
        """Handle download completion."""
        card = self._get_card(model_id)
        if card:
            card.download_finished(success=True)

        self._download_threads.pop(model_id, None)

    @Slot(str, str)
    def _on_download_error(self, model_id: str, error_message: str):
        """Handle download error."""
        if not self._shutting_down:
            card = self._get_card(model_id)
            if card:
                card.download_finished(success=False, error_message=error_message)

        self._download_threads.pop(model_id, None)

    def shutdown_downloads(self):
        """Stop active download threads before the widget is destroyed."""
        self._shutting_down = True
        for thread in list(self._download_threads.values()):
            if thread.isRunning():
                thread.requestInterruption()
        for thread in list(self._download_threads.values()):
            if thread.isRunning():
                thread.wait()
        self._download_threads.clear()

    def closeEvent(self, event: QCloseEvent):
        """Ensure active download threads finish before closing."""
        self.shutdown_downloads()
        super().closeEvent(event)

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        return self._registry.is_installed(model_id)

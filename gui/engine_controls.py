"""
Engine-specific control widgets for VoiceCast.

Provides parameter controls for different TTS engines.
"""

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

# Source-of-truth voice data from MlxAudioEngine
from engines.mlx_audio_engine import KOKORO_LANG_CODES, KOKORO_VOICES
from gui.styled_widgets import (
    StyledButton,
    StyledComboBox,
    StyledGroupBox,
    StyledLabel,
    StyledSlider,
)
from gui.theme import SPACING
from tts_factory import TTSFactory


class EngineControlsBase(QWidget):
    """Base class for engine-specific control widgets."""

    parameters_changed = Signal(dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        info_label = StyledLabel(
            "No engine-specific controls available.",
            role="muted",
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

    def get_parameters(self) -> dict[str, Any]:
        """Return current parameter values."""
        return {}


class CoquiControls(EngineControlsBase):
    """Control widget for Coqui XTTS v2 engine."""

    LANGUAGES = [
        ("English", "en"),
        ("Spanish", "es"),
        ("French", "fr"),
        ("German", "de"),
        ("Italian", "it"),
        ("Portuguese", "pt"),
        ("Polish", "pl"),
        ("Turkish", "tr"),
        ("Russian", "ru"),
        ("Dutch", "nl"),
        ("Czech", "cs"),
        ("Arabic", "ar"),
        ("Chinese", "zh"),
        ("Japanese", "ja"),
        ("Hungarian", "hu"),
        ("Korean", "ko"),
    ]

    def __init__(self):
        super().__init__()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        # Language selector
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(SPACING.sm)

        lang_label = StyledLabel("Language:", role="secondary")
        lang_layout.addWidget(lang_label)

        self.lang_combo = StyledComboBox()
        for display_name, code in self.LANGUAGES:
            self.lang_combo.addItem(display_name, code)
        self.lang_combo.currentIndexChanged.connect(self._on_param_changed)
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addStretch()

        layout.addLayout(lang_layout)

        # Temperature slider
        temp_group = StyledGroupBox("Temperature")
        temp_layout = QVBoxLayout()
        temp_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.temp_slider = StyledSlider(
            min_val=10,
            max_val=100,
            initial=70,
            value_format="{:.2f}",
            value_scale=0.01,
        )
        self.temp_slider.slider.valueChanged.connect(self._on_param_changed)
        temp_layout.addWidget(self.temp_slider)

        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)

    def _on_param_changed(self):
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict[str, Any]:
        return {
            "language": self.lang_combo.currentData(),
            "temperature": self.temp_slider.scaled_value(),
        }


class ChatterboxControls(EngineControlsBase):
    """Control widget for Chatterbox engine."""

    PARALINGUISTIC_TAGS = ["laugh", "chuckle", "cough", "sigh", "gasp", "yawn"]

    def __init__(self, variant: str = "turbo"):
        self.variant = variant
        # ``variant`` belongs to this control, not QWidget. Do not forward it
        # through EngineControlsBase, whose kwargs are Qt constructor options.
        super().__init__()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        # CFG Weight slider
        cfg_group = StyledGroupBox("CFG Weight (text adherence)")
        cfg_layout = QVBoxLayout()
        cfg_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.cfg_slider = StyledSlider(
            min_val=0,
            max_val=100,
            initial=50,
            min_label="Less",
            max_label="More",
            value_format="{:.2f}",
            value_scale=0.01,
        )
        self.cfg_slider.slider.valueChanged.connect(self._on_param_changed)
        cfg_layout.addWidget(self.cfg_slider)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        # Exaggeration slider
        exag_group = StyledGroupBox("Exaggeration (expressiveness)")
        exag_layout = QVBoxLayout()
        exag_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.exag_slider = StyledSlider(
            min_val=0,
            max_val=150,
            initial=50,
            min_label="Subtle",
            max_label="Dramatic",
            value_format="{:.2f}",
            value_scale=0.01,
        )
        self.exag_slider.slider.valueChanged.connect(self._on_param_changed)
        exag_layout.addWidget(self.exag_slider)

        exag_group.setLayout(exag_layout)
        layout.addWidget(exag_group)

        # Paralinguistic tags help (Turbo only)
        if self.variant == "turbo":
            tag_layout = QHBoxLayout()
            tag_layout.setSpacing(SPACING.sm)

            self.tag_button = StyledButton("Paralinguistic Tags Help", variant="ghost")
            self.tag_button.clicked.connect(self._show_tags_help)
            tag_layout.addWidget(self.tag_button)
            tag_layout.addStretch()

            layout.addLayout(tag_layout)

    def _on_param_changed(self):
        self.parameters_changed.emit(self.get_parameters())

    def _show_tags_help(self):
        tags_text = ", ".join([f"[{tag}]" for tag in self.PARALINGUISTIC_TAGS])
        QMessageBox.information(
            self,
            "Paralinguistic Tags",
            f"Chatterbox Turbo supports these expressive tags:\n\n"
            f"{tags_text}\n\n"
            f"Example usage:\n"
            f"'That's hilarious [laugh]!'\n"
            f"'*sighs* [sigh] I can't believe this...'\n\n"
            f"Place tags where you want the sound to occur.",
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "cfg_weight": self.cfg_slider.scaled_value(),
            "exaggeration": self.exag_slider.scaled_value(),
        }


class MlxKokoroControls(EngineControlsBase):
    """Control widget for MLX Kokoro preset voices."""

    # Voice data sourced from MlxAudioEngine (single source of truth)
    VOICE_GROUPS = KOKORO_VOICES
    LANG_CODES = KOKORO_LANG_CODES

    def __init__(self, variant: str = "kokoro"):
        super().__init__()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        # Language selector
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(SPACING.sm)

        lang_label = StyledLabel("Language:", role="secondary")
        lang_layout.addWidget(lang_label)

        self.lang_combo = StyledComboBox()
        for lang in self.VOICE_GROUPS:
            self.lang_combo.addItem(lang, self.LANG_CODES[lang])
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addStretch()

        layout.addLayout(lang_layout)

        # Category selector (Male/Female)
        cat_layout = QHBoxLayout()
        cat_layout.setSpacing(SPACING.sm)

        cat_label = StyledLabel("Category:", role="secondary")
        cat_layout.addWidget(cat_label)

        self.category_combo = StyledComboBox()
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.category_combo)
        cat_layout.addStretch()

        layout.addLayout(cat_layout)

        # Voice selector
        voice_layout = QHBoxLayout()
        voice_layout.setSpacing(SPACING.sm)

        voice_label = StyledLabel("Voice:", role="secondary")
        voice_layout.addWidget(voice_label)

        self.voice_combo = StyledComboBox()
        self.voice_combo.currentIndexChanged.connect(self._on_param_changed)
        voice_layout.addWidget(self.voice_combo)
        voice_layout.addStretch()

        layout.addLayout(voice_layout)

        # Speed slider
        speed_group = StyledGroupBox("Speed")
        speed_layout = QVBoxLayout()
        speed_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.speed_slider = StyledSlider(
            min_val=50,
            max_val=200,
            initial=100,
            min_label="0.5x",
            max_label="2.0x",
            value_format="{:.2f}x",
            value_scale=0.01,
        )
        self.speed_slider.slider.valueChanged.connect(self._on_param_changed)
        speed_layout.addWidget(self.speed_slider)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Initialize dropdowns
        self._populate_categories()

    def _populate_categories(self):
        """Populate category dropdown based on selected language."""
        self.category_combo.blockSignals(True)
        self.category_combo.clear()

        current_lang = self.lang_combo.currentText()
        if current_lang in self.VOICE_GROUPS:
            for category in self.VOICE_GROUPS[current_lang]:
                self.category_combo.addItem(category)

        self.category_combo.blockSignals(False)
        self._populate_voices()

    def _populate_voices(self):
        """Populate voice dropdown based on selected language and category."""
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()

        current_lang = self.lang_combo.currentText()
        current_cat = self.category_combo.currentText()

        if current_lang in self.VOICE_GROUPS and current_cat in self.VOICE_GROUPS[current_lang]:
            for voice_id, voice_name in self.VOICE_GROUPS[current_lang][current_cat]:
                self.voice_combo.addItem(voice_name, voice_id)

        self.voice_combo.blockSignals(False)
        self._on_param_changed()

    def _on_language_changed(self):
        """Handle language selection change."""
        self._populate_categories()

    def _on_category_changed(self):
        """Handle category selection change."""
        self._populate_voices()

    def _on_param_changed(self):
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict[str, Any]:
        return {
            "voice": self.voice_combo.currentData() or "af_heart",
            "speed": self.speed_slider.scaled_value(),
            "lang_code": self.lang_combo.currentData() or "a",
        }


class MlxCsmControls(EngineControlsBase):
    """Control widget for MLX CSM voice cloning."""

    def __init__(self, variant: str = "csm"):
        super().__init__()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        # Speed slider
        speed_group = StyledGroupBox("Speed")
        speed_layout = QVBoxLayout()
        speed_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.speed_slider = StyledSlider(
            min_val=50,
            max_val=200,
            initial=100,
            min_label="0.5x",
            max_label="2.0x",
            value_format="{:.2f}x",
            value_scale=0.01,
        )
        self.speed_slider.slider.valueChanged.connect(self._on_param_changed)
        speed_layout.addWidget(self.speed_slider)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Info label
        info_label = StyledLabel(
            "CSM uses the selected voice reference file for voice cloning.",
            role="muted",
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    def _on_param_changed(self):
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict[str, Any]:
        return {
            "speed": self.speed_slider.scaled_value(),
        }


class Audio8Controls(EngineControlsBase):
    """Control widget for Audio8 TTS engine."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.md)

        # Speed slider
        speed_group = StyledGroupBox("Speed")
        speed_layout = QVBoxLayout()
        speed_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)

        self.speed_slider = StyledSlider(
            min_val=50,
            max_val=200,
            initial=100,
            min_label="0.5x",
            max_label="2.0x",
            value_format="{:.2f}x",
            value_scale=0.01,
        )
        self.speed_slider.slider.valueChanged.connect(self._on_param_changed)
        speed_layout.addWidget(self.speed_slider)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Info label
        info_label = StyledLabel(
            "Audio8 uses the selected voice reference file for voice cloning.",
            role="muted",
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    def _on_param_changed(self):
        self.parameters_changed.emit(self.get_parameters())

    def get_parameters(self) -> dict[str, Any]:
        return {
            "speed": self.speed_slider.scaled_value(),
        }


class EngineControlsFactory:
    """Factory for creating engine-specific control widgets."""

    @staticmethod
    def create(engine_name: str) -> EngineControlsBase:
        """
        Create control widget for the specified engine.

        Looks up the registered EngineDescriptor from TTSFactory
        and instantiates its associated controls_class.

        Args:
            engine_name: Engine identifier (e.g., "coqui", "chatterbox-turbo")

        Returns:
            EngineControlsBase widget instance
        """
        controls_class = TTSFactory.get_controls_class(engine_name)
        if controls_class is not None:
            kwargs = TTSFactory.get_default_kwargs(engine_name)
            return controls_class(**kwargs)
        return EngineControlsBase()

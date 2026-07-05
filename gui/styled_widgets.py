"""
Styled widget components for VoiceCast UI.

Provides pre-styled widgets that follow the design system.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import (
    SPACING,
    TYPOGRAPHY,
    generate_button_style,
    generate_card_style,
    generate_combobox_style,
    generate_groupbox_style,
    generate_label_style,
    generate_progressbar_style,
    generate_slider_style,
    generate_tabwidget_style,
    generate_textedit_style,
    get_theme_manager,
)


class StyledButton(QPushButton):
    """
    Styled button with variant support.

    Variants:
        - primary: Accent-colored border with accent text
        - secondary: Subtle border with standard text
        - ghost: Transparent background, visible on hover
    """

    def __init__(
        self,
        text: str = "",
        variant: str = "primary",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._variant = variant
        self._apply_style()

        # Connect to theme changes
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_button_style(self._variant))

    def set_variant(self, variant: str):
        """Change button variant."""
        self._variant = variant
        self._apply_style()


class StyledGroupBox(QGroupBox):
    """Styled group box with depth effect."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_groupbox_style())


class StyledSlider(QWidget):
    """
    Styled horizontal slider with label.

    Shows current value and supports min/max labels.
    """

    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 100,
        initial: int = 50,
        min_label: str = "",
        max_label: str = "",
        value_format: str = "{:.2f}",
        value_scale: float = 0.01,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._value_format = value_format
        self._value_scale = value_scale
        self._has_min_label = bool(min_label)
        self._has_max_label = bool(max_label)

        # Ensure widget has minimum size
        self.setMinimumHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(SPACING.sm)

        # Min label
        if min_label:
            self._min_label = QLabel(min_label)
            layout.addWidget(self._min_label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(min_val, max_val)
        self._slider.setValue(initial)
        self._slider.setMinimumHeight(24)
        self._slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._slider, stretch=1)

        # Max label
        if max_label:
            self._max_label = QLabel(max_label)
            layout.addWidget(self._max_label)

        # Value label
        self._value_label = QLabel()
        self._value_label.setMinimumWidth(50)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value_label)

        # Apply styles after all widgets are created
        self._apply_slider_style()
        self._update_value_label()

        get_theme_manager().theme_changed.connect(self._apply_slider_style)

    def _apply_slider_style(self):
        palette = get_theme_manager().palette
        # Make sure the container widget is transparent
        self.setStyleSheet(f"background-color: transparent;")
        self._slider.setStyleSheet(generate_slider_style())
        self._value_label.setStyleSheet(
            f"QLabel {{ color: {palette.accent}; font-weight: bold; font-size: {TYPOGRAPHY.size_md}px; background-color: transparent; }}"
        )
        if self._has_min_label:
            self._min_label.setStyleSheet(f"QLabel {{ color: {palette.text_muted}; font-size: {TYPOGRAPHY.size_sm}px; background-color: transparent; }}")
        if self._has_max_label:
            self._max_label.setStyleSheet(f"QLabel {{ color: {palette.text_muted}; font-size: {TYPOGRAPHY.size_sm}px; background-color: transparent; }}")

    def _on_value_changed(self, value: int):
        self._update_value_label()

    def _update_value_label(self):
        scaled = self._slider.value() * self._value_scale
        self._value_label.setText(self._value_format.format(scaled))

    def value(self) -> int:
        """Get raw slider value."""
        return self._slider.value()

    def scaled_value(self) -> float:
        """Get scaled slider value."""
        return self._slider.value() * self._value_scale

    def setValue(self, value: int):
        """Set slider value."""
        self._slider.setValue(value)

    @property
    def slider(self) -> QSlider:
        """Access underlying slider for signal connections."""
        return self._slider


class StyledComboBox(QComboBox):
    """Styled combo box."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_combobox_style())


class StyledTextEdit(QTextEdit):
    """Styled text edit with proper focus states."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_textedit_style())


class StyledProgressBar(QProgressBar):
    """Styled progress bar with accent color."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_progressbar_style())


class StyledTabWidget(QTabWidget):
    """Styled tab widget with accent indicators."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_tabwidget_style())


class StyledCard(QFrame):
    """
    Card component with depth effect.

    Use for model cards and similar elevated content.
    Subclasses should call _apply_style() after their own setup is complete.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "card")
        # Don't call _apply_style here - let subclasses call it after setup
        # For direct StyledCard usage, apply style now
        if type(self) is StyledCard:
            self._apply_style()
            get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        palette = get_theme_manager().palette
        self.setStyleSheet(f"""
            StyledCard {{
                background-color: {palette.bg_secondary};
                border: 1px solid {palette.border_primary};
                border-radius: 8px;
            }}
            StyledCard:hover {{
                border-color: {palette.accent};
            }}
        """)


class StyledLabel(QLabel):
    """
    Styled label with role support.

    Roles:
        - primary: Main text color
        - secondary: Dimmer text
        - muted: Very dim text
        - accent: Bright green highlight
        - danger/warning/info/success: Status colors (text only)
    """

    def __init__(
        self,
        text: str = "",
        role: str = "primary",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._role = role
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        self.setStyleSheet(generate_label_style(self._role))

    def set_role(self, role: str):
        """Change label role."""
        self._role = role
        self._apply_style()


class StatusLabel(QLabel):
    """
    Status label for displaying success/error/warning/info states.

    Colors are applied to text only, never as background.
    """

    def __init__(
        self,
        text: str = "",
        status: str = "info",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._status = status
        self._apply_style()
        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_style(self):
        palette = get_theme_manager().palette
        status_colors = {
            "success": palette.success,
            "danger": palette.danger,
            "warning": palette.warning,
            "info": palette.info,
        }
        color = status_colors.get(self._status, palette.text_primary)
        self.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                font-size: {TYPOGRAPHY.size_md}px;
                background-color: transparent;
            }}
        """)

    def set_status(self, status: str, text: Optional[str] = None):
        """Change status and optionally text."""
        self._status = status
        if text is not None:
            self.setText(text)
        self._apply_style()


class IconLabel(QWidget):
    """Label with icon indicator (colored dot/line)."""

    def __init__(
        self,
        text: str = "",
        color: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.sm)

        # Color indicator
        self._indicator = QFrame()
        self._indicator.setFixedSize(8, 8)
        self._apply_indicator_style()
        layout.addWidget(self._indicator)

        # Text
        self._label = QLabel(text)
        layout.addWidget(self._label)
        layout.addStretch()

        get_theme_manager().theme_changed.connect(self._apply_style)

    def _apply_indicator_style(self):
        palette = get_theme_manager().palette
        color = self._color or palette.accent
        self._indicator.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)

    def _apply_style(self):
        self._apply_indicator_style()
        palette = get_theme_manager().palette
        self._label.setStyleSheet(f"color: {palette.text_primary};")

    def set_color(self, color: str):
        """Set indicator color."""
        self._color = color
        self._apply_indicator_style()

    def setText(self, text: str):
        """Set label text."""
        self._label.setText(text)


class SectionHeader(QWidget):
    """Section header with title and optional subtitle."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, SPACING.sm)
        layout.setSpacing(SPACING.xs)

        # Title
        self._title = QLabel(title)
        self._apply_title_style()
        layout.addWidget(self._title)

        # Subtitle
        if subtitle:
            self._subtitle = QLabel(subtitle)
            self._apply_subtitle_style()
            self._subtitle.setWordWrap(True)
            layout.addWidget(self._subtitle)

        get_theme_manager().theme_changed.connect(self._apply_styles)

    def _apply_title_style(self):
        palette = get_theme_manager().palette
        self._title.setStyleSheet(f"""
            font-size: {TYPOGRAPHY.size_xl}px;
            font-weight: bold;
            color: {palette.text_primary};
        """)

    def _apply_subtitle_style(self):
        if hasattr(self, "_subtitle"):
            palette = get_theme_manager().palette
            self._subtitle.setStyleSheet(f"""
                font-size: {TYPOGRAPHY.size_md}px;
                color: {palette.text_muted};
            """)

    def _apply_styles(self):
        self._apply_title_style()
        self._apply_subtitle_style()

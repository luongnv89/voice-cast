"""
Centralized theme system for VoiceCast UI.

Provides Light and Dark mode support with a strict 4-color palette:
- Black, White, Gray (range), and Bright Green (highlights only)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


class ThemeMode(Enum):
    """Available theme modes."""

    LIGHT = auto()
    DARK = auto()
    SYSTEM = auto()


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for a theme."""

    # Backgrounds (never use accent as background)
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_elevated: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str

    # Borders
    border_primary: str
    border_secondary: str
    border_focus: str

    # Accent (Bright Green - highlights only, NEVER as background)
    accent: str
    accent_hover: str
    accent_muted: str

    # Status colors (text only, never as backgrounds)
    danger: str
    warning: str
    info: str
    success: str

    # Interactive states
    hover_overlay: str
    pressed_overlay: str
    disabled_bg: str
    disabled_text: str


# Light Mode Palette
LIGHT_PALETTE = ColorPalette(
    # Backgrounds
    bg_primary="#FFFFFF",
    bg_secondary="#F5F5F5",
    bg_tertiary="#EBEBEB",
    bg_elevated="#FFFFFF",
    # Text
    text_primary="#000000",
    text_secondary="#4A4A4A",
    text_muted="#666666",
    # Borders
    border_primary="#D0D0D0",
    border_secondary="#E0E0E0",
    border_focus="#000000",
    # Accent (Dark Green)
    accent="#2E7D32",
    accent_hover="#1B5E20",
    accent_muted="#81C784",
    # Status (text only)
    danger="#DC3545",
    warning="#8A6D00",
    info="#0D6EFD",
    success="#1E7E34",
    # Interactive
    hover_overlay="rgba(0, 0, 0, 0.04)",
    pressed_overlay="rgba(0, 0, 0, 0.08)",
    disabled_bg="#F0F0F0",
    disabled_text="#A0A0A0",
)

# Dark Mode Palette
DARK_PALETTE = ColorPalette(
    # Backgrounds
    bg_primary="#0A0A0A",
    bg_secondary="#141414",
    bg_tertiary="#1E1E1E",
    bg_elevated="#1E1E1E",
    # Text
    text_primary="#FFFFFF",
    text_secondary="#C0C0C0",
    text_muted="#808080",
    # Borders
    border_primary="#404040",
    border_secondary="#303030",
    border_focus="#FFFFFF",
    # Accent (Dark Green - slightly brighter for dark mode visibility)
    accent="#4CAF50",
    accent_hover="#66BB6A",
    accent_muted="#2E7D32",
    # Status (text only)
    danger="#FF6B6B",
    warning="#FFD93D",
    info="#6EA8FE",
    success="#75B798",
    # Interactive
    hover_overlay="rgba(255, 255, 255, 0.04)",
    pressed_overlay="rgba(255, 255, 255, 0.08)",
    disabled_bg="#1A1A1A",
    disabled_text="#606060",
)


@dataclass(frozen=True)
class Spacing:
    """Spacing constants based on 8px grid system."""

    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32
    xxl: int = 48


@dataclass(frozen=True)
class Typography:
    """Typography scale."""

    # Font sizes
    size_xs: int = 10
    size_sm: int = 12
    size_md: int = 14
    size_lg: int = 16
    size_xl: int = 20
    size_xxl: int = 24
    size_h1: int = 32

    # Font weights (for reference in stylesheets)
    weight_normal: str = "normal"
    weight_medium: str = "500"
    weight_bold: str = "bold"


@dataclass(frozen=True)
class Shadows:
    """Shadow definitions for visual depth."""

    # Light mode shadows
    light_sm: str = "0 1px 2px rgba(0, 0, 0, 0.05)"
    light_md: str = "0 2px 4px rgba(0, 0, 0, 0.08)"
    light_lg: str = "0 4px 8px rgba(0, 0, 0, 0.1)"

    # Dark mode shadows (more subtle)
    dark_sm: str = "0 1px 2px rgba(0, 0, 0, 0.3)"
    dark_md: str = "0 2px 4px rgba(0, 0, 0, 0.4)"
    dark_lg: str = "0 4px 8px rgba(0, 0, 0, 0.5)"


# Global instances
SPACING = Spacing()
TYPOGRAPHY = Typography()
SHADOWS = Shadows()


class ThemeManager(QObject):
    """
    Manages application theme state and provides style generation.

    Signals:
        theme_changed: Emitted when the theme mode changes
    """

    theme_changed = Signal(ThemeMode)

    _instance: Optional["ThemeManager"] = None

    def __new__(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        self._settings = QSettings("VoiceCast", "VoiceCast")
        self._current_mode = self._load_theme_preference()
        self._palette = self._get_palette_for_mode(self._current_mode)

    def _load_theme_preference(self) -> ThemeMode:
        """Load theme preference from settings."""
        saved = self._settings.value("theme/mode", "system")
        mode_map = {
            "light": ThemeMode.LIGHT,
            "dark": ThemeMode.DARK,
            "system": ThemeMode.SYSTEM,
        }
        return mode_map.get(saved, ThemeMode.SYSTEM)

    def _save_theme_preference(self, mode: ThemeMode):
        """Save theme preference to settings."""
        mode_map = {
            ThemeMode.LIGHT: "light",
            ThemeMode.DARK: "dark",
            ThemeMode.SYSTEM: "system",
        }
        self._settings.setValue("theme/mode", mode_map[mode])

    def _detect_system_theme(self) -> ThemeMode:
        """Detect system theme preference."""
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.ColorRole.Window)
            # If background is dark, system is in dark mode
            if bg_color.lightness() < 128:
                return ThemeMode.DARK
        return ThemeMode.LIGHT

    def _get_palette_for_mode(self, mode: ThemeMode) -> ColorPalette:
        """Get the color palette for a given mode."""
        if mode == ThemeMode.SYSTEM:
            effective_mode = self._detect_system_theme()
        else:
            effective_mode = mode

        return DARK_PALETTE if effective_mode == ThemeMode.DARK else LIGHT_PALETTE

    @property
    def current_mode(self) -> ThemeMode:
        """Get current theme mode."""
        return self._current_mode

    @property
    def palette(self) -> ColorPalette:
        """Get current color palette."""
        return self._palette

    @property
    def is_dark(self) -> bool:
        """Check if current effective theme is dark."""
        if self._current_mode == ThemeMode.SYSTEM:
            return self._detect_system_theme() == ThemeMode.DARK
        return self._current_mode == ThemeMode.DARK

    def set_theme(self, mode: ThemeMode):
        """Set the theme mode."""
        if mode != self._current_mode:
            self._current_mode = mode
            self._palette = self._get_palette_for_mode(mode)
            self._save_theme_preference(mode)
            self.theme_changed.emit(mode)

    def get_shadow(self, size: str = "md") -> str:
        """Get appropriate shadow for current theme."""
        if self.is_dark:
            shadows = {"sm": SHADOWS.dark_sm, "md": SHADOWS.dark_md, "lg": SHADOWS.dark_lg}
        else:
            shadows = {"sm": SHADOWS.light_sm, "md": SHADOWS.light_md, "lg": SHADOWS.light_lg}
        return shadows.get(size, shadows["md"])


def get_theme_manager() -> ThemeManager:
    """Get the global ThemeManager instance."""
    return ThemeManager()


def generate_button_style(
    variant: str = "primary",
    palette: ColorPalette | None = None,
) -> str:
    """
    Generate button stylesheet.

    Args:
        variant: 'primary', 'secondary', or 'ghost'
        palette: Color palette to use (defaults to current theme)
    """
    if palette is None:
        palette = get_theme_manager().palette

    base_style = f"""
        QPushButton {{
            font-size: {TYPOGRAPHY.size_md}px;
            font-weight: {TYPOGRAPHY.weight_medium};
            padding: {SPACING.sm}px {SPACING.md}px;
            border-radius: 4px;
            min-height: 32px;
            min-width: 80px;
        }}
        QPushButton:focus {{
            outline: none;
            border: 2px solid {palette.border_focus};
        }}
    """

    if variant == "primary":
        return (
            base_style
            + f"""
            QPushButton {{
                background-color: {palette.bg_tertiary};
                color: {palette.accent};
                border: 2px solid {palette.accent};
            }}
            QPushButton:hover {{
                background-color: {palette.bg_secondary};
                border-color: {palette.accent_hover};
            }}
            QPushButton:pressed {{
                background-color: {palette.bg_tertiary};
            }}
            QPushButton:disabled {{
                background-color: {palette.disabled_bg};
                color: {palette.disabled_text};
                border-color: {palette.border_secondary};
            }}
        """
        )
    elif variant == "secondary":
        return (
            base_style
            + f"""
            QPushButton {{
                background-color: {palette.bg_secondary};
                color: {palette.text_primary};
                border: 1px solid {palette.border_primary};
            }}
            QPushButton:hover {{
                background-color: {palette.bg_tertiary};
                border-color: {palette.accent};
            }}
            QPushButton:pressed {{
                background-color: {palette.bg_tertiary};
            }}
            QPushButton:disabled {{
                background-color: {palette.disabled_bg};
                color: {palette.disabled_text};
                border-color: {palette.border_secondary};
            }}
        """
        )
    else:  # ghost
        return (
            base_style
            + f"""
            QPushButton {{
                background-color: transparent;
                color: {palette.text_primary};
                border: 1px solid transparent;
            }}
            QPushButton:hover {{
                background-color: {palette.bg_secondary};
                border-color: {palette.border_secondary};
            }}
            QPushButton:pressed {{
                background-color: {palette.bg_tertiary};
            }}
            QPushButton:disabled {{
                color: {palette.disabled_text};
            }}
        """
        )


def generate_groupbox_style(palette: ColorPalette | None = None) -> str:
    """Generate group box stylesheet with depth effect."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QGroupBox {{
            font-size: {TYPOGRAPHY.size_md}px;
            font-weight: {TYPOGRAPHY.weight_bold};
            color: {palette.text_primary};
            background-color: {palette.bg_secondary};
            border: 1px solid {palette.border_primary};
            border-radius: 6px;
            margin-top: 16px;
            padding-top: 24px;
            padding-left: {SPACING.md}px;
            padding-right: {SPACING.md}px;
            padding-bottom: {SPACING.md}px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: {SPACING.md}px;
            top: 4px;
            padding: 0 {SPACING.sm}px;
            background-color: {palette.bg_secondary};
            color: {palette.accent};
        }}
    """


def generate_slider_style(palette: ColorPalette | None = None) -> str:
    """Generate slider stylesheet with accent track."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QSlider {{
            min-height: 24px;
        }}
        QSlider::groove:horizontal {{
            height: 8px;
            background-color: {palette.border_primary};
            border: 1px solid {palette.border_secondary};
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            width: 18px;
            height: 18px;
            margin: -6px 0;
            background-color: {palette.accent};
            border: 2px solid {palette.accent};
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            background-color: {palette.accent_hover};
            border-color: {palette.accent_hover};
        }}
        QSlider::handle:horizontal:pressed {{
            background-color: {palette.accent_muted};
        }}
        QSlider::sub-page:horizontal {{
            background-color: {palette.accent};
            border-radius: 4px;
        }}
        QSlider::add-page:horizontal {{
            background-color: {palette.border_primary};
            border-radius: 4px;
        }}
    """


def generate_combobox_style(palette: ColorPalette | None = None) -> str:
    """Generate combobox stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QComboBox {{
            background-color: {palette.bg_secondary};
            color: {palette.text_primary};
            border: 1px solid {palette.border_primary};
            border-radius: 4px;
            padding: {SPACING.sm}px {SPACING.md}px;
            min-height: 32px;
            font-size: {TYPOGRAPHY.size_md}px;
        }}
        QComboBox:hover {{
            border-color: {palette.accent};
        }}
        QComboBox:focus {{
            border: 2px solid {palette.accent};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {palette.text_secondary};
            margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {palette.bg_elevated};
            color: {palette.text_primary};
            border: 1px solid {palette.border_primary};
            selection-background-color: {palette.bg_tertiary};
            selection-color: {palette.accent};
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            padding: {SPACING.sm}px;
            min-height: 28px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {palette.bg_tertiary};
        }}
    """


def generate_textedit_style(palette: ColorPalette | None = None) -> str:
    """Generate text edit stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QTextEdit {{
            background-color: {palette.bg_secondary};
            color: {palette.text_primary};
            border: 1px solid {palette.border_primary};
            border-radius: 4px;
            padding: {SPACING.sm}px;
            font-size: {TYPOGRAPHY.size_md}px;
            selection-background-color: {palette.accent_muted};
            selection-color: {palette.text_primary};
        }}
        QTextEdit:focus {{
            border: 2px solid {palette.accent};
        }}
    """


def generate_progressbar_style(palette: ColorPalette | None = None) -> str:
    """Generate progress bar stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QProgressBar {{
            background-color: {palette.bg_tertiary};
            border: 1px solid {palette.border_secondary};
            border-radius: 4px;
            text-align: center;
            color: {palette.text_primary};
            font-size: {TYPOGRAPHY.size_sm}px;
            min-height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {palette.accent};
            border-radius: 3px;
        }}
    """


def generate_tabwidget_style(palette: ColorPalette | None = None) -> str:
    """Generate tab widget stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QTabWidget::pane {{
            background-color: {palette.bg_primary};
            border: 1px solid {palette.border_primary};
            border-top: none;
            border-radius: 0 0 6px 6px;
        }}
        QTabBar::tab {{
            background-color: {palette.bg_secondary};
            color: {palette.text_secondary};
            border: 1px solid {palette.border_primary};
            border-bottom: none;
            padding: {SPACING.sm}px {SPACING.lg}px;
            margin-right: 2px;
            border-radius: 4px 4px 0 0;
            font-size: {TYPOGRAPHY.size_md}px;
            min-width: 100px;
        }}
        QTabBar::tab:selected {{
            background-color: {palette.bg_primary};
            color: {palette.accent};
            border-bottom: 2px solid {palette.accent};
            font-weight: {TYPOGRAPHY.weight_bold};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {palette.bg_tertiary};
            color: {palette.text_primary};
        }}
        QTabBar::tab:focus {{
            outline: 2px solid {palette.accent};
            outline-offset: -2px;
        }}
    """


def generate_scrollarea_style(palette: ColorPalette | None = None) -> str:
    """Generate scroll area stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QScrollArea {{
            background-color: {palette.bg_primary};
            border: none;
        }}
        QScrollBar:vertical {{
            background-color: {palette.bg_secondary};
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {palette.border_primary};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {palette.text_muted};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background-color: {palette.bg_secondary};
            height: 12px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {palette.border_primary};
            border-radius: 5px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {palette.text_muted};
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
    """


def generate_card_style(palette: ColorPalette | None = None) -> str:
    """Generate card component stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QFrame[class="card"] {{
            background-color: {palette.bg_secondary};
            border: 1px solid {palette.border_primary};
            border-radius: 8px;
            padding: {SPACING.md}px;
        }}
        QFrame[class="card"]:hover {{
            border-color: {palette.accent};
        }}
    """


def generate_label_style(
    role: str = "primary",
    palette: ColorPalette | None = None,
) -> str:
    """
    Generate label stylesheet.

    Args:
        role: 'primary', 'secondary', 'muted', 'danger', 'warning', 'info', 'success', 'accent'
        palette: Color palette to use
    """
    if palette is None:
        palette = get_theme_manager().palette

    color_map = {
        "primary": palette.text_primary,
        "secondary": palette.text_secondary,
        "muted": palette.text_muted,
        "danger": palette.danger,
        "warning": palette.warning,
        "info": palette.info,
        "success": palette.success,
        "accent": palette.accent,
    }
    color = color_map.get(role, palette.text_primary)

    return f"""
        QLabel {{
            color: {color};
            font-size: {TYPOGRAPHY.size_md}px;
        }}
    """


def generate_main_window_style(palette: ColorPalette | None = None) -> str:
    """Generate main window stylesheet."""
    if palette is None:
        palette = get_theme_manager().palette

    return f"""
        QMainWindow {{
            background-color: {palette.bg_primary};
        }}
        QWidget {{
            background-color: {palette.bg_primary};
            color: {palette.text_primary};
            font-size: {TYPOGRAPHY.size_md}px;
        }}
        QLabel {{
            background-color: transparent;
            color: {palette.text_primary};
        }}
        QMenuBar {{
            background-color: {palette.bg_secondary};
            color: {palette.text_primary};
            border-bottom: 1px solid {palette.border_primary};
            padding: 2px;
        }}
        QMenuBar::item {{
            padding: {SPACING.sm}px {SPACING.md}px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{
            background-color: {palette.bg_tertiary};
        }}
        QMenu {{
            background-color: {palette.bg_elevated};
            color: {palette.text_primary};
            border: 1px solid {palette.border_primary};
            border-radius: 6px;
            padding: {SPACING.xs}px;
        }}
        QMenu::item {{
            padding: {SPACING.sm}px {SPACING.lg}px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background-color: {palette.bg_tertiary};
            color: {palette.accent};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {palette.border_secondary};
            margin: {SPACING.xs}px {SPACING.sm}px;
        }}
        QMessageBox {{
            background-color: {palette.bg_primary};
        }}
        QMessageBox QLabel {{
            color: {palette.text_primary};
        }}
    """


def apply_theme(app: QApplication):
    """Apply the current theme to the entire application."""
    tm = get_theme_manager()
    palette = tm.palette

    # Build complete application stylesheet
    stylesheet = f"""
        {generate_main_window_style(palette)}
        {generate_tabwidget_style(palette)}
        {generate_groupbox_style(palette)}
        {generate_combobox_style(palette)}
        {generate_textedit_style(palette)}
        {generate_slider_style(palette)}
        {generate_progressbar_style(palette)}
        {generate_scrollarea_style(palette)}
    """

    app.setStyleSheet(stylesheet)

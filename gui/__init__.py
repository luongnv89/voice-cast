"""
VoiceCast GUI components.

Provides the desktop application UI built with PySide6.
"""

from .engine_controls import ChatterboxControls, CoquiControls, EngineControlsBase, EngineControlsFactory
from .model_manager_widget import ModelManagerWidget
from .styled_widgets import (
    IconLabel,
    SectionHeader,
    StatusLabel,
    StyledButton,
    StyledCard,
    StyledComboBox,
    StyledGroupBox,
    StyledLabel,
    StyledProgressBar,
    StyledSlider,
    StyledTabWidget,
    StyledTextEdit,
)
from .theme import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
    ThemeManager,
    ThemeMode,
    apply_theme,
    get_theme_manager,
)

__all__ = [
    # Engine controls
    "EngineControlsBase",
    "CoquiControls",
    "ChatterboxControls",
    "EngineControlsFactory",
    # Model manager
    "ModelManagerWidget",
    # Theme system
    "ThemeMode",
    "ThemeManager",
    "ColorPalette",
    "LIGHT_PALETTE",
    "DARK_PALETTE",
    "SPACING",
    "TYPOGRAPHY",
    "apply_theme",
    "get_theme_manager",
    # Styled widgets
    "StyledButton",
    "StyledGroupBox",
    "StyledSlider",
    "StyledComboBox",
    "StyledTextEdit",
    "StyledProgressBar",
    "StyledTabWidget",
    "StyledCard",
    "StyledLabel",
    "StatusLabel",
    "IconLabel",
    "SectionHeader",
]

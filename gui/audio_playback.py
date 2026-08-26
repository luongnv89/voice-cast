"""Audio playback controller for VoiceCloningApp.

Extracts audio playback and save functionality from the main application window.
"""

import uuid
from pathlib import Path

import pygame


class AudioPlaybackController:
    """Manages audio playback and save operations.

    Delegated to by VoiceCloningApp for playing and saving generated audio.
    """

    def __init__(self):
        """Initialize the controller."""
        self._current_audio_path = None
        self._audio_available = True
        self._init_pygame()

    @property
    def current_audio(self):
        """Path to the currently generated audio file."""
        return self._current_audio_path

    @current_audio.setter
    def current_audio(self, value):
        """Set the current audio path."""
        self._current_audio_path = value

    @property
    def audio_available(self) -> bool:
        """Whether audio playback is available."""
        return self._audio_available

    def _init_pygame(self):
        """Initialize pygame mixer for audio playback."""
        try:
            pygame.mixer.init()
        except pygame.error as e:
            self._audio_available = False
            print(f"Warning: Audio playback unavailable — pygame.mixer.init() failed: {e}")

    def play(self, ui_delegate) -> bool:
        """Play the current audio file.

        Args:
            ui_delegate: Object providing UI access with:
                - warning(title, message)

        Returns:
            True if playback started, False if unavailable.
        """
        if not self._audio_available:
            ui_delegate.warning(
                "Playback Unavailable",
                "Audio playback is not available on this system.",
            )
            return False

        if self._current_audio and Path(self._current_audio).exists():
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self._current_audio)
            pygame.mixer.music.play()
            return True
        return False

    def save(self, ui_delegate) -> bool:
        """Save the current audio to a file.

        Args:
            ui_delegate: Object providing UI access with:
                - get_save_filename(default_name) -> tuple[str, str]

        Returns:
            True if saved, False otherwise.
        """
        if not self._current_audio:
            return False

        file_path, _ = ui_delegate.get_save_filename(f"cloned_voice_{uuid.uuid4().hex[:8]}.wav", "Wave Files (*.wav)")
        if file_path:
            import shutil

            try:
                shutil.copy2(self._current_audio, file_path)
                return True
            except (OSError, shutil.Error) as e:
                ui_delegate.critical("Save Error", f"Failed to save audio: {e}")
                return False
        return False

    def reset(self):
        """Reset the controller state."""
        self._current_audio = None

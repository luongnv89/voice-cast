"""Model information dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about a TTS model."""

    id: str
    """Unique identifier for the model (e.g., 'coqui-xtts-v2')."""

    engine: str
    """Engine this model belongs to (e.g., 'coqui', 'chatterbox')."""

    name: str
    """Human-readable display name."""

    size_mb: int
    """Approximate size in megabytes."""

    description: str
    """Brief description of the model."""

    model_path_checker: str | None = None
    """Subpath to check for installation (relative to cache dir)."""

    is_installed: bool = field(default=False, compare=False)
    """Whether the model is currently installed."""

    install_path: Path | None = field(default=None, compare=False)
    """Path where model is installed, if installed."""

    def __post_init__(self):
        """Validate model info."""
        if not self.id:
            raise ValueError("Model ID cannot be empty")
        if not self.engine:
            raise ValueError("Engine name cannot be empty")
        if self.size_mb < 0:
            raise ValueError("Size must be non-negative")

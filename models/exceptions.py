"""Custom exceptions for model management."""


class ModelNotInstalledError(Exception):
    """Raised when attempting to use a model that is not installed."""

    def __init__(self, model_id: str, engine: str, install_command: str | None = None):
        self.model_id = model_id
        self.engine = engine
        self.install_command = install_command or f"python vcloner.py --download-models {model_id}"

        message = (
            f"Model '{model_id}' for engine '{engine}' is not installed.\n"
            f"Download it with: {self.install_command}\n"
            f"Or use the GUI Model Manager to download it."
        )
        super().__init__(message)


class ModelDownloadError(Exception):
    """Raised when model download fails."""

    def __init__(self, model_id: str, reason: str):
        self.model_id = model_id
        self.reason = reason
        message = f"Failed to download model '{model_id}': {reason}"
        super().__init__(message)


class ModelNotFoundError(Exception):
    """Raised when a model ID is not found in the registry."""

    def __init__(self, model_id: str, available_models: list[str] | None = None):
        self.model_id = model_id
        self.available_models = available_models or []

        message = f"Model '{model_id}' not found in registry."
        if self.available_models:
            message += f" Available models: {', '.join(self.available_models)}"
        super().__init__(message)

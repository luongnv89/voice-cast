"""Tests for model management functionality."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from models.download_progress import DownloadProgress
from models.exceptions import ModelDownloadError, ModelNotFoundError, ModelNotInstalledError
from models.model_info import ModelInfo
from models.model_registry import ModelRegistry, get_registry


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            id="test-model",
            engine="test-engine",
            name="Test Model",
            size_mb=100,
            description="A test model",
        )

        assert model.id == "test-model"
        assert model.engine == "test-engine"
        assert model.name == "Test Model"
        assert model.size_mb == 100
        assert model.description == "A test model"
        assert model.is_installed is False
        assert model.install_path is None

    def test_model_info_validation(self):
        """Test ModelInfo validation."""
        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            ModelInfo(id="", engine="test", name="Test", size_mb=100, description="Test")

        with pytest.raises(ValueError, match="Engine name cannot be empty"):
            ModelInfo(id="test", engine="", name="Test", size_mb=100, description="Test")

        with pytest.raises(ValueError, match="Size must be non-negative"):
            ModelInfo(id="test", engine="test", name="Test", size_mb=-1, description="Test")


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = DownloadProgress(
            downloaded_bytes=50 * 1024 * 1024,
            total_bytes=100 * 1024 * 1024,
            speed_bytes_per_sec=1024 * 1024,
            eta_seconds=50,
        )

        assert progress.percentage == 50.0
        assert progress.downloaded_mb == 50.0
        assert progress.total_mb == 100.0
        assert progress.speed_mb_per_sec == 1.0

    def test_percentage_with_zero_total(self):
        """Test percentage with unknown total."""
        progress = DownloadProgress(
            downloaded_bytes=1024,
            total_bytes=0,
            speed_bytes_per_sec=1024,
            eta_seconds=-1,
        )

        assert progress.percentage == 0.0

    def test_format_eta(self):
        """Test ETA formatting."""
        # Seconds
        progress = DownloadProgress(0, 100, 0, eta_seconds=45)
        assert progress.format_eta() == "45s"

        # Minutes
        progress = DownloadProgress(0, 100, 0, eta_seconds=125)
        assert progress.format_eta() == "2m 5s"

        # Hours
        progress = DownloadProgress(0, 100, 0, eta_seconds=3725)
        assert progress.format_eta() == "1h 2m"

        # Unknown
        progress = DownloadProgress(0, 100, 0, eta_seconds=-1)
        assert progress.format_eta() == "unknown"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_singleton_pattern(self):
        """Test that ModelRegistry is a singleton."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        assert registry1 is registry2

    def test_list_models(self):
        """Test listing all models."""
        registry = get_registry()
        models = registry.list_models()

        assert len(models) >= 3
        model_ids = [m.id for m in models]
        assert "coqui-xtts-v2" in model_ids
        assert "chatterbox-turbo" in model_ids
        assert "chatterbox-standard" in model_ids

    def test_get_model(self):
        """Test getting a specific model."""
        registry = get_registry()
        model = registry.get_model("coqui-xtts-v2")

        assert model.id == "coqui-xtts-v2"
        assert model.engine == "coqui"
        assert model.name == "Coqui XTTS v2"
        assert model.size_mb > 0

    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist."""
        registry = get_registry()

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_model("nonexistent-model")

        assert "nonexistent-model" in str(exc_info.value)

    def test_get_models_for_engine(self):
        """Test getting models for a specific engine."""
        registry = get_registry()

        coqui_models = registry.get_models_for_engine("coqui")
        assert len(coqui_models) >= 1
        assert all(m.engine == "coqui" for m in coqui_models)

        chatterbox_models = registry.get_models_for_engine("chatterbox")
        assert len(chatterbox_models) >= 2
        assert all(m.engine == "chatterbox" for m in chatterbox_models)

    def test_get_engine_for_model(self):
        """Test getting engine name for a model."""
        registry = get_registry()

        assert registry.get_engine_for_model("coqui-xtts-v2") == "coqui"
        assert registry.get_engine_for_model("chatterbox-turbo") == "chatterbox"

    def test_get_model_id_for_public_engine_name(self):
        """Test mapping generation engine names to model IDs."""
        registry = get_registry()

        assert registry.get_model_id_for_engine("coqui") == "coqui-xtts-v2"
        assert registry.get_model_id_for_engine("chatterbox-turbo") == "chatterbox-turbo"
        assert registry.get_model_id_for_engine("chatterbox-standard") == "chatterbox-standard"

    def test_get_models_for_engine_accepts_groups_and_public_names(self):
        """Test model listing by engine group and public factory name."""
        registry = get_registry()

        chatterbox_models = registry.get_models_for_engine("chatterbox")
        assert {m.id for m in chatterbox_models} >= {"chatterbox-turbo", "chatterbox-standard"}

        turbo_models = registry.get_models_for_engine("chatterbox-turbo")
        assert [m.id for m in turbo_models] == ["chatterbox-turbo"]


class TestExceptions:
    """Tests for custom exceptions."""

    def test_model_not_installed_error(self):
        """Test ModelNotInstalledError."""
        error = ModelNotInstalledError(
            model_id="test-model",
            engine="test-engine",
        )

        assert error.model_id == "test-model"
        assert error.engine == "test-engine"
        assert "test-model" in str(error)
        assert "test-engine" in str(error)
        assert "vcloner.py --download-models" in str(error)

    def test_model_not_installed_error_custom_command(self):
        """Test ModelNotInstalledError with custom install command."""
        error = ModelNotInstalledError(
            model_id="test-model",
            engine="test-engine",
            install_command="custom-install-command",
        )

        assert error.install_command == "custom-install-command"
        assert "custom-install-command" in str(error)

    def test_model_download_error(self):
        """Test ModelDownloadError."""
        error = ModelDownloadError(
            model_id="test-model",
            reason="Network error",
        )

        assert error.model_id == "test-model"
        assert error.reason == "Network error"
        assert "test-model" in str(error)
        assert "Network error" in str(error)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError(
            model_id="unknown-model",
            available_models=["model-a", "model-b"],
        )

        assert error.model_id == "unknown-model"
        assert "unknown-model" in str(error)
        assert "model-a" in str(error)
        assert "model-b" in str(error)


class TestModelDownloader:
    """Tests for ModelDownloader (with mocked network calls)."""

    def test_download_already_installed(self):
        """Test downloading a model that's already installed."""
        from models.model_downloader import ModelDownloader

        downloader = ModelDownloader()

        # Mock the registry to report model as installed
        with patch.object(downloader._registry, "get_model") as mock_get:
            mock_model = ModelInfo(
                id="test-model",
                engine="test",
                name="Test",
                size_mb=100,
                description="Test",
                is_installed=True,
                install_path=Path("/fake/path"),
            )
            mock_get.return_value = mock_model

            result = downloader.download("test-model")
            assert result == Path("/fake/path")

    def test_download_with_progress_callback(self):
        """Test that progress callback is invoked."""
        from models.model_downloader import ModelDownloader

        downloader = ModelDownloader()
        progress_updates = []

        def callback(progress: DownloadProgress):
            progress_updates.append(progress)

        # Mock model as installed to trigger "already installed" path
        with patch.object(downloader._registry, "get_model") as mock_get:
            mock_model = ModelInfo(
                id="test-model",
                engine="test",
                name="Test",
                size_mb=100,
                description="Test",
                is_installed=True,
                install_path=Path("/fake/path"),
            )
            mock_get.return_value = mock_model

            downloader.download("test-model", progress_callback=callback)

            # Should have received at least one progress update
            assert len(progress_updates) >= 1
            assert progress_updates[-1].status == "completed"

    def test_mlx_audio_downloader_is_registered(self):
        """Test MLX Audio models have an explicit downloader route."""
        from models.downloaders.mlx_downloader import MlxDownloader
        from models.model_downloader import ModelDownloader

        downloader = ModelDownloader()
        assert isinstance(downloader._get_downloader("mlx-audio"), MlxDownloader)


def _function_def(source_path: str, class_name: str, function_name: str) -> ast.FunctionDef:
    """Read a function definition without importing heavyweight runtime deps."""
    tree = ast.parse(Path(source_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} not found")


def _default_for_argument(func: ast.FunctionDef, argument_name: str):
    """Return the AST default for an argument."""
    args = func.args.args
    defaults = func.args.defaults
    default_offset = len(args) - len(defaults)
    for index, arg in enumerate(args):
        if arg.arg == argument_name:
            default_index = index - default_offset
            if default_index < 0:
                return None
            return defaults[default_index]
    raise AssertionError(f"argument {argument_name} not found")


def test_engine_auto_download_defaults_to_false():
    """Generation engines must not download models unless explicitly opted in."""
    engine_specs = [
        ("engines/coqui_engine.py", "CoquiEngine"),
        ("engines/chatterbox_engine.py", "ChatterboxEngine"),
        ("engines/mlx_audio_engine.py", "MlxAudioEngine"),
    ]

    for path, class_name in engine_specs:
        init_func = _function_def(path, class_name, "__init__")
        default = _default_for_argument(init_func, "auto_download")
        assert isinstance(default, ast.Constant)
        assert default.value is False


def test_voice_cloner_api_declares_explicit_model_management_methods():
    """VoiceCloner exposes model-management and switching APIs."""
    tree = ast.parse(Path("voice_cloner.py").read_text())
    voice_cloner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VoiceCloner")
    method_names = {node.name for node in voice_cloner.body if isinstance(node, ast.FunctionDef)}

    assert {
        "list_models",
        "is_model_installed",
        "download_model",
        "get_model_id_for_engine",
        "switch_engine",
    } <= method_names

    init_func = _function_def("voice_cloner.py", "VoiceCloner", "__init__")
    default = _default_for_argument(init_func, "auto_download")
    assert isinstance(default, ast.Constant)
    assert default.value is False

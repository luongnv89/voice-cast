"""Tests for vcloner CLI failure signaling (#58) and shared-registry access (#59)."""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import vcloner
from models.exceptions import ModelNotInstalledError
from models.model_info import ModelInfo
from models.model_registry import get_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_main(argv):
    with patch.object(sys, "argv", ["vcloner.py", *argv]):
        vcloner.main()


@pytest.fixture
def shared_canary():
    """Register a canary model on the documented global registry."""
    registry = get_registry()
    registry.register_model(
        ModelInfo(id="cli-shared-canary", engine="test", name="Canary", size_mb=1, description="canary")
    )
    yield registry
    registry._models.pop("cli-shared-canary", None)


class TestCliExitCodes:
    """CLI error paths must exit nonzero so scripts/CI can detect failure."""

    def test_missing_required_generation_args_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            run_main(["-t", "hello"])
        assert excinfo.value.code == 1

    def test_download_models_without_ids_or_engine_exits_nonzero(self):
        with (
            patch("vcloner.bootstrap_engines"),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(["--download-models"])
        assert excinfo.value.code == 1

    def test_model_group_as_generation_engine_exits_nonzero(self):
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(["-i", "voice.wav", "-t", "hi", "-o", "out.wav", "-e", "chatterbox"])
        assert excinfo.value.code == 1

    @pytest.mark.parametrize(
        "error",
        [
            ModelNotInstalledError("coqui-xtts-v2", "coqui"),
            FileNotFoundError("voice.wav"),
            ImportError("No module named 'TTS'"),
            RuntimeError("boom"),
        ],
    )
    def test_generation_exception_handlers_exit_nonzero(self, error):
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            patch("vcloner.VoiceCloner", side_effect=error),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(["-i", "voice.wav", "-t", "hi", "-o", "out.wav"])
        assert excinfo.value.code == 1

    def test_unknown_engine_for_download_errors_nonzero(self, shared_canary):
        with pytest.raises(SystemExit) as excinfo:
            vcloner.download_models([], engine="bogus-engine")
        assert excinfo.value.code == 1


class TestCliSuccessPathsDoNotExit:
    """Success paths must keep returning normally (exit code 0)."""

    def test_list_engines_returns_normally(self):
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.get_engine_info", return_value={"coqui": "Coqui XTTS v2"}),
            patch("vcloner.TTSFactory.is_available", return_value=True),
        ):
            assert run_main(["--list-engines"]) is None

    def test_happy_generation_path_returns_normally(self, tmp_path):
        cloner = MagicMock()
        out = tmp_path / "out.wav"
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            patch("vcloner.VoiceCloner", return_value=cloner) as mock_cloner,
        ):
            assert run_main(["-i", "voice.wav", "-t", "hi", "-o", str(out)]) is None
        mock_cloner.assert_called_once()
        cloner.generate.assert_called_once()
        assert cloner.generate.call_args.kwargs["chunk_size"] is None
        assert cloner.generate.call_args.kwargs["silence_duration"] == 200
        assert cloner.generate.call_args.kwargs["play_audio"] is True

    def test_generation_forwards_chunking_overrides(self, tmp_path):
        cloner = MagicMock()
        out = tmp_path / "out.wav"
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            patch("vcloner.VoiceCloner", return_value=cloner),
        ):
            run_main(
                [
                    "-i",
                    "voice.wav",
                    "-t",
                    "hi",
                    "-o",
                    str(out),
                    "--chunk-size",
                    "120",
                    "--silence-duration",
                    "350",
                    "--no-play",
                ]
            )

        assert cloner.generate.call_args.kwargs["chunk_size"] == 120
        assert cloner.generate.call_args.kwargs["silence_duration"] == 350
        assert cloner.generate.call_args.kwargs["play_audio"] is False
        assert callable(cloner.generate.call_args.kwargs["chunk_progress_callback"])

    def test_generation_renders_chunk_progress_and_completes_after_success(self, tmp_path):
        cloner = MagicMock()
        progress = MagicMock()
        progress.add_task.return_value = 7
        progress_context = MagicMock()
        progress_context.__enter__.return_value = progress
        progress_context.__exit__.return_value = False

        def generate(*_args, **kwargs):
            callback = kwargs["chunk_progress_callback"]
            callback(1, 3)
            callback(2, 3)
            callback(3, 3)

        cloner.generate.side_effect = generate
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            patch("vcloner.VoiceCloner", return_value=cloner),
            patch("vcloner.Progress", return_value=progress_context),
        ):
            run_main(["-i", "voice.wav", "-t", "long text", "-o", str(tmp_path / "out.wav")])

        progress.add_task.assert_called_once_with("Preparing chunks...", total=None)
        updates = [(item.args, item.kwargs) for item in progress.update.call_args_list]
        assert updates == [
            ((7,), {"total": 3, "completed": 0, "description": "Chunk 1 of 3"}),
            ((7,), {"total": 3, "completed": 1, "description": "Chunk 2 of 3"}),
            ((7,), {"total": 3, "completed": 2, "description": "Chunk 3 of 3"}),
            ((7,), {"completed": 3, "description": "Chunk 3 of 3"}),
        ]

    def test_generation_failure_does_not_mark_chunk_progress_complete(self, tmp_path):
        cloner = MagicMock()
        progress = MagicMock()
        progress.add_task.return_value = 9
        progress_context = MagicMock()
        progress_context.__enter__.return_value = progress
        progress_context.__exit__.return_value = False

        def fail(*_args, **kwargs):
            kwargs["chunk_progress_callback"](1, 2)
            raise RuntimeError("boom")

        cloner.generate.side_effect = fail
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            patch("vcloner.VoiceCloner", return_value=cloner),
            patch("vcloner.Progress", return_value=progress_context),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(["-i", "voice.wav", "-t", "long text", "-o", str(tmp_path / "out.wav")])

        assert excinfo.value.code == 1
        progress.update.assert_called_once_with(
            9,
            total=2,
            completed=0,
            description="Chunk 1 of 2",
        )

    def test_help_documents_chunking_options(self, capsys):
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(["--help"])

        assert excinfo.value.code == 0
        help_text = " ".join(capsys.readouterr().out.split())
        assert "--silence-duration MS" in help_text
        assert "Default: 200" in help_text
        assert "--chunk-size CHARS" in help_text
        assert "Default: selected engine limit" in help_text

    @pytest.mark.parametrize(
        "invalid_option",
        [
            ["--silence-duration", "-1"],
            ["--chunk-size", "0"],
        ],
    )
    def test_chunking_options_reject_invalid_values(self, invalid_option):
        with (
            patch("vcloner.bootstrap_engines"),
            patch("vcloner.TTSFactory.available_engines", return_value=["coqui"]),
            pytest.raises(SystemExit) as excinfo,
        ):
            run_main(invalid_option)
        assert excinfo.value.code == 2


class TestSharedRegistrySites:
    """Production registry access must go through the get_registry() singleton."""

    def test_cli_list_models_observes_shared_registry_state(self, shared_canary):
        with vcloner.console.capture() as capture:
            vcloner.list_models()
        assert "cli-shared-canary" in capture.get()

    def test_cli_download_models_resolves_ids_from_shared_registry(self, shared_canary):
        downloader = MagicMock()
        with patch("vcloner.ModelDownloader", return_value=downloader):
            vcloner.download_models(["cli-shared-canary"])
        downloader.download.assert_called_once()

    def test_no_private_modelregistry_construction_outside_definition(self):
        """grep-equivalent: ModelRegistry() may only appear in its definition module.

        The GUI site cannot be imported here (PySide6 is not a test dependency),
        so the guard is enforced statically for every non-test source file.
        """
        offenders = []
        skip_dirs = {"tests", "__pycache__", ".git", ".venv", "venv", ".gitissue", "node_modules", "build"}
        for path in REPO_ROOT.rglob("*.py"):
            if skip_dirs.intersection(path.parts) or any(part.startswith(".") for part in path.parts[:-1]):
                continue
            if path.relative_to(REPO_ROOT).as_posix() == "models/model_registry.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ModelRegistry":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert offenders == []

    def test_gui_model_manager_source_uses_get_registry(self):
        source = (REPO_ROOT / "gui" / "model_manager_widget.py").read_text(encoding="utf-8")
        assert "self._registry = get_registry()" in source

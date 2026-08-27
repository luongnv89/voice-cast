"""Tests for Audio8 TTS model downloader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from models.download_progress import DownloadProgress


class TestAudio8DownloaderModelInfo:
    """Tests for Audio8Downloader model information."""

    def test_model_repos_contains_audio8_tts(self):
        """Test that MODEL_REPOS contains audio8-tts."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        assert "audio8-tts" in Audio8Downloader.MODEL_REPOS

    def test_model_repos_value(self):
        """Test the correct HuggingFace repo ID."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        assert Audio8Downloader.MODEL_REPOS["audio8-tts"] == "Audio8/audio8-TTS-0.1B-ONNX-INT8"

    def test_model_sizes_contains_audio8_tts(self):
        """Test that MODEL_SIZES contains audio8-tts."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        assert "audio8-tts" in Audio8Downloader.MODEL_SIZES

    def test_model_size_is_reasonable(self):
        """Test that the model size is approximately 2GB."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        size = Audio8Downloader.MODEL_SIZES["audio8-tts"]
        assert size == 2000 * 1024 * 1024  # 2 GB


class TestAudio8DownloaderDownload:
    """Tests for the download method."""

    def test_download_unknown_model_raises(self):
        """Test that downloading an unknown model raises ValueError."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        downloader = Audio8Downloader()

        with pytest.raises(ValueError, match="Unknown Audio8 model"):
            downloader.download("unknown-model")

    @patch("huggingface_hub.snapshot_download")
    def test_download_returns_path(self, mock_snapshot):
        """Test that download returns a Path."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        mock_snapshot.return_value = "/tmp/hf-cache/audio8-tts"

        downloader = Audio8Downloader()
        result = downloader.download("audio8-tts")

        assert isinstance(result, Path)
        mock_snapshot.assert_called_once()

    @patch("huggingface_hub.snapshot_download")
    def test_download_with_progress_callback(self, mock_snapshot):
        """Test that progress callback is invoked."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        mock_snapshot.return_value = "/tmp/hf-cache/audio8-tts"

        downloader = Audio8Downloader()
        progress_updates = []

        def callback(progress: DownloadProgress):
            progress_updates.append(progress)

        downloader.download("audio8-tts", progress_callback=callback)

        # Should have at least start and completion
        assert len(progress_updates) >= 2
        assert progress_updates[0].status == "downloading"
        assert progress_updates[-1].status == "completed"

    @patch("huggingface_hub.snapshot_download")
    def test_download_calls_snapshot_with_correct_repo(self, mock_snapshot):
        """Test that snapshot_download is called with the correct repo."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        mock_snapshot.return_value = "/tmp/hf-cache/audio8-tts"

        downloader = Audio8Downloader()
        downloader.download("audio8-tts")

        call_kwargs = mock_snapshot.call_args
        assert call_kwargs.kwargs["repo_id"] == "Audio8/audio8-TTS-0.1B-ONNX-INT8"

    @patch("huggingface_hub.snapshot_download")
    def test_download_uses_registry_cache_dir(self, mock_snapshot):
        """Test that download uses the registry cache directory."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        mock_snapshot.return_value = "/tmp/hf-cache/audio8-tts"

        downloader = Audio8Downloader()
        downloader.download("audio8-tts")

        call_kwargs = mock_snapshot.call_args
        assert "cache_dir" in call_kwargs.kwargs


class TestAudio8DownloaderGetModelSize:
    """Tests for get_model_size method."""

    def test_get_model_size_known_model(self):
        """Test get_model_size for known model."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        downloader = Audio8Downloader()
        size = downloader.get_model_size("audio8-tts")

        assert size == 2000 * 1024 * 1024

    def test_get_model_size_unknown_model(self):
        """Test get_model_size for unknown model returns 0."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        downloader = Audio8Downloader()
        size = downloader.get_model_size("unknown-model")

        assert size == 0


class TestAudio8DownloaderIntegration:
    """Integration-style tests for Audio8Downloader."""

    def test_downloader_instantiation(self):
        """Test that Audio8Downloader can be instantiated."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        downloader = Audio8Downloader()
        assert downloader is not None

    def test_downloader_has_required_methods(self):
        """Test that Audio8Downloader has required interface methods."""
        from models.downloaders.audio8_downloader import Audio8Downloader

        downloader = Audio8Downloader()
        assert hasattr(downloader, "download")
        assert hasattr(downloader, "get_model_size")

"""CLI and file output must not initialize the native playback backend."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["help", "list-engines", "generate"])
def test_headless_operations_do_not_import_sounddevice(mode, tmp_path):
    script = """
import builtins
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Use the suite's centralized optional-ML stubs, not real models or downloads.
runpy.run_path("tests/conftest.py")
original_import = builtins.__import__
def block_playback(name, *args, **kwargs):
    if name == "sounddevice":
        raise OSError("PortAudio library not found")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_playback

mode, output = sys.argv[1:]
if mode == "generate":
    import numpy as np
    import soundfile as sf
    from voice_cloner import VoiceCloner
    engine = MagicMock()
    engine.name = "synthetic"
    engine.requires_reference_audio = False
    engine.MAX_CHUNK_CHARS = 0
    engine.generate.return_value = (np.zeros(16, dtype=np.float32), 22050)
    cloner = VoiceCloner(speaker_wav="", engine=engine)
    assert cloner.generate("synthetic test", output_file=output, play_audio=False) == output
    audio, rate = sf.read(output)
    assert len(audio) == 16 and rate == 22050
    assert "sounddevice" not in sys.modules
else:
    sys.argv = ["vcloner", "--" + mode]
    runpy.run_module("vcloner", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode, str(tmp_path / "synthetic.wav")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

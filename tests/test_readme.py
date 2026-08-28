"""Regression tests for the documented installation workflow."""

import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_quickstart_installs_a_supported_default_engine():
    """The literal quickstart must install an engine before downloading its model."""
    content = README.read_text()
    quickstart = content.split("## Get Started in 60 Seconds", 1)[1].split("```", 2)[1]

    assert re.search(r'^pip install -e "\.\[coqui\]"$', quickstart, re.MULTILINE)
    assert "python vcloner.py --download-models coqui-xtts-v2" in quickstart


def test_readme_documents_incompatible_engine_environments():
    """README must not promise seamless switching across incompatible extras."""
    content = README.read_text().lower()

    assert "seamless" not in content
    assert "separate virtual environment" in content

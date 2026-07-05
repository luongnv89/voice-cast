"""Tests to verify the VoiceCast rename was successful."""

import os


def test_project_name_in_pyproject():
    """Verify pyproject.toml has the new project name."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()

    assert 'name = "voicecast"' in content
    assert "Your words, any voice" in content


def test_readme_has_new_name():
    """Verify README.md has VoiceCast branding."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme_path) as f:
        content = f.read()

    assert "VoiceCast" in content
    assert "Your words, any voice" in content
    assert "voice-cast.git" in content
    assert "voicecast-app.png" in content


def test_screenshot_renamed():
    """Verify screenshot file was renamed."""
    screenshot_path = os.path.join(os.path.dirname(__file__), "..", "voicecast-app.png")
    old_screenshot = os.path.join(os.path.dirname(__file__), "..", "voice-cloner-app.png")

    assert os.path.exists(screenshot_path), "New screenshot voicecast-app.png should exist"
    assert not os.path.exists(old_screenshot), "Old screenshot voice-cloner-app.png should not exist"


def test_docs_updated():
    """Verify docs have been updated with new name."""
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")

    # Check troubleshooting.md
    with open(os.path.join(docs_dir, "troubleshooting.md")) as f:
        content = f.read()
    assert "VoiceCast" in content
    assert "luongnv89/voice-cast/issues" in content

    # Check development.md
    with open(os.path.join(docs_dir, "development.md")) as f:
        content = f.read()
    assert "VoiceCast" in content
    assert "voice-cast.git" in content

    # Check gui-guide.md
    with open(os.path.join(docs_dir, "gui-guide.md")) as f:
        content = f.read()
    assert "VoiceCast" in content
    assert "voicecast-app.png" in content

    # Check architecture.md
    with open(os.path.join(docs_dir, "architecture.md")) as f:
        content = f.read()
    assert "VoiceCast" in content


def test_ci_workflow_updated():
    """Verify CI workflow comment was updated."""
    ci_path = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml")
    with open(ci_path) as f:
        content = f.read()

    assert "VoiceCast" in content


def test_no_old_references_in_key_files():
    """Verify no 'voice-cloner' references remain in key files."""
    files_to_check = [
        ("pyproject.toml", ["name = ", "description = "]),
        ("README.md", ["# ", "git clone", "cd "]),
        ("docs/troubleshooting.md", ["# ", "github.com"]),
        ("docs/development.md", ["# ", "git clone", "cd ", "## Project Structure"]),
    ]

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    for filename, contexts in files_to_check:
        filepath = os.path.join(base_dir, filename)
        with open(filepath) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            for context in contexts:
                if context in line:
                    # These lines should not contain 'voice-cloner'
                    assert (
                        "voice-cloner" not in line.lower()
                    ), f"Found 'voice-cloner' in {filename}:{i + 1}: {line.strip()}"

"""Tests guarding the pinned requirements.txt lockfile.

The lockfile is consumed by the CI security job (`safety check -r
requirements.txt`) and the pre-commit safety hook (`files:
requirements\\.txt$`). These tests keep it present, fully pinned, and in
sync with the runtime dependencies declared in pyproject.toml.
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_dep_names():
    """Return canonical names of the [project] dependencies in pyproject.toml."""
    content = _read("pyproject.toml")
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", content, re.S | re.MULTILINE)
    assert match, "pyproject.toml must declare [project] dependencies"
    names = re.findall(r'"([A-Za-z0-9_.-]+?)\s*[\[=<>!~;"]', match.group(1))
    return sorted(_canonical(name) for name in names)


def _pinned_names():
    """Return {canonical name: pinned version} from requirements.txt."""
    pins = {}
    for line in _read("requirements.txt").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+)", line)
        if match:
            pins[_canonical(match.group(1))] = match.group(2)
    return pins


def test_requirements_txt_exists_at_repo_root():
    assert os.path.isfile(os.path.join(ROOT, "requirements.txt")), "requirements.txt must exist at the repo root"


def test_every_entry_is_exactly_pinned():
    lines = [
        line.strip()
        for line in _read("requirements.txt").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "requirements.txt must contain pinned entries"
    unpinned = [line for line in lines if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*==\S+", line)]
    assert not unpinned, f"All entries must be exact == pins, found: {unpinned}"


def test_all_runtime_deps_are_pinned():
    runtime_deps = _runtime_dep_names()
    assert len(runtime_deps) >= 10, "expected the 10 declared runtime dependencies"
    pins = _pinned_names()
    missing = [name for name in runtime_deps if name not in pins]
    assert not missing, f"Runtime dependencies missing from requirements.txt: {missing}"


def test_lockfile_consumers_reference_it_validly():
    ci_workflow = _read(os.path.join(".github", "workflows", "ci.yml"))
    assert "safety check -r requirements.txt" in ci_workflow
    pre_commit = _read(".pre-commit-config.yaml")
    assert re.search(
        r"^\s+files:\s*requirements\\\.txt\$\s*$", pre_commit, re.MULTILINE
    ), "pre-commit safety hook must target requirements.txt"

"""Tests guarding declared dependency floors and tool pin alignment.

Covers the pyproject.toml dependency-surface contracts: bumped floors for
soundfile/PySide6/rich/mlx/numpy (#53), mutually exclusive engine extras, and
one ruff version line shared by ci.yml, the dev extra, and the pre-commit rev.
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_specs():
    """Return {canonical name: version spec} for [project] dependencies."""
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        content = f.read()
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", content, re.S | re.MULTILINE)
    assert match, "pyproject.toml must declare [project] dependencies"
    specs = {}
    for raw in match.group(1).splitlines():
        entry = raw.strip().strip(",").strip('"')
        if not entry or entry.startswith("#"):
            continue
        name = re.match(r"[A-Za-z0-9_.-]+", entry).group(0)
        spec = entry[len(name) :].split(";")[0].strip()
        specs[_canonical(name)] = spec
    return specs


def _optional_extra_specs(extra):
    """Return {canonical name: version spec} for one [project.optional-dependencies] entry."""
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        content = f.read()
    match = re.search(rf"^{extra}\s*=\s*\[(.*?)^\]", content, re.S | re.MULTILINE)
    assert match, f"pyproject.toml must declare the {extra!r} optional extra"
    specs = {}
    for raw in match.group(1).splitlines():
        entry = raw.strip().strip(",").strip('"')
        if not entry or entry.startswith("#"):
            continue
        name = re.match(r"[A-Za-z0-9_.-]+", entry).group(0)
        spec = entry[len(name) :].split(";")[0].strip()
        specs[_canonical(name)] = spec
    return specs


def _floor_major_minor(spec):
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert match, f"spec {spec!r} must carry a >=x.y floor"
    return (int(match.group(1)), int(match.group(2)))


def test_dependency_floors_track_current_stable():
    specs = _declared_specs()
    expected_floors = {
        "soundfile": (0, 14),
        "pyside6": (6, 11),
        "rich": (15, 0),
        "numpy": (1, 26),
    }
    for name, floor in expected_floors.items():
        assert name in specs, f"{name} must stay a declared runtime dependency"
        assert _floor_major_minor(specs[name]) == floor, (
            f"{name} floor must be >={floor[0]}.{floor[1]}, found {specs[name]!r}"
        )


def test_mlx_optional_extra_floor_tracks_current_stable():
    specs = _optional_extra_specs("mlx")
    assert "mlx" in specs, "the mlx optional extra must declare mlx"
    assert _floor_major_minor(specs["mlx"]) == (0, 32)


def test_engine_extras_keep_transformers_constraints_separate():
    core_specs = _declared_specs()
    coqui_specs = _optional_extra_specs("coqui")
    chatterbox_specs = _optional_extra_specs("chatterbox")
    audio8_specs = _optional_extra_specs("audio8")

    assert "transformers" not in core_specs, "incompatible engine constraints must not be in the shared core"
    assert coqui_specs["torch"].endswith("<2.9")
    assert coqui_specs["torchaudio"].endswith("<2.9")
    assert _floor_major_minor(coqui_specs["transformers"]) == (5, 16)
    assert coqui_specs["transformers"].endswith("<5.17")
    assert chatterbox_specs["transformers"] == "==5.2.0"
    assert _floor_major_minor(audio8_specs["transformers"]) == (4, 46)
    assert audio8_specs["transformers"].endswith("<5.0.0")
    assert _floor_major_minor(audio8_specs["huggingface-hub"]) == (0, 24)
    assert audio8_specs["huggingface-hub"].endswith("<1.0")
    assert "coqui-tts" in coqui_specs
    assert "tts" not in core_specs and "chatterbox-tts" not in core_specs

    with open(os.path.join(ROOT, "voice_cloner.py")) as f:
        source = f.read()
    assert re.search(r"^\s*from transformers import logging", source, re.M), (
        "the optional Transformers edge is guarded by a lazy import"
    )


def test_incompatible_engine_extras_are_not_combined():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        content = f.read()
    assert re.search(r"^coqui\s*=", content, re.M)
    assert re.search(r"^chatterbox\s*=", content, re.M)
    assert not re.search(r"^all\s*=", content, re.M), "there must be no resolver-invalid all-engines extra"


def _dev_ruff_spec():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        content = f.read()
    match = re.search(r'"ruff([><=][^"]*)"', content)
    assert match, "pyproject dev extra must pin ruff"
    return match.group(1)


def test_ruff_pins_agree_across_ci_pyproject_and_pre_commit():
    dev_spec = _dev_ruff_spec()
    dev_line = _floor_major_minor(dev_spec)

    with open(os.path.join(ROOT, ".github", "workflows", "ci.yml")) as f:
        ci_workflow = f.read()
    ci_match = re.search(r"'ruff[><=][^']*'", ci_workflow)
    assert ci_match, "ci.yml lint job must pin ruff"
    assert _floor_major_minor(ci_match.group(0)[1:-1]) == dev_line, (
        "ci.yml ruff pin must share the dev extra's major.minor line"
    )

    with open(os.path.join(ROOT, ".pre-commit-config.yaml")) as f:
        pre_commit = f.read()
    rev_match = re.search(r"astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v(\d+)\.(\d+)\.(\d+)", pre_commit)
    assert rev_match, "pre-commit must pin an astral-sh/ruff-pre-commit rev"
    precommit_line = (int(rev_match.group(1)), int(rev_match.group(2)))
    assert precommit_line == dev_line, "pre-commit ruff rev must share the dev extra's major.minor line"

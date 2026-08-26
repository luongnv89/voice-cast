# Agent Environment Notes

Setup and run notes for AI agents working autonomously on VoiceCast. Everything
here is derived from project files (`pyproject.toml`, `.github/workflows/ci.yml`,
`.pre-commit-config.yaml`). Human contributors can use it too, but the audience
is an agent that must set up, build, test, and lint without asking questions.

## Toolchain

| Requirement | Value | Source |
|-------------|-------|--------|
| Python | **3.10 – 3.12** (CI test matrix) | `.github/workflows/ci.yml` |
| Python (lint/security jobs) | 3.11 | `.github/workflows/ci.yml` |
| `requires-python` | `>=3.10` | `pyproject.toml` |

- The local interpreter on some dev machines (e.g. Python 3.14.x) is **untested**
  against this project — prefer 3.10, 3.11, or 3.12.
- Ruff targets `py310` (`pyproject.toml` → `[tool.ruff] target-version`).

### Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Installing dependencies

```bash
# Full development environment (core + dev tools)
pip install -e ".[dev]"

# Minimal environment sufficient for the test suite (what CI does)
pip install sounddevice soundfile rich numpy
pip install -e . --no-deps
```

The test suite mocks heavyweight modules (`torch`, `torchaudio`, `TTS`,
`chatterbox-tts`, `transformers` are **not** imported by tests) — see the
`sys.modules` stubs at the top of `tests/test_voice_cloner.py`. You do not need
the ML stack to run tests.

## System prerequisites

| Package | Why | Install (Debian/Ubuntu) |
|---------|-----|--------------------------|
| `libportaudio2` | Audio playback/recording (`sounddevice`) | `sudo apt-get install -y libportaudio2` |
| `libsndfile1` | WAV/audio file I/O (`soundfile`) | `sudo apt-get install -y libsndfile1` |

Both are installed by the CI test job (`.github/workflows/ci.yml`).

### torch / torchaudio

- Core dependency: `torch>=2.5.1`, `torchaudio>=2.5.1`.
- Plain `pip install` gets CPU wheels. For NVIDIA GPUs, install from the PyTorch
  CUDA index instead (https://pytorch.org → pick your CUDA version).
- Optional MLX backend (Apple Silicon only): `pip install -e ".[mlx]"`.

### PySide6 (GUI)

- Core dependency: `PySide6>=6.4.0`; pip wheels bundle the Qt libraries.
- The desktop GUI (`voice_cloning_app.py`) requires a running display server
  (X11/Wayland). CI never launches the GUI, so headless environments can skip
  it — but it remains part of `-e .` core installs.

## Environment variables

Only two are read from the environment (see `models/model_registry.py`):

| Variable | Purpose | Default when unset |
|----------|---------|--------------------|
| `HF_HOME` | Root of the HuggingFace hub cache | `~/.cache/huggingface` |
| `LOCALAPPDATA` | Windows-only base for the Coqui TTS cache | user home directory |

There is **no `.env` / dotenv mechanism** in this codebase and no `.env.example`
file — do not create one and expect the app to read it. Model downloads are
always explicit (`python vcloner.py --list-models`,
`python vcloner.py --download-models <model-id>`); nothing downloads at import
or install time.

## Commands of record

Run these from the repository root inside the virtual environment:

```bash
# Tests (command of record)
pytest tests/ -v --tb=short

# Lint and format check (CI lint job)
ruff check .
ruff format --check .

# Security scan (CI security job)
bandit -c pyproject.toml -r . --exclude tests

# Pre-commit hooks (CI pre-commit job)
pre-commit install        # once per clone
pre-commit run --all-files
```

CI runs the same commands (`.github/workflows/ci.yml`): lint → ruff/bandit,
test matrix → `pytest tests/ -v --tb=short` on 3.10/3.11/3.12, plus pre-commit
and security jobs.

## Known gaps

- **No lockfile yet.** Sprint 0 of the modernization plan supplies one. Until
  then, installs resolve loose version ranges from `pyproject.toml`, and the
  commands above may fail locally even though they pass in CI.
- **No `requirements.txt`.** The CI *security* job references one, but the step
  is non-blocking (`|| true`) and the file does not exist.

## See also

- [Development Guide](development.md) — contributor workflow, engine how-to
- [Model Management](model-management.md) — explicit download/cache behavior
- [Architecture](architecture.md) — system design

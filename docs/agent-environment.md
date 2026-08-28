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
| `requires-python` | `>=3.10,<3.13` | `pyproject.toml` |

- The local interpreter on some dev machines (e.g. Python 3.14.x) is **unsupported**
  by this project — use 3.10, 3.11, or 3.12.
- Ruff targets `py310` (`pyproject.toml` → `[tool.ruff] target-version`).

### Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Installing dependencies

```bash
# Full development environment (shared core + dev tools)
pip install -e ".[dev]"

# Optional engine environment — choose one, never both:
pip install -e ".[coqui]"       # Coqui TTS fork, Torch 2.5–2.8, Transformers 5.16
# pip install -e ".[chatterbox]" # Python 3.10–3.12, Torch 2.6, Transformers 5.2
# pip install -e ".[audio8]"     # Audio8 ONNX, Transformers 4.x (>=4.46.3)
# pip install -e ".[mlx]"        # Apple Silicon only

# Minimal environment sufficient for the test suite (what CI does)
pip install sounddevice soundfile rich numpy
pip install -e . --no-deps
```

The test suite never requires heavyweight modules (`torch`, `TTS`,
`chatterbox`, `mlx_audio`, `transformers`): `tests/conftest.py` installs a
meta-path finder that serves lightweight stubs for these — but only when the
real package is absent, so environments that have them exercise the real
imports. Stubbing is centralized and order-independent; no test file mutates
import state itself. You do not need the ML stack to run tests.

### Bootstrapping from the lockfile (proven)

The lockfile pins the shared runtime **and dev** tooling (`pytest`,
`pytest-cov`, `ruff`, `bandit`, `pre-commit`), so one install gives you
everything the commands of record below need. Engine extras are intentionally
not included because the Transformers-backed engines require incompatible
major versions. Install one engine extra in a dedicated environment when
running that engine.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
pytest tests/ -v --tb=short
```

Gotchas discovered during that verification:

- **Pin the interpreter to 3.10–3.12 explicitly.** Dev-machine defaults can be
  Python 3.14 (unsupported); the project declares `<3.13`. Use
  `python3.11`/`python3.12`, not bare `python3`.
- **Engine extras are mutually exclusive.** The supported Coqui TTS fork uses
  Transformers 5.16, Chatterbox 0.1.7 uses `transformers==5.2.0`, and Audio8
  uses Transformers 4.x (>=4.46.3) on Python 3.10–3.12. Use separate virtual
  environments; there is deliberately no `all` extra.
- **Regenerate the shared lock with `--extra dev`.**
  `uv pip compile pyproject.toml --universal --extra dev -o requirements.txt`
  excludes engine extras by design. The dev extra caps `ruff < 0.17` on
  purpose — that is the range the CI lint job uses, so the locked formatter
  never disagrees with CI.
- **Headless Qt needs the offscreen platform.** PySide6 ships in the lock, so
  the engine-control widget tests actually execute locally (20 tests that CI
  skips via `pytest.importorskip("PySide6")` because its minimal install never
  installs PySide6). Without a display server, creating a `QApplication`
  aborts the process; `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`
  unless you override it.

## System prerequisites

| Package | Why | Install (Debian/Ubuntu) |
|---------|-----|--------------------------|
| `libportaudio2` | Audio playback/recording (`sounddevice`) | `sudo apt-get install -y libportaudio2` |
| `libsndfile1` | WAV/audio file I/O (`soundfile`) | `sudo apt-get install -y libsndfile1` |

Both are installed by the CI test job (`.github/workflows/ci.yml`).

### Engine dependencies

- The shared install does not include PyTorch or Transformers.
- Coqui: `pip install -e ".[coqui]"` in Python 3.10–3.12; this selects
  `coqui-tts==0.27.5`, PyTorch 2.5–2.8, and Transformers 5.16.x. The Torch
  ceiling avoids the coqui-tts torchcodec import requirement.
- Chatterbox: `pip install -e ".[chatterbox]"` in Python 3.10–3.12; this
  selects Chatterbox 0.1.7, PyTorch 2.6.0, and Transformers 5.2.0.
- Audio8: `pip install -e ".[audio8]"`; this selects ONNX Runtime, the
  Transformers 4.x tokenizer (>=4.46.3), and the Hugging Face Hub downloader.
- For CPU wheels, install matching PyTorch packages from the CPU index before
  the engine extra. For NVIDIA GPUs, use the appropriate PyTorch CUDA index
  instead (https://pytorch.org). Never combine incompatible engine extras.
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
| `TTS_HOME` | Base for the Coqui TTS cache (outranks `XDG_DATA_HOME`) | unset |
| `XDG_DATA_HOME` | Linux base for the Coqui TTS cache when `TTS_HOME` is unset | `~/.local/share` |
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
scripts/pip_audit_lockfile.sh

# Pre-commit hooks (CI pre-commit job)
pre-commit install        # once per clone
pre-commit run --all-files
```

CI runs the same commands (`.github/workflows/ci.yml`): lint → ruff/bandit,
test matrix → `pytest tests/ -v --tb=short` on 3.10/3.11/3.12, plus pre-commit
and security jobs.

## Known gaps

- **The CI *test* job does not install from the lockfile** — it uses the
  minimal `--no-deps` install above, so heavyweight engines stay unexercised
  there and PySide6-dependent tests skip. Full-lock local environments are the
  ones that exercise them (see the bootstrap section above).
- `requirements.txt` is the fully pinned universal lock of the shared
  `pyproject.toml` dependencies plus `dev`; it is consumed by the CI security
  job and the pre-commit pip-audit hook. Regenerate it whenever those
  dependencies change (`tests/test_requirements_lock.py` enforces sync).
  Engine extras are resolved and audited in their own virtual environments and
  must not be combined. The pip-audit baseline lives in
  `scripts/pip_audit_lockfile.sh`: every advisory known at baseline time is an
  explicit `--ignore-vuln`, and any advisory outside that list fails CI — remove
  entries by upgrading the affected package, not by re-baselining.

## See also

- [Development Guide](development.md) — contributor workflow, engine how-to
- [Model Management](model-management.md) — explicit download/cache behavior
- [Architecture](architecture.md) — system design

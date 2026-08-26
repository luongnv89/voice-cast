# Changelog

## Unreleased

### Fixed

- Model downloads now stream progress through the previously unwired callback
  wrappers (`HuggingFaceProgressCallback`, `CoquiProgressCallback`) instead of
  jumping 0→100%: Chatterbox and MLX forward `snapshot_download` bar updates
  via a `tqdm_class` bridge, Coqui streams TTS's own transfer bars into a
  reporthook adapter, so cancellation (`isInterruptionRequested()`) is now
  reachable mid-transfer; closing the app waits on active download threads for
  at most 5 s per thread instead of an unbounded join (#57)

- The model registry's Coqui cache dir is now derived from the Coqui backend's
  own default (`TTS.utils.generic_utils.get_user_data_dir("tts")`, TTS 0.22.0)
  instead of hardcoding `~/.local/share` on Linux, so `TTS_HOME` and
  `XDG_DATA_HOME` overrides are honored and the download→registry→engine
  lazy-loader path agreement no longer rests on implicit convention; pinned by
  a download→`is_installed` round-trip regression test (#55)

### Documentation

- Added `docs/agent-environment.md` with agent-facing setup and run notes (#46)
- Added `CLAUDE.md` and `AGENTS.md` at the repo root pointing to the environment notes (#47, #48)
- Documented the proven lockfile bootstrap (fresh venv → install → suite) with
  discovered gotchas: interpreter pinning, CUDA torch wheel size, `--extra dev`
  regeneration, and headless Qt (#50)

### Infrastructure

- Extended the pinned `requirements.txt` to cover the `dev` extra (`pytest`,
  `pytest-cov`, `ruff`, `bandit` + transitives) so a fresh
  `pip install -r requirements.txt -e .` yields a complete test environment;
  all existing pins unchanged (#50)
- Added `tests/conftest.py` defaulting `QT_QPA_PLATFORM=offscreen` so Qt
  widget tests can run on headless machines where PySide6 is installed (#50)
- Added fully pinned `requirements.txt` (universal uv lock of `pyproject.toml`
  runtime deps) so the CI security job's `safety check` and the pre-commit
  safety hook resolve; documented regeneration in `docs/agent-environment.md`
  (#49)
- CI pre-commit job now sets `SKIP=python-safety-dependencies-check`, matching
  the skip already declared in `.pre-commit-config.yaml` (`ci.skip` is only
  honored by pre-commit.ci, so the hook otherwise runs on GitHub Actions once
  `requirements.txt` exists); dependency scanning remains in the non-blocking
  Security Scan job

### Testing

- Replaced per-file module-level import stubs in `tests/test_voice_cloner.py`,
  `tests/test_tts_factory.py`, and `tests/test_engine_controls.py` with a single
  meta-path finder in `tests/conftest.py` that stubs heavyweight optional
  dependencies (`torch`, `transformers`, `TTS`, `chatterbox`, `mlx_audio`) only
  when genuinely absent; removes order-dependent mock leakage (the
  `pop("tts_factory")` workaround) so every test file passes individually (#66)

## v0.2.0 — 2026-03-18

First tagged release of VoiceCast (formerly VoiceCloner).

### Features

- Voice cloning from 5–30 second audio samples with Coqui XTTS v2
- Multi-engine architecture with Chatterbox TTS support (Turbo and Standard)
- Expressive speech with paralinguistic tags (`[laugh]`, `[sigh]`, `[gasp]`, etc.)
- 16-language multilingual support via Coqui XTTS v2
- Desktop GUI application with threaded audio generation
- Command-line interface for batch processing
- Python API for programmatic integration
- MLX Audio backend for Apple Silicon hardware acceleration

### Infrastructure

- CI/CD pipeline with pre-commit hooks and GitHub Actions
- Ruff linting and formatting, Bandit security scanning
- Comprehensive documentation: API reference, CLI reference, GUI guide, engines guide, architecture, development, and troubleshooting
- Landing-page-style README for better project visibility

### Project

- Renamed from VoiceCloner to VoiceCast
- MIT licensed, fully open source

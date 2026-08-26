# Changelog

## Unreleased

### Fixed

- Chatterbox variant selection is now honored end-to-end: the engine loads
  `ChatterboxTurboTTS` from `ResembleAI/chatterbox-turbo` for Turbo and
  `ChatterboxTTS` from `ResembleAI/chatterbox` for Standard (previously both
  variants loaded the same backend checkpoint regardless of the selection),
  the downloader pins distinct repos and revisions per variant, and registry
  install status no longer reports Turbo installed when only the Standard
  cache exists (#56)

- The `vcloner.py` CLI now exits with status 1 on every error path (missing
  required arguments, model-group passed as a generation engine, unusable
  `--download-models` invocations, and the `ModelNotInstalledError`,
  `FileNotFoundError`, `ImportError`, and catch-all exception handlers) so
  scripts and CI can detect failures instead of observing exit code 0 (#58)

- `VoiceCloner` fails fast at construction with `ValueError` when an engine
  requires reference audio but `speaker_wav` is empty or `None`, instead of
  deferring the failure into `engine.generate()`; engine instances are now
  also honored via their `requires_reference_audio` attribute (#58)

- All production registry access is routed through the documented
  `get_registry()` singleton (`vcloner.py` list/download helpers and the GUI
  Model Manager previously constructed private `ModelRegistry()` instances,
  so listings could disagree with the main window's install state); an AST
  guard test pins `ModelRegistry()` construction to its definition module (#59)

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

- Added `docs/decisions/tts-fork.md`: evidence-backed adopt decision for the
  maintained `idiap/coqui-ai-TTS` fork (`coqui-tts` on PyPI) of the unmaintained
  Coqui TTS package, with API-diff verification against the engine, downloader,
  and factory call sites, a license check (MPL-2.0 code / CPML XTTS v2 weights),
  and a follow-up migration outline; no code migration in this change (#64)
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

# Changelog

## Unreleased

### Added

- `utils/text_chunker.split_into_chunks()`: split long text into
  sentence-boundary-respecting chunks up to a configurable character
  limit, for future use chunking text before TTS synthesis (#125).
- `VoiceCloner.say()` and `VoiceCloner.generate()`: optional `chunk_size`
  and `silence_duration` parameters. When `chunk_size` is set and the text is
  longer than it, the text is split on sentence boundaries, synthesized chunk
  by chunk, and the chunks are concatenated with `silence_duration`
  milliseconds of silence between them (default 200 ms, between chunks only,
  no leading or trailing pad). Invalid chunking values are rejected. Shorter
  texts are still synthesized in a single engine call with the original
  unmodified text. `say()` returns the path to the written WAV file when
  `save_audio=True`, and `None` otherwise; `generate()` always saves and
  returns the final WAV path (#127, #128, #129).
- Engine-specific `MAX_CHUNK_CHARS` defaults are now applied automatically by
  `VoiceCloner` and can be overridden per synthesis call with `chunk_size`;
  the effective limit is propagated through every engine's `generate()` API
  (#130, #131, #132).

### Fixed

- Audio8: implement the real four-session ArkTTS pipeline (codec encoder,
  slow AR, fast AR, codec decoder) with per-token KV-cache stepping, so
  `--engine audio8-onnx` actually synthesizes 44.1 kHz speech from the
  `Audio8/audio8-TTS-0.1B-ONNX-INT8` model instead of failing to feed the
  model's stateful input interface.
- Audio8: adopt the model tokenizer's existing `<|pad|>` token (id 0) when it
  ships without a configured `pad_token`, so generation no longer fails on the
  `padding=True` tokenizer call.
- CLI: allow `--download-models --engine audio8` (add the `audio8` group alias
  to the engine-group choices alongside `chatterbox` and `mlx-audio`).
- Dependency fixes: Audio8 now installs its Transformers tokenizer and Hugging
  Face downloader dependencies, while Coqui is capped at PyTorch 2.8 to avoid
  the coqui-tts torchcodec import path.
- Coqui checkpoint loading now overrides PyTorch's `weights_only` default only
  inside the scoped model-construction compatibility context; GUI completion
  feedback includes the generated path and exposes the current stage to
  assistive technology.
- GUI shutdown now remains responsive while generation finishes cooperatively,
  cancellation cleans only unique per-run output files, and the quickstart
  installs the default Coqui engine extra with separate-environment guidance.

### Documentation

- Document the Audio8 engine, model download, and CLI usage in
  `docs/engines.md` and `docs/model-management.md`.
# Changelog

## v0.3.0 — 2025-06-22

### Added

- Audio8 TTS model support via the new `Audio8TTS` engine descriptor (#110)
- GUI Generate button is now gated until all inputs (text, source, target) are valid (#105)
- Explicit model management via CLI (`vcloner.py --download-models`) and GUI Model Manager (#6)
- Chatterbox TTS multi-engine architecture with Turbo and Standard variants (#2)

### Fixed

- Generation UX: persist save filename, stage text/speed, and temp copy (#107)
- Safely terminate the generation thread on window close to avoid hangs (#106)
- Wire Chatterbox model variant to its checkpoint end-to-end (#88)
- Signal CLI failures via exit codes; fail fast on missing reference audio (#87)
- Stream model download progress and bound close waits (#86)
- Derive Coqui cache directory from backend defaults (#85)
- Fix install command and remove stale `main.py` references (#109)
- Replace invalid engine-controls fallback (#39)
- Make CI fail on unit-test failures (#38)

### Changed

- Single `list_models()` sweep per refresh instead of repeated calls (#104)
- Cache cloner and engine per engine name at app scope (#103)
- Extract collaborators from `VoiceCloningApp` (#101)
- Clean up `engine-controls` access paths (#100)
- Resolve stale root-level files (#98)
- Replace model registry singleton with injected registry (#43)
- Descriptor-based engine controls and MLX voice dedup (#42)
- Introduce `EngineDescriptor` and explicit engine bootstrap (#41)
- Untrack 66 MB sample media from git (#99)

### Documentation

- Re-capture screenshots for the current VoiceCast UI (#108)
- Add `CODE_REVIEW.md` with clean and cleanup audit results (#102)
- Document Python matrix decision and EOL revisit trigger (#92)
- Add TTS fork adopt/defer spike decision (#89)
- Add agent environment and config docs (`AGENTS.md`, `CLAUDE.md`) (#81)

### Infrastructure

- Bump `actions/setup-python` from v5 to v7 (#95)
- Bump `actions/upload-artifact` from v4 to v7 (#94)
- Bump `actions/checkout` from v4 to v7 (#93)
- Make security scan real: refresh ruff, declare `transformers` (#91)
- Pin dev tooling and prove lockfile bootstrap (#83)
- Add pinned `requirements.txt` lockfile (#82)

### Testing

- Add characterization tests at component seams (#97)
- Add coverage measurement to the CI test job (#96)
- Isolate module mocking behind `conftest` import stubs (#84)
- Add GUI import/offscreen smoke test (#90)

## Unreleased

### Added

- Engine dependencies are now explicit, mutually exclusive extras: the
  maintained Coqui TTS fork with Transformers 5.16.x, Chatterbox 0.1.7 with
  Transformers 5.2.0, or Audio8 with Transformers 4.x. The shared universal
  lock intentionally contains only the core and dev dependencies (#121)
- Dependency floors refreshed toward current stable for soundfile (`0.14.0`),
  PySide6 (`6.11.0`), rich (`15.0.0`), mlx (`0.32.0`), and numpy (`1.26.0`);
  ruff CI pin and dev-extra cap moved to `>=0.16.0,<0.17.0`, aligned with the
  pre-commit rev `v0.16.4`; lockfile regenerated (#53)

### Changed

- Coqui, Chatterbox, and Audio8 are no longer installed in the same environment
  because their resolver constraints require incompatible Transformers
  versions; installation guidance now documents separate engine environments
  (#121)
- The CI security job can now fail: bandit runs blocking (no `|| true`) and
  `safety check` is replaced by a pip-audit scan of the fully pinned lockfile
  via `scripts/pip_audit_lockfile.sh` (`--no-deps --disable-pip`; no package
  installs). The 43 advisories known at establishment time (2026-08-26:
  torch/onnx/starlette/diffusers/gradio/transformers) are explicit documented
  `--ignore-vuln` entries in that wrapper — any advisory outside the baseline
  fails the job (#52)

### Fixed

- Chatterbox model-missing errors now remain deterministic even when a local
  Hugging Face cache contains another variant, and MLX engine controls no
  longer pass engine-only `variant` values to QWidget (#121)
- The pre-commit dependency-vulnerability hook now executes when invoked: the
  skipped/unmaintained safety hook was replaced by a local `pip-audit-lockfile`
  hook targeting `requirements.txt` through the same shared wrapper (network
  access to advisory DBs required), and the CI pre-commit job no longer sets
  `SKIP` for it (#52)

- GUI import/offscreen smoke test (`tests/test_gui_smoke.py`) that imports the
  `voice_cloning_app` entry point and constructs the main window headlessly;
  the CI test job now installs PySide6 and pygame (with `libegl1`/`libgl1`
  system libraries and an explicit `QT_QPA_PLATFORM=offscreen`) so the test
  exercises the real Qt chain in CI instead of skipping, plus `tqdm` for the
  HF download progress bridge that installing PySide6 un-skips in
  `tests/test_download_streaming.py` (#51)

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

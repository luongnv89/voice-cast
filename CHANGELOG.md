# Changelog

## Unreleased

### Documentation

- Added `docs/agent-environment.md` with agent-facing setup and run notes (#46)
- Added `CLAUDE.md` and `AGENTS.md` at the repo root pointing to the environment notes (#47, #48)

### Infrastructure

- Added fully pinned `requirements.txt` (universal uv lock of `pyproject.toml`
  runtime deps) so the CI security job's `safety check` and the pre-commit
  safety hook resolve; documented regeneration in `docs/agent-environment.md`
  (#49)
- CI pre-commit job now sets `SKIP=python-safety-dependencies-check`, matching
  the skip already declared in `.pre-commit-config.yaml` (`ci.skip` is only
  honored by pre-commit.ci, so the hook otherwise runs on GitHub Actions once
  `requirements.txt` exists); dependency scanning remains in the non-blocking
  Security Scan job

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

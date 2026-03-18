# Changelog

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

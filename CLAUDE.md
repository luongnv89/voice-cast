# CLAUDE.md

VoiceCast: Python voice-cloning / TTS app (Coqui XTTS v2 + Chatterbox engines,
PySide6 GUI, CLI). Python 3.10–3.12 only (CI matrix; newer local interpreters
are untested).

Environment setup, system prerequisites, and env vars:
@docs/agent-environment.md

## Critical Commands

```bash
# Tests (command of record)
pytest tests/ -v --tb=short

# Lint + format check (must pass before commit)
ruff check .
ruff format --check .

# Security scan
bandit -c pyproject.toml -r . --exclude tests

# Pre-commit hooks
pre-commit run --all-files

# Minimal test install (no ML stack needed — tests mock torch/TTS)
pip install sounddevice soundfile rich numpy && pip install -e . --no-deps
```

## Architecture Map

- `voice_cloner.py` — core VoiceCloner API; `vcloner.py` — CLI entry point
- `tts_engine_base.py` / `tts_factory.py` — engine interface, registry, bootstrap
- `engines/` — Coqui XTTS v2 and Chatterbox implementations
- `gui/` — PySide6 app (`voice_cloning_app.py`) and dynamic engine controls
- `models/` — model registry/downloader; explicit downloads only
- `utils/`, `tests/`, `docs/` — helpers, pytest suite, documentation

## Hard Rules

- IMPORTANT: Test command of record is `pytest tests/ -v --tb=short`. Run it
  before declaring any change done.
- Never commit `.env` files, API keys, or model/audio binaries.
- Never make model downloads implicit — downloads are explicit user actions.
- Do not add dependencies without updating `pyproject.toml`.
- Python compatibility floor is 3.10 — no 3.11+-only syntax.
- Keep changes minimal for small fixes; do not reformat untouched code.

## Workflow Preferences

- Branches: `docs/`, `fix/`, `feat/`, `refactor/<n>-<desc>`; commits follow
  conventional commits (`type(scope): description (#N)`).
- Lint and format check must be clean before you commit.
- Update `CHANGELOG.md` and relevant `docs/*.md` when behavior changes.

## Token Efficiency

- Never re-read files you just wrote or edited. You know the contents.
- Never re-run commands to "verify" unless the outcome was uncertain.
- Don't echo back large blocks of code or file contents unless asked.
- Batch related edits into single operations. Don't make 5 edits when 1 handles it.
- Skip confirmations like "I'll continue..." Just do it.
- If a task needs 1 tool call, don't use 3. Plan before acting.
- Do not summarize what you just did unless the result is ambiguous or you need additional input.

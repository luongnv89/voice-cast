# AGENTS.md

Agent-facing conventions for working in this repository. Any coding agent
(Claude, Codex, Cursor, etc.) should follow this file. It deliberately does not
duplicate the commands of record — they live in the two files below.

- Build/test/lint commands and environment notes: `docs/agent-environment.md`
- Claude-specific project config: `CLAUDE.md`

## Project

VoiceCast — Python voice-cloning / TTS app (Coqui XTTS v2 + Chatterbox engines,
PySide6 GUI, CLI). Python compatibility floor is **3.10** (CI tests 3.10–3.12);
do not use 3.11+-only syntax.

## Conventions

- Branches: `feat/`, `fix/`, `docs/`, `refactor/<n>-<desc>` — lowercase,
  hyphen-separated, include the issue number.
- Commits: conventional commits — `type(scope): description (#N)`, imperative
  mood, first line under 72 characters.
- Tests: run `pytest tests/ -v --tb=short` before declaring any change done.
- Style: ruff enforces lint + format (`ruff check .`,
  `ruff format --check .`); both must be clean before you commit.

## Hard Rules

- IMPORTANT: Never make model downloads implicit — downloads are explicit user
  actions only (`vcloner.py --download-models`, GUI Model Manager).
- Never commit `.env` files, API keys, or model/audio binaries.
- Do not add dependencies without updating `pyproject.toml`.
- Keep changes minimal for small fixes; do not reformat untouched code.
- Update `CHANGELOG.md` and relevant `docs/*.md` when behavior changes.
- The test suite mocks the ML stack — do not require torch/TTS installs to run
  tests.

## Token Efficiency

- Never re-read files you just wrote or edited. You know the contents.
- Never re-run commands to "verify" unless the outcome was uncertain.
- Don't echo back large blocks of code or file contents unless asked.
- Batch related edits into single operations. Don't make 5 edits when 1 handles it.
- Skip confirmations like "I'll continue..." Just do it.
- If a task needs 1 tool call, don't use 3. Plan before acting.
- Do not summarize what you just did unless the result is ambiguous or you need additional input.

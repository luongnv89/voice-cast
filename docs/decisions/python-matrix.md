# Decision: Python Version Matrix (Defer 3.13/3.14)

- **Status:** Accepted (defer — keep 3.10–3.12 matrix; revisit on trigger)
- **Date:** 2026-08-26
- **Issue:** #60 (closes F-DEP-007; part of epic #45, MODERNIZATION_PLAN.md Task 3.1, Phase P2 — Modernize)
- **Scope:** Decision record only. **No CI matrix change in this task.**
- **Blocks:** Task 3.4 (setup-python pins interpreters) — this decision must land first.

## Context

At audit time the declared and tested Python support was split three ways:

| Source | Value |
|---|---|
| `pyproject.toml:7` `requires-python` | `>=3.10` (floor 3.10) |
| `.github/workflows/ci.yml:52` test matrix | `["3.10", "3.11", "3.12"]` |
| `pyproject.toml:49` Ruff `target-version` | `py310` |
| `docs/agent-environment.md:12` | Documents 3.10–3.12 as the supported range |
| Local dev interpreter (example) | 3.14.7 — **untested** by CI |

CPython 3.10 reaches **end-of-life 2026-10-04** (PEP 619). Its approaching EOL
forces a revisit, but the local 3.14 interpreter is not yet covered by CI and
several of our heavy dependencies have historically lagged on new CPython
releases (notably `TTS`/`coqui-tts` and the CUDA `torch` wheels). A deliberate
choice between "extend now" and "defer with a trigger" is required before
Task 3.4 (`setup-python`) pins the interpreter list for contributors and CI.

## Current Matrix (unchanged in this task)

```yaml
# .github/workflows/ci.yml:52
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
```

This file is intentionally left at 3.10–3.12 by this decision. No change to
`.github/workflows/ci.yml`, `pyproject.toml`, or `docs/agent-environment.md`
is made here.

## Options Considered

| # | Option | Pros | Cons |
|---|---|---|---|
| A | **Extend now** — add 3.13 (and/or 3.14) to the CI matrix | Proves newer interpreters eagerly; catches incompatibilities early | Expands CI cost; 3.13/3.14 ecosystem gaps not yet triaged (e.g. `TTS` fork only recently added 3.14 wheels, lockfile has no `<3.13` universal split); heavier validation scope than a P2 decision task warrants |
| B | **Defer with explicit trigger** — keep 3.10–3.12, record an EOL-based revisit | Minimal diff, preserves green baseline, lands before Task 3.4 as required; makes the revisit obligation visible and dated | 3.13/3.14 stay untested by CI until the trigger fires |

## Decision

**Choose Option B — Defer.**

Keep the CI matrix at `["3.10", "3.11", "3.12"]` and `requires-python >=3.10`
until a revisit trigger fires (see below). This is the simpler deferral
approach called out in the issue: it satisfies F-DEP-007 with a single
decision artifact and no CI change, unblocking Task 3.4 which will pin the
interpreters from this recorded baseline.

### Rationale

1. **Precondition for Task 3.4.** `setup-python` needs a frozen matrix to pin.
   A doc-only decision unblocks it without introducing new CI variables to
   debug in the same change.
2. **3.10 is still supported today.** Until 2026-10-04, dropping or adding
   versions is churn; the cost/benefit tips after EOL.
3. **Ecosystem readiness.** Python 3.13 had significant C-API changes
   (PEP 703-adjacent, `immortal` objects, etc.) and downstream wheels lagged;
   3.14 is even newer (released 2025-10-07). Deferring avoids pinning CI to
   interpreters our lockfile and engine deps have not yet proven.
4. **Baseline-green preserved.** The existing matrix is known-green; no extra
   CI minutes or flake risk is introduced.

## Revisit Triggers

Re-open this decision (and file a follow-up issue) when **any** of these fires
— whichever comes first:

- **Primary — CPython 3.10 EOL: 2026-10-04.** On or shortly before this date:
  drop 3.10 from the matrix and `requires-python`, add 3.13 (and evaluate
  3.14), and bump `tool.ruff.target-version` accordingly. This is the
  issue-mandated trigger.
- **Dependency readiness.** `coqui-tts`, `torch`, `PySide6`, and the lockfile
  each publish stable 3.13/3.14 wheels and the universal lock resolves without
  the current `python_version < '3.13'` split. When all four are true, an
  early extension to 3.13 may be proposed even before EOL.
- **Security or CI failure on 3.10.** Any CVE or CI breakage isolated to 3.10
  accelerates the drop.
- **Calendar backstop — 2026-11-01.** Even if no other trigger fires, revisit
  no later than one month post-EOL to avoid shipping on an EOL interpreter.

## Follow-Up When Trigger Fires

1. Update `.github/workflows/ci.yml:52` matrix — remove `3.10`, add `3.13`
   (and `3.14` if wheels/lockfile allow).
2. Bump `pyproject.toml:7` `requires-python` to `>=3.11` (or `>=3.12` if
   dropping two versions).
3. Bump `pyproject.toml:49` `tool.ruff.target-version` from `py310` to match
   the new floor.
4. Update `docs/agent-environment.md` toolchain table and all prose that
   lists 3.10–3.12.
5. Regenerate `requirements.txt` with `uv pip compile --universal --extra dev`
   on the new floor and re-prove `pytest tests/ -v --tb=short` across the new
   matrix.
6. Update `CHANGELOG.md`.

## Non-Goals

- No change to CI, `pyproject.toml`, or docs prose in this task.
- No commitment to a specific post-EOL floor beyond "≥3.11"; the follow-up
  task will choose between 3.11 and 3.12 based on ecosystem state at that time.
- No 3.13/3.14 validation is claimed — local 3.14.x interpreters remain
  explicitly untested until added to CI.

## Sources (retrieved 2026-08-26)

- `pyproject.toml:7` and `pyproject.toml:49` — floor and Ruff target.
- `.github/workflows/ci.yml:52` — current matrix `["3.10","3.11","3.12"]`.
- PEP 619 — Python 3.10 release schedule — EOL 2026-10-04 (5-year support window).
- `docs/agent-environment.md:12` — CI-supported range documentation.
- MODERNIZATION_PLAN.md Task 3.1 (P2) — original task text quoted in #60 body.
- `docs/decisions/tts-fork.md` — prior P2 decision-doc precedent (EOL/cadence
  trigger pattern reused here).

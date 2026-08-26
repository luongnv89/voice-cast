# Decision: Coqui TTS Fork Adoption Spike

- **Status:** Accepted (adopt — execution deferred to a follow-up migration task)
- **Date:** 2026-08-26
- **Issue:** #64 (closes F-DEP-002; part of epic #45, MODERNIZATION_PLAN.md task 3.5)
- **Scope:** Decision document only. **No code migration in this task.**

## Context

VoiceCast depends on Coqui's `TTS` package (`pyproject.toml`: `"TTS>=0.22.0"`) for its
XTTS v2 cloning engine. The original project shipped its final release,
v0.22.0, on 2023-12-12 and is unmaintained — the maintained community fork's own
FAQ states plainly that "the original Coqui package had its last release in
December 2023" and recommends the fork "for compatibility with recent Python
and dependency versions." At audit time (F-DEP-002) the fork recommendation was
directional only: no evidence-backed evaluation existed against our actual call
sites. This spike produces that evaluation and an adopt/defer decision.

## Candidate Forks Evaluated

| Criterion | Original — `coqui-ai/TTS` (PyPI `TTS`) | Fork — `idiap/coqui-ai-TTS` (PyPI `coqui-tts`) |
|---|---|---|
| Latest release | 0.22.0 (2023-12-12) | 0.27.5 (2026-01-26) |
| Release cadence | Dead (~21 months silent at audit) | Active: ~20 releases Apr 2024 – Jan 2026 |
| Python support | ≤3.11 era; breaks on newer stacks | `>=3.10,<3.15` (covers our 3.10 floor and CI 3.10–3.12) |
| PyTorch | Pinned-era deps; breaks on modern torch | Supports PyTorch 2.10 and Python 3.14 (since 0.27.4) |
| Wheels | Source builds common | Prebuilt Linux/macOS/Windows wheels (since 0.24.2) |
| XTTS v2 | Yes | Yes — retained, plus voice-clone caching (0.27.0) |
| License (code) | MPL-2.0 | MPL-2.0 |
| Maintainer | Coqui (company defunct) | Idiap Research Institute + community |

No other fork is a credible candidate: other forks of `coqui-ai/TTS` are
personal mirrors without release cadence or institutional backing; the Idiap
fork is the one the ecosystem, documentation, and downstream users converged on.

## API-Diff Notes Against Our Call Sites

Verified against the fork source at tag `v0.27.5` (`TTS/utils/manage.py`) and
the fork's published docs (August 2026). The fork deliberately keeps the `TTS`
import namespace, so all three production call sites compile unchanged.

| # | VoiceCast call site | Usage | v0.27.5 status |
|---|---|---|---|
| 1 | `engines/coqui_engine.py:92-95` | `from TTS.api import TTS`; `TTS(model_name=…, progress_bar=True, gpu=(device=="cuda"))` | Unchanged — import path kept identical by design |
| 2 | `engines/coqui_engine.py:125-132` | `tts_to_file(text=…, speaker_wav=…, file_path=…, language=…, gpt_cond_len=…, temperature=…)` | Unchanged — signature preserved incl. XTTS-specific kwargs |
| 3 | `models/downloaders/coqui_downloader.py:43-44` | `from TTS.utils import manage`; `ModelManager(output_prefix=…, progress_bar=…)` | Compatible — constructor is `(models_file=None, output_prefix=None, progress_bar=False)`; both kwargs valid (our `TypeError` fallback chain becomes dead but harmless) |
| 4 | `models/downloaders/coqui_downloader.py:55-69` | Patches module-global `manage.tqdm` under `_TQDM_PATCH_LOCK` to forward transfer progress | Intact — `from tqdm import tqdm` remains a module-level symbol; downloads still stream via `iter_content(1024)`, matching `_TQDM_BLOCK_SIZE = 1024` |
| 5 | `models/downloaders/coqui_downloader.py:118-137` | Unpacks `download_model(model_name)` as a 3-tuple; wraps result in `Path(...).parent` | Compatible — still returns `(model_path, config_path, model_item)`; paths are now `Path` objects instead of `str`, which `Path()` accepts either way |
| 6 | `tts_factory.py:27-28` | `import TTS` availability probe for the `coqui` engine descriptor | Unchanged — top-level package name stays `TTS` |
| 7 | `pyproject.toml:11` | Dependency declaration `"TTS>=0.22.0"` | **Must change** — PyPI distribution renames to `coqui-tts` (import name does not) |
| 8 | `tests/test_download_streaming.py:61` | Test mock of `TTS.utils.manage` | Test-only; re-validate mocks after swap |

### Behavioral differences found (not blockers)

1. **torch no longer auto-installed (from 0.27.4):** the fork stopped pulling
   `torch`/`torchaudio` by default. Our migration must declare torch explicitly
   (direct dependency or fork extra).
2. **XTTS v2 terms-of-service gate is unchanged:** first download of XTTS v2
   requires agreeing to CPML — interactively via `input()` prompt, headlessly via
   `COQUI_TOS_AGREED=1`. Our downloader drives `ModelManager` non-interactively,
   so the migration must surface the agreement explicitly to the user rather
   than silently setting the env var (downloads stay explicit user actions per
   repository policy).
3. **Cache-path default:** with `output_prefix=None` the fork resolves the
   default cache root through `trainer.io.get_user_data_dir("tts")`; our
   registry derives the Coqui cache dir from backend defaults (#85). The
   migration must re-prove that invariant.
4. **Heavier dependency tree:** adds `fsspec`, `trainer`, `typing_extensions`,
   among others — acceptable; wheels avoid build friction.

## License Check

- **Code (MPL-2.0, both original and fork):** compatible with VoiceCast's MIT
  license. MPL-2.0 is file-level weak copyleft that applies to modifications of
  the library's own files; using it unmodified as a pip dependency imposes no
  obligations on our source.
- **XTTS v2 model weights (Coqui Public Model License, non-commercial):
  unchanged by the fork switch.** The weights keep their CPML terms regardless
  of which code loads them; commercial deployment of XTTS v2 output requires a
  Coqui commercial license. This constraint exists today and is orthogonal to
  the adopt decision.

## Decision

**ADOPT** `idiap/coqui-ai-TTS` (PyPI `coqui-tts`) as the Coqui backend
distribution, executed as a follow-up migration task — not in this spike.
Rationale: every axis that made the original a liability (dead cadence, broken
modern-Python/torch compatibility) is actively fixed in the fork; all three
production call sites are verified drop-in compatible against the current
release; and the code license is identical. Deferring would keep us pinned to a
package that cannot receive security or compatibility fixes.

## Follow-Up Migration Outline (adopt path)

1. Swap `pyproject.toml` dependency `TTS>=0.22.0` → `coqui-tts>=0.27.5` plus an
   explicit `torch` requirement matched to CI; refresh `requirements.txt`.
2. Re-prove the registry cache-dir invariant (#85) against the fork's
   `get_user_data_dir("tts")` default when `output_prefix` is unset.
3. Make the CPML agreement explicit in the download UX (prompt/GUI checkbox
   feeding `COQUI_TOS_AGREED=1`) without making any download implicit.
4. Update `tests/test_download_streaming.py` mocks and any import-stub fixtures
   for the renamed distribution.
5. Update user-facing docs (`README`, `docs/engines.md`,
   `docs/model-management.md`) and `CHANGELOG.md`.
6. Smoke-test on a real install: engine generation, download-progress
   forwarding through the patched `tqdm` seam, and cancel behavior.

## Revisit Triggers (post-adoption)

Even though we adopt, re-open this decision if any of these fire:

- No fork release for **> 18 months** (current cadence is roughly quarterly;
  last: 0.27.5 on 2026-01-26).
- XTTS v2 support is removed or broken upstream without a replacement path.
- The fork's code license changes away from MPL-2.0.

## Sources (retrieved August 2026)

- PyPI `TTS` version history — last upload 0.22.0 era, Dec 2023.
- PyPI `coqui-tts` — 0.27.5 metadata (license, Python `<3.15,>=3.10`, history).
- GitHub `idiap/coqui-ai-TTS` releases v0.24.2 … v0.27.5 (wheels, torch 2.10 /
  Python 3.14, torch-no-longer-default notes).
- Fork FAQ — statement that the original is unmaintained.
- Fork source `TTS/utils/manage.py` @ tag `v0.27.5` (call-site verification in
  the table above).

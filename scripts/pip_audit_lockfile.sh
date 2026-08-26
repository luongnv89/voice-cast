#!/usr/bin/env bash
# pip-audit advisory baseline for the compiled lockfile (requirements.txt).
#
# The scan below IS the advisory baseline (issue #52): every known advisory at
# establishment time (2026-08-26) is listed as an explicit --ignore-vuln so
# that any NEW advisory fails the run. Each entry must be removed by upgrading
# the affected package past its fixed version, never by silently re-baselining.
# Review this list whenever dependencies are regenerated.

set -euo pipefail

cd "$(dirname "$0")/.."

IGNORED_VULN_IDS=(
    # diffusers 0.29.0
    PYSEC-2026-40
    PYSEC-2026-41
    PYSEC-2026-2446
    # gradio 6.8.0
    PYSEC-2026-211
    PYSEC-2026-2178
    PYSEC-2026-2179
    # onnx 1.19.0
    GHSA-q56x-g2fj-4rj6
    PYSEC-2026-103
    PYSEC-2026-104
    PYSEC-2026-2239
    PYSEC-2026-2240
    PYSEC-2026-2241
    PYSEC-2026-2689
    PYSEC-2026-3587
    # starlette 0.52.1
    PYSEC-2026-161
    PYSEC-2026-248
    PYSEC-2026-249
    PYSEC-2026-2280
    PYSEC-2026-2281
    # torch 2.6.0
    CVE-2025-2148
    CVE-2025-2149
    CVE-2025-2998
    CVE-2025-2999
    CVE-2025-3001
    PYSEC-2025-191
    PYSEC-2025-194
    PYSEC-2025-198
    PYSEC-2025-199
    PYSEC-2025-200
    PYSEC-2025-201
    PYSEC-2025-202
    PYSEC-2025-203
    PYSEC-2025-204
    PYSEC-2025-205
    PYSEC-2025-206
    PYSEC-2025-207
    PYSEC-2025-208
    PYSEC-2025-209
    PYSEC-2026-139
    PYSEC-2026-1970
    PYSEC-2026-2286
    # transformers 5.2.0
    PYSEC-2026-2289
    PYSEC-2026-2290
)

IGNORE_ARGS=()
for id in "${IGNORED_VULN_IDS[@]}"; do
    IGNORE_ARGS+=("--ignore-vuln" "$id")
done

exec pip-audit \
    --requirement requirements.txt \
    --no-deps \
    --disable-pip \
    --strict \
    --progress-spinner off \
    "${IGNORE_ARGS[@]}"

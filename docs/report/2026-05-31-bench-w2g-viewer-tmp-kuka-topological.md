# BENCH-W2G Viewer Temp Path And KUKA Topological Check

## Scope

- Changed `scripts/bench-viewer.sh` to write benchmark JSON to `/tmp` instead
  of macOS `$TMPDIR`.
- Verified `kuka_1` uses the plain chessboard path, which now dispatches to
  `DetectorParams::Topological`.

## Files Changed

- `scripts/bench-viewer.sh`
- `docs/backlog.md`
- `docs/report/2026-05-31-bench-w2g-viewer-tmp-kuka-topological.md`

## Validation Run

- PASS: `bash -n scripts/bench-viewer.sh`
- PASS: `calib-bench run --dataset kuka_1`

## Findings

- macOS `$TMPDIR` resolves under `/var/folders/...`; Vite rejects `/@fs` reads
  there with `403 Forbidden` because the viewer allowlist only includes the
  workspace and `/tmp`.
- `kuka_1` now reports hand-eye `0.13594 px` versus `0.13472 px` intrinsic
  floor, with 30 / 30 images used and 14,280 / 14,280 features detected.
- Robot correction for `kuka_1` is `0.456 / 1.223 mm` translation mean/max and
  `0.067 / 0.141 deg` rotation mean/max.

## Follow-Ups / Remaining Risks

- Detection still dominates `kuka_1` runtime at about `18.2 s`; cache
  detections before running repeated hand-eye diagnostics.

# W3 — stereo_charuco oracle closure

Status: **closed** (verified 2026-05-31). Both cross-checks pass.

## What the oracle is

`data/stereo_charuco/calibration.json` is an *external* reference calibration of
the same two-camera ChArUco scene, produced by a different toolchain. It is **not**
fully parameter-comparable:

- Its distortion vector is `[0.96, -300.8, 0, 0, -1.37]` (k2 ≈ −300), a different
  model from the Brown–Conrady the bench fits — so distortion/intrinsic
  coefficients are not directly comparable.
- Its translations are in **millimetres**; the bench reports **metres**.
- It declares `frame_cols=2048, frame_rows=1536`.

So the honest cross-checks are the reprojection band and the recovered baseline
magnitude, not raw coefficients.

## Cross-check 1 — reprojection band (PASS)

Bench run: `calib-bench run --dataset stereo_charuco` (clean, exit 0).
`reproj_report.headline_px == fit.reported_mean_reproj_px` (the regression
invariant holds).

| quantity                            | bench    | oracle                |
| ----------------------------------- | -------- | --------------------- |
| headline reprojection               | 0.5509 px | —                     |
| intrinsic floor (mean)              | 0.4016 px | cam0 0.4960 / cam1 0.4964 |
| rig-extrinsic / joint solve (mean)  | 0.5509 px | cam1 extrinsic 0.6290 |
| per-camera (intrinsic floor)        | 0.3985 / 0.4049 px | —          |
| per-camera (joint solve)            | 0.5820 / 0.5164 px | —          |

The bench's joint-solve reprojection (**0.5509 px**) sits between the oracle's
per-camera intrinsic level (≈0.496 px) and its camera1 extrinsic residual
(0.629 px) — exactly where a joint stereo solve should land, and **better** than
the oracle's own extrinsic number. The bench's intrinsic *floor* (0.402 px, free
per-view PnP through the calibrated model) is tighter still, as expected for a
per-view-free pose. The example's Step-2 per-camera intrinsic reprojection
(0.399 / 0.405 px) corroborates the analysis-module floor.

## Cross-check 2 — recovered baseline magnitude (PASS, 0.80 mm)

The recovered camera0→camera1 relative pose is carried by
`RigExtrinsicsExport.cam_se3_rig` but is not (yet) copied into `BenchRecord`
(`Fit` holds only reprojection stats). The `stereo_charuco_session` example
prints its **magnitude** (only — not the components):

```
--- Step 3: Rig Extrinsics Initialization ---
  Initial baseline: |t(cam1->rig)| = 0.1149 m (114.91 mm)
--- Step 4: Rig Bundle Adjustment ---
  Rig BA mean reprojection error: 0.5509 px
    Camera 0: 0.5820 px
    Camera 1: 0.5164 px
  Rig baseline (after BA): |t(cam1->rig)| = 0.1146 m (114.59 mm)
```

| quantity                 | bench (after BA) | oracle      | Δ        |
| ------------------------ | ---------------- | ----------- | -------- |
| baseline ‖t‖             | 114.59 mm        | 113.79 mm   | 0.80 mm  |

Oracle ‖t‖ = ‖[111.383, −3.927, 22.951] mm‖ = 113.79 mm. The bench recovers the
same baseline length to **0.80 mm (0.7 %)**, independently corroborating the rig
extrinsic beyond the reprojection number. Only the magnitude is compared here
because that is all the example exposes; a per-component / rotation check is
deferred to the follow-up below.

## Why there is no standalone CI test for this

The entire `data/` directory is gitignored (`.gitignore:6` = `data/stereo_charuco`,
and `git ls-files data` lists only a handful of non-ignored files) — so **no
benchmark dataset reaches CI**, not just the private ones. A data-driven oracle
test would always skip in CI. The regression-pinning assertion therefore belongs
in task **G** (the gate), which must either (a) commit image-free frozen fixtures
(`fixtures.rs::FrozenFixture` IR already exists for exactly this) or
(b) skip-if-data-absent. Proposed pins for G, derived from the verified numbers
above:

- headline reprojection within **[0.45, 0.75] px** (oracle intrinsic floor →
  oracle extrinsic + margin),
- recovered baseline ‖t‖ within **2 mm** of 113.79 mm (actual margin 0.80 mm).

## Follow-up worth doing (not blocking)

Surface the recovered rig extrinsic (`cam_se3_rig` → baseline vector + rotation,
per-camera-to-rig pose) into `BenchRecord` so the geometry cross-check is a
first-class bench metric (full translation + rotation, not just magnitude) rather
than an example-only printout. This is also the home for the cross-camera rig
diagnostics on the roadmap.

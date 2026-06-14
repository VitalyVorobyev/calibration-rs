# M-track: three additive distortion models (rational, thin-prism, division)

**Date:** 2026-06-14
**Scope:** M1 (rational k4–k6), M2 (thin-prism s1–s4), M3 (division model) — the
distortion-slot models gated on M0 (ADR 0020, factor-IR-as-data).

## What shipped

Three new distortion models, added **additively** at two layers:

1. **Core runtime model** (`vision-calibration-core`):
   - `RationalPolynomial<S>` — OpenCV rational, params `[k1,k2,k3,k4,k5,k6,p1,p2]`;
     `radial = (1+k1r²+k2r⁴+k3r⁶)/(1+k4r²+k5r⁴+k6r⁶)`, Brown-Conrady tangential;
     fixed-point inverse.
   - `ThinPrism<S>` — Brown-Conrady + `[s1,s2,s3,s4]` (`x+=s1r²+s2r⁴`,
     `y+=s3r²+s4r⁴`); fixed-point inverse.
   - `Division<S>` — Fitzgibbon single-parameter `lambda`; **closed-form** both
     directions (undistort `1/(1+λr²)`, distort via the quadratic root).
   - New `DistortionParams` serde variants (`rational` / `thin_prism` /
     `division`) and `AnyDistortion` arms, so `CameraParams::build()` produces a
     runtime camera with any of the new models (projection, undistort, export
     residual computation).

2. **Optim IR / backend** (`vision-calibration-optim`):
   - `DistortionKind::{Rational8, ThinPrism9, Division1}` (+ `dim()`).
   - `CameraModelDesc` constants `PINHOLE4_{RATIONAL8,THINPRISM9,DIVISION1}` and
     their `_SCHEIMPFLUG2` pairs.
   - ZST kernels `RationalKernel` / `ThinPrismKernel` / `DivisionKernel`
     (autodiff-generic over `T: RealField`), 6 new `dispatch_camera_model!`
     rows. Each model composes with the Scheimpflug sensor for free (sensor is a
     separate descriptor slot).

## Deliberate scope boundary

This is **additive only** — no problem builder, pipeline config, init, or export
path was touched, and the Brown-Conrady production paths (validated on rtv3d) are
byte-identical. The models are usable today via the runtime `CameraParams` /
`AnyDistortion` camera model and via hand-built `ProblemIR` using the new
`CameraModelDesc` constants.

**Not done (follow-up — see backlog `M-WIRE`):** wiring a user-facing distortion-
model *selection* through the pipeline configs (`PlanarIntrinsicsConfig`,
`ScheimpflugIntrinsicsConfig`, …) into the problem builders that currently
hardcode `CameraModelDesc::PINHOLE4_DIST5*`. That requires: generalizing the
BC5-hardwired `DistortionFixMask`, model-aware distortion init seeds, variable-
dim pack/unpack, and export reconstruction for the selected model — all of which
touch the validated calibration paths and were deliberately deferred out of an
unsupervised batch.

## Numerical notes (for the wiring follow-up)

- **Rational** has a numerator/denominator correlation (k1↔k4 partially trade
  off); the synthetic 3-view recovery lands ~2.4 % on k1/k2 (intrinsics still
  sub-0.5 px). It needs wider field coverage than BC5.
- **Division** has a gradient-zero degenerate point at `lambda = 0`; any solve
  must seed `lambda` with a rough non-zero prior (the integration test seeds
  −0.05).

## Tests

- Core: per-model zero-param identity + distort→undistort roundtrip on a 5×5
  normalized-coord grid (Division at 1e-9, fixed-point models at 1e-4).
- Optim: per-kernel zero-coefficient == `NoDistortionKernel`; synthetic
  ground-truth optimization recovering Rational and Division parameters
  (`tests/distortion_models.rs`).

## Gates

`cargo fmt --check`, `clippy --workspace --all-targets --all-features -D
warnings`, `cargo test --workspace --all-features`, and `RUSTDOCFLAGS=-D
warnings cargo doc --workspace --all-features --no-deps` all green. Schemas
unchanged (no JsonSchema-deriving config field added).

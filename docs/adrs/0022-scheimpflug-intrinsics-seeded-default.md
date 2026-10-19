# ADR 0022: Scheimpflug Intrinsics — User-Seeded Initialization is the Supported Default

- Status: Accepted
- Date: 2026-06-17

## Context

From-scratch Scheimpflug **intrinsics** calibration is ill-conditioned. The camera
model couples sensor tilt, focal length, and radial distortion (`pixel =
K(sensor_tilt(distortion(project(dir))))`), and these three trade off against one
another: the classic tilt-focal-distortion degeneracy, giving a long, flat valley
in the cost. Zhang's pinhole initialization ignores tilt, so on a tilted sensor it
underestimates the focal length, and the non-linear solve then settles into a wrong
tilt/focal basin. Tilt-aware linear initializers and rig-level auto-recovery are
basin-fragile on hard cameras, so a from-scratch warm-up is research-grade.

In practice the data needed to *seed* the solve is already on hand: the lens focal
length (a datasheet value) and the Scheimpflug **mount tilt** (a deliberately set
mechanical angle, the whole point of a Scheimpflug setup). Seeding both removes the
fragility.

Even a seeded solve has a trap: starting from `k1 = 0`, the fixed-tilt sub-problem
has a **spurious local minimum at `k1 ≈ 0`**. The per-view poses adapt to absorb the
radial distortion, and the gradient then pushes `k1` back toward zero. More
iterations do not escape it; a multi-start over `k1` does.

## Decision

**User-provided initialization is the supported default for Scheimpflug
intrinsics; from-scratch (unseeded) auto-init is experimental.**

The supported workflow seeds a coarse prior and lets bundle adjustment refine it:

```rust
let mut seed = ScheimpflugManualInit::default();
seed.intrinsics = Some(FxFyCxCySkew { fx: nominal, fy: nominal, cx, cy, skew: 0.0 });
seed.sensor     = Some(ScheimpflugParams { tilt_x: mount_angle, tilt_y: 0.0 });
step_init_with_seed(&mut session, seed, None)?; // distortion/poses auto
step_optimize(&mut session, None)?;
```

Both the focal **and** the tilt seed are load-bearing:

- A user-seeded sensor tilt sets `state.initial_sensor_manual`, which makes
  `step_optimize` **trust that tilt basin** instead of running the cold-start
  multi-start tilt sweep.
- The trusted-seed solve is two-phase: **Phase A** sweeps `k1` start points
  `{0, −0.20, −0.40}` with intrinsics *and tilt fixed* (so the degenerate
  high-tilt basin is inaccessible during the search), each preceded by a
  pose-only adaptation to the seeded `k1`; **Phase B** is a bounded joint refine
  with tilt held to `seed ± 0.10 rad`, focal to `[0.75, 1.5]×`, and the principal
  point free.
- The first pose is **not** fixed on the seeded path. Planar intrinsics has no
  global pose gauge to remove, and the homography-recovered seed pose is biased
  under strong distortion; pinning it there holds the solve off the optimum.

**Hard acceptance gate: mean reprojection must be ≤ 0.5 px per camera.** A
reprojection error > 0.5 px is never a success — the gate is not relaxed to make a
run "pass".

From-scratch `step_init` (no intrinsics seed) is retained for convenience but is
documented as experimental and emits a non-fatal warning in the session init log.

Seeded init is the **official acceptance route** for all datasets; seeds come
from the structured `DeviceSpec` layer (ADR 0023).

**Distortion model.** `ScheimpflugIntrinsicsConfig::distortion_model` (default
`DistortionKind::BrownConrady5`) selects the model `step_optimize` refines. The
Phase A `k1` multi-start grid generalizes through `with_leading_radial`, which
sweeps the **leading radial term** (`k1` for BrownConrady5/Rational8/ThinPrism9,
`lambda` for Division1) and is a no-op for `DistortionKind::None`. The
`fix_distortion` (`DistortionFixMask`) mask, five named bits `{k1, k2, k3, p1,
p2}`, is translated **by name** onto each model's packed layout by
`fix_mask_indices` (used for both user masks and the staging masks). For
BrownConrady5 it is `DistortionFixMask::to_indices()`. For the extended models
each extra coefficient follows the bit of its family: Rational8's `k4,k5,k6`
follow `k3`; ThinPrism9's `s1..s4` are fixed iff **both** `p1` and `p2` are
fixed; Division1's `lambda` follows the leading-radial `k1` bit. The staging
invariants (A0 all-fixed, A1 `{k1,k2}` free with `k3/p1/p2` fixed, default
`fix_k3`) hold across every model. Rigs are unaffected:
`SensorMode::Scheimpflug::distortion_model` remains BC5-typed and is rejected
up front for non-BC5 values in `validate_config` (ADR 0019).

**Convergence basin.** `calib-bench basin`
(`crates/vision-calibration-bench/src/basin.rs`) perturbs the ADR 0023
device-spec seed over independent focal / tilt / principal-point sweeps and
re-runs the seeded route per cell against each entry's acceptance threshold.
On the benchmark datasets the all-camera-pass basin is x[0.85, 1.30] on focal
and +-4 deg on tilt (the full tested sweep), containing the realistic
spec-error envelope (focal +-5 %, tilt +-2 deg) with 2-3x margin. Measured
tables: `docs/notes/scheimpflug-intrinsics.md`.

## Consequences

- **Deterministic, robust production path.** A coarse shared seed (nominal focal and
  mount tilt, zero distortion) calibrates every camera of a strongly tilted rig to
  oracle parity (see the measured results in `docs/notes/scheimpflug-intrinsics.md`).
- **Public CI guard.** The synthetic regression test
  `seeded_coarse_prior_converges_on_strong_tilt_distortion`
  (`crates/vision-calibration/tests/scheimpflug_intrinsics.rs`) seeds a coarse focal
  (about 9 % low) on strong-tilt, strong-distortion data and converges to ground
  truth.
- **The cold-start staged solver is shared** with the experimental from-scratch and
  rig paths. Its Stage 2 honors the caller's robust loss and Stage 3 anchors its
  focal bounds to the seed. The from-scratch *rig* numbers are not gated.

## References

- ADR 0011 — manual initialization workflow (`ScheimpflugManualInit`, the seeding
  mechanism this builds on).
- ADR 0005 — composable camera model (the `sensor` tilt stage).
- ADR 0023 — device-spec seed derivation (structured source of the seeds).
- `docs/notes/scheimpflug-intrinsics.md` — convergence-basin evidence.

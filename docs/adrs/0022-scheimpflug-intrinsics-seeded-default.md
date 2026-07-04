# ADR 0022: Scheimpflug Intrinsics — User-Seeded Initialization is the Supported Default

- Status: Accepted
- Date: 2026-06-17
- Note (2026-07-02): the production-grade program builds on this decision —
  seeded init is the **official acceptance route** for all datasets. Track S
  turns the hand-coded seeds into a structured `DeviceSpec` layer (ADR 0023,
  planned), and **Q6** adds the convergence-basin study quantifying how much
  spec error the seeded route tolerates (the empirical evidence backing this
  ADR). See `docs/backlog.md` Tracks S/Q.
- Note (2026-07-04): Q4-MWIRE-SCHEIMPFLUG generalized the seeded route beyond
  Brown-Conrady-5. `ScheimpflugIntrinsicsConfig::distortion_model` (default
  `DistortionKind::BrownConrady5`) now selects the model `step_optimize`
  refines. The Phase A `k1` multi-start grid `{0, −0.20, −0.40}` generalizes
  through `with_leading_radial`, which sweeps the **leading radial term** —
  `k1` for BrownConrady5/Rational8/ThinPrism9, `lambda` for Division1 — and is
  a no-op for `DistortionKind::None`. The `fix_distortion`
  (`DistortionFixMask`) mask — five named bits `{k1, k2, k3, p1, p2}` — is
  translated **by name** onto each model's packed layout by
  `fix_mask_indices` (the single mechanism used for both user masks and the
  A0/A1 staging masks). For BrownConrady5 it is exactly
  `DistortionFixMask::to_indices()` (byte-identical to the pre-M-WIRE path).
  For the extended models the shared coefficients map by name and each model's
  extra coefficients follow the mask bit of the family they belong to:
  Rational8's higher-order radial block `k4,k5,k6` follows the `k3` bit;
  ThinPrism9's prism block `s1..s4` is fixed iff **both** `p1` and `p2` are
  fixed; Division1's single `lambda` follows the leading-radial `k1` bit
  (consistent with `with_leading_radial`). This keeps the staging invariants —
  A0 all-fixed, A1 `{k1,k2 free; k3/p1/p2 fixed}` and the default `fix_k3`
  semantics — intact across every model. Rigs are unaffected:
  `SensorMode::Scheimpflug::distortion_model` remains BC5-typed
  and is rejected up front for non-BC5 values in `validate_config` (ADR
  0019) — the joint rig bundle adjustment stays Brown-Conrady-only.
- Note (2026-07-04): **Q6-BASIN-STUDY closes the forward pointer above** —
  `calib-bench basin` (`crates/vision-calibration-bench/src/basin.rs`)
  perturbs the ADR 0023 device-spec seed over independent focal / tilt /
  principal-point sweeps and re-runs the seeded route per cell, gated on
  each entry's acceptance threshold. Measured on both private families
  (`rtv3d_ref`, 6 cameras, gate ≤ 0.5 px; `rtv3d_ringgrid`, 6 cameras, gate
  ≤ 1.0 px): the all-camera-pass basin is ×[0.85, 1.30] on focal and
  ±4° on tilt (the full tested tilt sweep — no failure observed at any
  offset) on **both** families, comfortably containing the decision
  rule's realistic-spec-error envelope (focal ±5 %, tilt ±2°) with 2-3×
  margin. See `docs/notes/scheimpflug-intrinsics.md` for the full tables
  and `docs/backlog.md` Q6 for the completion note.

## Context

From-scratch Scheimpflug **intrinsics** calibration is unstable. The camera model
couples sensor tilt, focal length, and radial distortion (`pixel =
K(sensor_tilt(distortion(project(dir))))`), and these three trade off against one
another — the classic tilt↔focal↔distortion degeneracy. Zhang's pinhole
initialization ignores tilt, so on a tilted sensor it underestimates the focal
(observed `fx≈944` vs an oracle `~1153`), and the non-linear solve then settles
into a wrong tilt/focal basin.

Two prior efforts confirm the fragility:

- The P6 work added a tilt-aware linear initializer plus a rig-level
  auto-recovery pass (see `docs/report/2026-06-16-P6-PERCAM-CONVERGENCE-*.md`).
  It reaches `~0.41 px` mean on the private `rtv3d_ref` rig from scratch, but is
  basin-fragile: a public synthetic regression failed at the branch base, and the
  final joint hand-eye leaves camera 0 at `~0.528 px`.
- Two further from-scratch sessions on `rtv3d_ref` had only partial success — the
  hard cameras (3 and 4) diverge.

A from-scratch warm-up is genuinely research-grade. But in practice the data
needed to *seed* the solve is already on hand: the lens focal length (a datasheet
value) and the Scheimpflug **mount tilt** (a deliberately-set mechanical angle —
the whole point of a Scheimpflug setup). Seeding both removes the fragility.

While validating this we also root-caused why even a seeded solve diverged on the
two hard cameras: starting from `k1 = 0`, the fixed-tilt sub-problem has a
**spurious local minimum at `k1 ≈ 0`** — the per-view poses adapt to absorb the
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

Both the focal **and** the tilt seed are load-bearing (this supersedes the
initial assumption that the tilt seed was merely advisory):

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

## Consequences

- **Deterministic, robust production path.** A coarse shared seed (nominal focal +
  `−5°` mount tilt) calibrates the full `rtv3d_ref` rig to oracle parity. The
  unseeded path is unchanged in behavior except for the added log warning.
- **Validation.** Private harness `rtv3d_ref_intrinsics` (in
  `vision-calibration-examples-private`) calibrates each of the 6 cameras from a
  coarse seed `fx=fy=1150, pp=(360,270), tilt_x=−0.087, distortion=0` and exits
  non-zero on any camera > 0.5 px. Result, all six **PASS**:

  | cam | ours (px) | oracle (px) |
  |-----|-----------|-------------|
  | 0   | 0.373     | 0.303       |
  | 1   | 0.267     | 0.292       |
  | 2   | 0.282     | 0.246       |
  | 3   | 0.473     | 0.272       |
  | 4   | 0.342     | 0.277       |
  | 5   | 0.321     | 0.273       |

  Camera 3 (`0.473 px`) is the tightest margin; it still clears the gate. The
  public CI guard is the synthetic regression test
  `seeded_coarse_prior_converges_on_strong_tilt_distortion`
  (`crates/vision-calibration/tests/scheimpflug_intrinsics.rs`), which seeds a
  coarse focal (≈9 % low) on rtv3d-like strong-tilt + strong-distortion data and
  converges to ground truth.
- **The cold-start staged solver is shared** with the experimental from-scratch
  and rig paths. Its Stage 2 now respects the caller's robust loss and Stage 3
  anchors its focal bounds to the seed; these are robustness improvements that
  keep all existing tests green but may shift the experimental from-scratch *rig*
  numbers (which are not gated).

## References

- ADR 0011 — manual initialization workflow (`ScheimpflugManualInit`, the seeding
  mechanism this builds on).
- ADR 0005 — composable camera model (the `sensor` tilt stage).
- `docs/report/2026-06-16-P6-PERCAM-CONVERGENCE-tilt-aware-init.md` and
  `…-diagnosis.md` — the from-scratch fragility this decision steps around.

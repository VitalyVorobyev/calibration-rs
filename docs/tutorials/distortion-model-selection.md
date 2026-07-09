# Distortion model selection

> Onboarding tutorial for [ADR 0020](../adrs/0020-camera-model-as-data-factor-ir.md)
> and [ADR 0022](../adrs/0022-scheimpflug-intrinsics-seeded-default.md).
> Runnable companions:
> [`planar_distortion_models.rs`](../../crates/vision-calibration-pipeline/tests/planar_distortion_models.rs),
> [`scheimpflug_distortion_models.rs`](../../crates/vision-calibration-pipeline/tests/scheimpflug_distortion_models.rs).

## Why

Every intrinsics-bearing problem type picks a lens distortion model and a
per-parameter fix mask. Get it wrong and you either overfit sparse data
(a free `k3` on 20 views chases noise) or underfit a genuinely distorted
wide-angle lens. This tutorial explains the available models, the default
fix mask and why it's the default, and — grounded in the Q4 measurement
study — when reaching for a richer model actually buys you anything.

## Mental model

### The models

[`DistortionKind`](../../crates/vision-calibration/src/lib.rs) (re-exported
by the facade as `vision_calibration::optim::DistortionKind`) selects the
packed coefficient layout a factor compiles against:

| Kind | dim | Packed layout |
|---|---:|---|
| `None` | 0 | (no distortion block) |
| `BrownConrady5` | 5 | `[k1, k2, k3, p1, p2]` |
| `Rational8` | 8 | `[k1, k2, k3, k4, k5, k6, p1, p2]` |
| `ThinPrism9` | 9 | `[k1, k2, k3, p1, p2, s1, s2, s3, s4]` |
| `Division1` | 1 | `[lambda]` |

`BrownConrady5` is the workspace default
(`common::config::default_distortion_kind`) and the **only** model every
downstream consumer accepts — rig bundle adjustment, hand-eye, and
laserline all expect a `PinholeCamera` with Brown-Conrady coefficients.
The extended models (`Rational8`, `ThinPrism9`, `Division1`) are wired
through exactly two single-camera paths: `PlanarIntrinsicsConfig` and
`ScheimpflugIntrinsicsConfig`. An extended-model export is a dead end —
it cannot flow into `RigExtrinsics` / `RigHandeye` / `LaserlineDevice`.

`SensorMode::Scheimpflug` (the rig sensor flavour) also carries a
`distortion_model` field for schema symmetry with the single-camera path,
but `validate_config` rejects anything but `BrownConrady5` up front — the
joint rig bundle adjustment is Brown-Conrady-typed throughout.

### The fix mask

`CameraFixMask { intrinsics: IntrinsicsFixMask, distortion: DistortionFixMask }`
is the one per-parameter mask idiom (ADR 0024). `DistortionFixMask`'s five
named bits are `{k1, k2, k3, p1, p2}` — always BC5-shaped, even when the
active model is one of the extended ones. `fix_mask_indices` translates
the mask **by name** onto each model's packed layout:

- `BrownConrady5`: identity (byte-identical to the pre-Q4 path).
- `Rational8`: the higher-order radial block `k4, k5, k6` follows the `k3` bit.
- `ThinPrism9`: the prism block `s1..s4` is fixed iff **both** `p1` and `p2` are fixed.
- `Division1`: the single `lambda` follows the `k1` bit.

`DistortionFixMask::default()` fixes `k3` and frees everything else —
**`fix_k3: true` is the default because k3 often overfits sparse or
noisy corner data; only relax it for wide-angle lenses or high-quality
data** (the same rule applies to the linear-init stage's
`IntrinsicsInitConfig::fix_k3`, which defaults to `true` for the same
reason). This applies whichever model is active: relaxing `k3` on
`Rational8` also relaxes `k4, k5, k6`.

### Scheimpflug-specific defaults

`ScheimpflugIntrinsicsConfig` overrides two of the shared sub-struct
defaults, both because sensor tilt couples with distortion:

- `init.fix_tangential = true` (vs. the shared default `false`) — a free
  tangential term is ill-posed during the linear init stage on a tilted
  sensor.
- `fix_camera.distortion = DistortionFixMask::radial_only()` (k3, p1, p2
  fixed; k1, k2 free) rather than the shared k3-only default — tilt can
  absorb tangential-distortion-like signal, so leaving `p1`/`p2` free by
  default overfits.

The tilt term itself has its own mask, `ScheimpflugFixMask { tilt_x, tilt_y }`
— a separate knob from `fix_camera.distortion`, since tilt is a sensor
parameter, not a distortion one.

`with_leading_radial` sweeps the model's **leading radial term** — `k1`
for `BrownConrady5`/`Rational8`/`ThinPrism9`, `lambda` for `Division1`, a
no-op for `None` — during the seeded route's Phase A multi-start (see
[ADR 0022](../adrs/0022-scheimpflug-intrinsics-seeded-default.md) and the
[manual initialization](./manual-init.md) tutorial for why Scheimpflug
intrinsics starts from a seed at all).

## Walkthrough

### 1. Do nothing (the default)

`PlanarIntrinsicsConfig::default()` and `ScheimpflugIntrinsicsConfig::default()`
already select `BrownConrady5` with `fix_k3: true`. Most datasets should
start here.

### 2. Choose an extended model explicitly

```rust
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;
use vision_calibration::core::{CameraFixMask, DistortionFixMask};
use vision_calibration::optim::DistortionKind;

let mut config = PlanarIntrinsicsConfig::default();
config.distortion_model = DistortionKind::Rational8;
config.fix_camera = CameraFixMask {
    distortion: DistortionFixMask::all_free(), // k1..k6, p1, p2 all refined
    ..Default::default()
};
```

> The facade re-exports `DistortionKind` as
> `vision_calibration::optim::DistortionKind`, alongside the other
> optim config-field enums (`RobustLoss`, `HandEyeMode`). It is a public
> field type on three re-exported configs (`PlanarIntrinsicsConfig`,
> `ScheimpflugIntrinsicsConfig`, `SensorMode::Scheimpflug`), so importing
> it through the facade keeps a consumer off the `vision_calibration_optim`
> crate directly.

Switching `distortion_model` alone does **not** free the model's extra
coefficients — they stay at their zero seed unless the fix mask also frees
them (see the Q4 measurement below for why this matters).

### 3. Scheimpflug path (seeded)

```rust
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsConfig, ScheimpflugManualInit, ScheimpflugIntrinsicsProblem,
    step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::optim::DistortionKind;

let mut config = ScheimpflugIntrinsicsConfig::default(); // BC5, radial_only, fix_tangential
config.distortion_model = DistortionKind::ThinPrism9;

let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
session.set_input(dataset)?;
session.set_config(config)?;

let seed = ScheimpflugManualInit {
    intrinsics: Some(nominal_k),   // datasheet focal / pixel pitch
    sensor: Some(mount_tilt),      // mechanical mount angle
    ..Default::default()
};
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
```

The linear init stage always produces Brown-Conrady coefficients; they are
embedded into the chosen model with any extra degrees of freedom zeroed
before the non-linear refine.

## The Q4 measurement: does a richer model actually help?

`Q4-MWIRE-SCHEIMPFLUG` (2026-07-04) measured mean reprojection error under
each distortion model on the two private Scheimpflug rigs, `rtv3d_ref`
(6 cameras) and `rtv3d_ringgrid` (6 cameras), via the `Q4_DISTORTION_SWEEP=1`
env-gated sweep in the private intrinsics examples. **Under the production
`radial_only` fix mask** — the config every shipped bench baseline uses —
**the model choice alone does not move the seeded floor**:

- `rtv3d_ref`: `BrownConrady5 = Rational8 = ThinPrism9 = 0.3316 px` median
  mean-reproj (the extra `k4-k6` / `s1-s4` coefficients stay at their zero
  seed under `radial_only` and never move); `Division1 = 0.3358 px`
  (stable, but its worst camera, cam3, sits alone at `0.838 px`).
- `rtv3d_ringgrid`: `0.4739 px` for the three polynomial models,
  `0.4745 px` for `Division1`.

This is expected: `radial_only` only frees `k1`/`k2` (or `lambda`), so
`Rational8` and `ThinPrism9` are Brown-Conrady in every model but name.

An earlier all-coefficients-free sweep (before the `fix_mask_indices`
codex-review fix landed) hinted that the extra degrees of freedom *can*
help — the ringgrid knife-edge camera (cam1) improved `0.5008 → 0.4988 px`
— but the same all-free configuration **diverged** on `rtv3d_ref`'s
thin-data cameras 3 and 4. A controlled all-free sweep needs a proper
per-camera warm start to be safe; that is tracked as a follow-up, not
shipped.

### Practical guidance

- **Default to `BrownConrady5` with `fix_k3: true`** (or Scheimpflug's
  `radial_only`) unless you have concrete evidence of higher-order
  distortion — very wide FOV or fisheye-adjacent lenses. Every committed
  bench baseline stays on `BrownConrady5` because, under the production
  mask, switching the model is a no-op.
- If you do reach for an extended model, **free its new coefficients
  deliberately** via `fix_camera.distortion` (or a Scheimpflug
  `fix_camera` override) — selecting `distortion_model` alone does not
  unlock any new degrees of freedom.
- Freeing every extra coefficient is genuinely experimental: it needs a
  per-camera warm start to avoid diverging on thin-data views. Do not do
  this on production data without first reproducing the Q4 measurement on
  your own dataset.

## Common variations

- **Wide-angle / fisheye-adjacent lenses**: set
  `fix_camera.distortion.k3 = false` (and, for planar intrinsics,
  `init.fix_k3 = false` so the linear stage also fits it) — the one
  config knob this tutorial's default advice tells you to avoid flipping
  without a reason.
- **`DistortionKind::None`**: pinhole-only, for pre-rectified or synthetic
  inputs. Not accepted on the Scheimpflug path — a Scheimpflug intrinsics
  solve always carries a distortion model (`scheimpflug_model_desc`
  rejects `None` with a typed error).
- **Rig / hand-eye / laserline pipelines**: always Brown-Conrady. There is
  no config knob to pass an extended model through — `SensorMode::Scheimpflug::distortion_model`
  exists only for schema symmetry and is validated to `BrownConrady5`.

## What to read next

- [ADR 0020](../adrs/0020-camera-model-as-data-factor-ir.md) — the
  descriptor-as-data IR design `DistortionKind` dispatches through.
- [ADR 0022](../adrs/0022-scheimpflug-intrinsics-seeded-default.md) — why
  Scheimpflug intrinsics is seeded by default, and the 2026-07-04 note
  with the full Q4 mechanics.
- [`docs/notes/scheimpflug-intrinsics.md`](../notes/scheimpflug-intrinsics.md)
  — the Scheimpflug proof-pack stub, including the basin study this
  tutorial's seeded-route guidance builds on.
- [Manual initialization](./manual-init.md) — the `ScheimpflugManualInit`
  seeding mechanism used in the walkthrough above.
- [Per-feature residuals](./per-feature-residuals.md) — drill into
  per-corner errors after a calibration finishes, model by model.

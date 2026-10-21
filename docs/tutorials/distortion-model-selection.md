# Distortion model selection

> Runnable companions:
> [`planar_distortion_models.rs`](../../crates/vision-calibration-pipeline/tests/planar_distortion_models.rs),
> [`scheimpflug_distortion_models.rs`](../../crates/vision-calibration-pipeline/tests/scheimpflug_distortion_models.rs).

## Why

Every intrinsics-bearing problem type picks a lens distortion model and a
per-parameter fix mask. Get it wrong and you either overfit sparse data
(a free `k3` on 20 views chases noise) or underfit a genuinely distorted
wide-angle lens. This tutorial explains the available models, the default
fix mask and why it's the default, and — grounded in the measurement
study — when reaching for a richer model actually buys you anything.

## Mental model

### The models

`DistortionKind` (`vision_calibration::core::DistortionKind`) fills the
distortion slot of a camera model, `camera.model.distortion` in
`PlanarIntrinsicsConfig`, and selects the packed coefficient layout a factor
compiles against:

| Kind | dim | Packed layout |
|---|---:|---|
| `None` | 0 | (no distortion block) |
| `BrownConrady5` | 5 | `[k1, k2, k3, p1, p2]` |
| `Rational8` | 8 | `[k1, k2, k3, k4, k5, k6, p1, p2]` |
| `ThinPrism9` | 9 | `[k1, k2, k3, p1, p2, s1, s2, s3, s4]` |
| `Division1` | 1 | `[lambda]` |

`BrownConrady5` is the default and the **only** model every downstream
consumer accepts — rig bundle adjustment, hand-eye, and laserline all expect
a `PinholeCamera` with Brown-Conrady coefficients. The extended models
(`Rational8`, `ThinPrism9`, `Division1`) are available in planar intrinsics,
with a straight or a tilted (Scheimpflug) sensor. An extended-model export
is a dead end — it cannot flow into `RigExtrinsics` / `RigHandeye` /
`LaserlineDevice`.

`SensorMode::Scheimpflug` (the rig sensor flavour) also carries a
`distortion_model` field, but `validate_config` rejects anything but
`BrownConrady5` up front — the joint rig bundle adjustment is
Brown-Conrady-typed throughout.

### The fix lists

`camera.fix` (a `CameraFix`) names, per slot, the parameters the non-linear
solve holds fixed — `intrinsics` (`fx, fy, cx, cy`), `distortion` (the
model's own coefficient names) and `sensor` (`tilt_x, tilt_y`). A slot left
`None` keeps the camera model's default; an explicit list replaces it; an
unknown name is a configuration error before any work.

The defaults:

- `BrownConrady5`: `k3` fixed — **`k3` often overfits sparse or noisy
  corner data; only relax it for wide-angle lenses or high-quality data**
  (the same rule applies to the linear-init stage's
  `IntrinsicsInitConfig::fix_k3`, which defaults to `true` for the same
  reason).
- The extended models: nothing fixed — they are chosen for their extra
  terms.
- Any model under a Scheimpflug sensor: everything but the low-order radial
  terms (`k1`, `k2`; the division model's `lambda`) fixed — see below.

### Scheimpflug-specific defaults

A Scheimpflug sensor (`camera.model.sensor = SensorKind::Scheimpflug`)
changes two defaults, both because sensor tilt couples with distortion:

- `init.fix_tangential` resolves to `true` (it is `false` for an untilted
  camera) — a free tangential term is ill-posed during the linear init stage
  on a tilted sensor.
- The default distortion fix list keeps only `k1`, `k2` (or `lambda`) free —
  tilt can absorb tangential-distortion-like signal, so leaving `p1`/`p2`
  free by default overfits.

The tilt itself is the `sensor` slot of the same fix list
(`camera.fix.sensor = Some(vec!["tilt_y".into()])` holds `tilt_y`), since
tilt is a sensor parameter, not a distortion one.

`with_leading_radial` sweeps the model's **leading radial term** — `k1`
for `BrownConrady5`/`Rational8`/`ThinPrism9`, `lambda` for `Division1`, a
no-op for `None` — during the seeded route's k1 multi-start sweep (see the
[manual initialization](./manual-init.md) tutorial for why a tilted camera
starts from a seed at all).

## Walkthrough

### 1. Do nothing (the default)

`PlanarIntrinsicsConfig::default()` selects `BrownConrady5` with `k3`
fixed and an untilted sensor. Most datasets should start here.

### 2. Choose an extended model explicitly

```rust
use vision_calibration::core::DistortionKind;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.distortion = DistortionKind::Rational8;
// The extended models fix nothing by default: k1..k6, p1, p2 are all refined.
// Name what to hold, e.g. the higher-order radial block:
config.camera.fix.distortion = Some(vec!["k4".into(), "k5".into(), "k6".into()]);
```

### 3. Tilted sensor (seeded)

```rust
use vision_calibration::core::{DistortionKind, SensorKind};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, PlanarManualInit, step_init_with_seed,
    step_optimize,
};
use vision_calibration::session::CalibrationSession;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;    // radial-only default fix list
config.camera.model.distortion = DistortionKind::ThinPrism9;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
session.set_config(config)?;

let mut seed = PlanarManualInit::default();
seed.intrinsics = Some(nominal_k);   // datasheet focal / pixel pitch
seed.sensor = Some(mount_tilt);      // mechanical mount angle
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
```

The linear init stage always produces Brown-Conrady coefficients; they are
embedded into the chosen model with any extra degrees of freedom zeroed
before the non-linear refine. A manual Brown-Conrady `distortion` seed is
embedded the same way, and the init log notes it.

## Does a richer model actually help?

Under a Scheimpflug sensor, selecting a richer distortion model is not, on
its own, a change to what is optimized — and that surprises people. We measured mean reprojection
error under each model on two six-camera Scheimpflug rigs. **Under the
radial-only fix list** (`k1`, `k2` free) — the configuration every shipped
baseline uses —
**the model choice alone does not move the floor**:

- Rig A: `BrownConrady5 = Rational8 = ThinPrism9 = 0.3316 px` median
  mean-reproj (the extra `k4-k6` / `s1-s4` coefficients stay at their zero
  seed under the radial-only list and never move); `Division1 = 0.3358 px`
  (stable, but its worst camera sits alone at `0.838 px`).
- Rig B: `0.4739 px` for the three polynomial models, `0.4745 px` for
  `Division1`.

This is expected: the radial-only list frees only `k1`/`k2` (or `lambda`), so
`Rational8` and `ThinPrism9` are Brown-Conrady in every model but name.

Freeing *every* coefficient does move the numbers, in both directions. On
one rig it improved the hardest camera slightly (`0.5008 → 0.4988 px`); on
the other it **diverged** on the two cameras with the thinnest data. An
all-free configuration needs a per-camera warm start to be safe, which is
why it is not the default.

### Practical guidance

- **Default to `BrownConrady5` with `k3` fixed** (or, under a Scheimpflug
  sensor, the radial-only list) unless you have concrete evidence of higher-order
  distortion — very wide FOV or fisheye-adjacent lenses. Every committed
  bench baseline stays on `BrownConrady5` because, under the production
  mask, switching the model is a no-op.
- If you do reach for an extended model, **free its new coefficients
  deliberately** via `camera.fix.distortion` when the sensor's default
  holds them — under a Scheimpflug sensor, selecting the model alone does
  not unlock any new degrees of freedom.
- Freeing every extra coefficient is genuinely experimental: it needs a
  per-camera warm start to avoid diverging on thin-data views. Do not do
  this on production data without first reproducing the comparison above
  on your own dataset.

## Common variations

- **Wide-angle / fisheye-adjacent lenses**: set
  `camera.fix.distortion = Some(vec![])` (and `init.fix_k3 = false` so the
  linear stage also fits `k3`) — the one
  config knob this tutorial's default advice tells you to avoid flipping
  without a reason.
- **`DistortionKind::None`**: pinhole-only, for pre-rectified or synthetic
  inputs. Not accepted with a Scheimpflug sensor — the tilt is estimated
  jointly with the radial terms, so config validation rejects `None` with a
  typed error.
- **Rig / hand-eye / laserline pipelines**: always Brown-Conrady. There is
  no config knob to pass an extended model through — `SensorMode::Scheimpflug::distortion_model`
  exists only for schema symmetry and is validated to `BrownConrady5`.

## What to read next

- [Manual initialization](./manual-init.md) — the `PlanarManualInit`
  seeding mechanism used in the walkthrough above.
- [Per-feature residuals](./per-feature-residuals.md) — drill into
  per-corner errors after a calibration finishes, model by model.

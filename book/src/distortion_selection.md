# Choosing a Distortion Model

Every workflow that estimates intrinsics picks a lens distortion model and a per-parameter fix list. Get it wrong and you either overfit sparse data (a free `k3` on 20 views chases noise) or underfit a genuinely distorted wide-angle lens. This chapter lists the available models, the default fix lists and why they are the defaults, and when reaching for a richer model actually buys anything.

## The Models

`DistortionKind` (`vision_calibration::core::DistortionKind`) fills the distortion slot of the camera model (`camera.model.distortion` in every `*Config`) and selects the packed coefficient layout the factors compile against:

| Kind | dim | Packed layout | Model |
|---|---:|---|---|
| `None` | 0 | (no distortion block) | no distortion |
| `BrownConrady5` | 5 | `[k1, k2, k3, p1, p2]` | Brown-Conrady radial and tangential (default) |
| `Rational8` | 8 | `[k1, k2, k3, k4, k5, k6, p1, p2]` | rational polynomial |
| `ThinPrism9` | 9 | `[k1, k2, k3, p1, p2, s1, s2, s3, s4]` | Brown-Conrady plus thin prism |
| `Division1` | 1 | `[lambda]` | Fitzgibbon division model |

Every workflow calibrates every kind with a straight or a tilted (Scheimpflug) sensor: planar intrinsics, single-camera hand-eye and the laserline device through `camera`, the rigs per camera (`camera` plus `camera_overrides`). The rig laser workflows take each camera of the upstream rig in its own model.

For the non-default models, `export.params.distortion()` returns `None`; read the coefficients from `export.params.camera.distortion` (`DistortionParams`, see [Serialization and Runtime-Dynamic Types](params.md)).

## The Fix Lists

`camera.fix` (a `CameraFix`) names, per slot, the parameters the non-linear solve holds fixed:

- `intrinsics`: any of `fx, fy, cx, cy`;
- `distortion`: the model's own coefficient names;
- `sensor`: `tilt_x, tilt_y`.

A slot left `None` keeps the camera model's default; an explicit list replaces it; an unknown name is a configuration error raised before any work. In JSON: `"camera": {"model": {...}, "fix": {"intrinsics": ["cx", "cy"], "distortion": ["k3", "p1", "p2"]}}`.

The defaults:

- `BrownConrady5`: `k3` fixed. **`k3` often overfits sparse or noisy corner data; relax it only for wide-angle lenses or high-quality data.** The same rule applies to the linear stage's `IntrinsicsInitConfig::fix_k3`, which defaults to `true` for the same reason.
- The extended models: nothing fixed, since they are chosen for their extra terms.
- Any model under a Scheimpflug sensor: everything but the low-order radial terms (`k1`, `k2`; the division model's `lambda`) fixed. See below.

### Scheimpflug-Specific Defaults

A Scheimpflug sensor (`camera.model.sensor = SensorKind::Scheimpflug`) changes two things, both because the sensor tilt couples with distortion:

- The default distortion fix list keeps only `k1`, `k2` (or `lambda`) free: the tilt can absorb tangential-distortion-like signal, so leaving `p1` and `p2` free overfits.
- The linear stage holds `p1` and `p2` at zero while the tilt is estimated, even when the fix list frees them: a free tangential term is ill-posed next to the tilt in the linear fit. The non-linear solve, which separates them, then moves them.

In every workflow the linear stage fits only the distortion terms the camera's solves estimate: a term the fix list holds stays at zero from the start, rather than at a linear fit the later solves would keep.

The tilt itself is the `sensor` slot of the same fix list (`camera.fix.sensor = Some(vec!["tilt_y".into()])` holds `tilt_y`), since the tilt is a sensor parameter, not a distortion one.

## Configuring a Model

### Do nothing

`PlanarIntrinsicsConfig::default()` selects `BrownConrady5` with `k3` fixed and an untilted sensor. Most datasets should start here.

### Choose an extended model

```rust
use vision_calibration::core::DistortionKind;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.distortion = DistortionKind::Rational8;
// The extended models fix nothing by default: k1..k6, p1, p2 are all refined.
// Name what to hold, e.g. the higher-order radial block:
config.camera.fix.distortion = Some(vec!["k4".into(), "k5".into(), "k6".into()]);
```

### Fix intrinsics or distortion terms

```rust
// Fix cx, cy but optimize fx, fy
session.update_config(|c| {
    c.camera.fix.intrinsics = Some(vec!["cx".into(), "cy".into()]);
})?;

// Fix k3 and the tangential terms
session.update_config(|c| {
    c.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
})?;
```

### A tilted sensor, seeded

```rust
use vision_calibration::core::{DistortionKind, SensorKind};
use vision_calibration::planar_intrinsics::{
    IntrinsicsManualInit, PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init_with_seed,
    step_optimize,
};
use vision_calibration::session::CalibrationSession;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;    // radial-only default fix list
config.camera.model.distortion = DistortionKind::ThinPrism9;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
session.set_config(config)?;

let mut seed = IntrinsicsManualInit::default();
seed.intrinsics = Some(nominal_k);   // datasheet focal length / pixel pitch
seed.sensor = Some(mount_tilt);      // mechanical mount angle
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
```

The linear stage always produces Brown-Conrady coefficients; they are embedded into the chosen model with any extra degrees of freedom zeroed before the non-linear refinement. A Brown-Conrady `distortion` seed is embedded the same way, and the initialization log notes it. See [Seeding with Prior Knowledge](manual_init.md) for the seeding mechanism and [Planar Intrinsics Calibration](planar_intrinsics.md#tilted-scheimpflug-sensor) for why a tilted camera starts from a seed.

## Does a Richer Model Help?

Under a Scheimpflug sensor, selecting a richer distortion model is not on its own a change to what is optimized, and that surprises people. Mean reprojection error was measured under each model on two six-camera Scheimpflug rigs. **Under the radial-only fix list** (`k1`, `k2` free), the configuration the shipped baselines use, **the model choice alone does not move the floor**:

- Rig A: `BrownConrady5 = Rational8 = ThinPrism9 = 0.3316 px` median mean reprojection error (the extra `k4`-`k6` and `s1`-`s4` coefficients stay at their zero seed under the radial-only list and never move); `Division1 = 0.3358 px` (stable, but its worst camera sits alone at `0.838 px`).
- Rig B: `0.4739 px` for the three polynomial models, `0.4745 px` for `Division1`.

This is expected: the radial-only list frees only `k1` and `k2` (or `lambda`), so `Rational8` and `ThinPrism9` are Brown-Conrady in everything but name.

Freeing *every* coefficient does move the numbers, in both directions. On one rig it improved the hardest camera slightly (`0.5008 -> 0.4988 px`); on the other it **diverged** on the two cameras with the thinnest data. An all-free configuration needs a per-camera warm start to be safe, which is why it is not the default.

### Practical Guidance

- **Default to `BrownConrady5` with `k3` fixed** (or, under a Scheimpflug sensor, the radial-only list) unless you have concrete evidence of higher-order distortion, such as a very wide field of view or a fisheye-adjacent lens. Every committed benchmark baseline stays on `BrownConrady5` because, under the production mask, switching the model is a no-op.
- If you reach for an extended model, **free its new coefficients deliberately** via `camera.fix.distortion` when the sensor's default holds them. Under a Scheimpflug sensor, selecting the model alone does not unlock any new degrees of freedom.
- Freeing every extra coefficient is experimental: it needs a per-camera warm start to avoid diverging on thin-data views. Reproduce the comparison above on your own dataset before doing it on production data.

## Special Cases

- **Wide-angle or fisheye-adjacent lenses**: set `camera.fix.distortion = Some(vec![])` and `init.fix_k3 = false` so the linear stage also fits `k3`. This is the one knob the default advice tells you not to flip without a reason.
- **`DistortionKind::None`**: pinhole-only, for pre-rectified or synthetic inputs. It is not accepted with a Scheimpflug sensor: the tilt is estimated jointly with the radial terms, so configuration validation rejects `None` with a typed error.
- **Hand-eye and laser workflows**: the same `camera.model.distortion` choice as planar intrinsics. A rig sets it per camera, and the rig laser workflows inherit each camera's model from the upstream rig.

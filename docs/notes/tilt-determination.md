# Tilt determination — when to estimate a Scheimpflug tilt, and what fixing it costs

Family: single-camera intrinsics on a tilted sensor, and what the camera is
worth downstream (rig extrinsics, hand-eye). Builds on
[tilt-identifiability.md](tilt-identifiability.md): the tilt is a gauge
without lens distortion (`K·H(τ) = K'·R(τ)`) and to first order a
decentring of the distortion with it (`p1 ≈ −k1·tan τx`, `p2 ≈ k1·tan τy`).
Design: [ADR 0031](../adrs/0031-tilt-determination.md) (proposed).

Question: the maintainer's procedure — fit the plain camera, decide from the
data whether the tilt is determined, estimate it if so, otherwise use a
prior if one is given, otherwise fix it at zero — what statistic decides,
what does each branch cost, and in what frame is the result?

Harness: `crates/vision-calibration-bench/examples/tilt_identifiability/`
(`determine.rs` the four fits and the decision statistics, `downstream.rs`
the gauge-invariant metrics, `verdict.rs` the sweep's tables,
`procedure.rs` the prior study and the end-to-end procedure). Every number
below is printed by

```text
cargo run --release -p vision-calibration-bench --example tilt_identifiability -- \
    --parts determine,prior,rule [--quick] [--dump draws.tsv]
```

(35 minutes on 12 cores: `determine` 17, `prior` 12, `rule` 5). Tests
(`cargo test --release -p vision-calibration-bench --example
tilt_identifiability`) pin the exact downstream evaluation, the decentring
seed and the plain camera's frame. **Measured** marks a number the harness
prints; **derived** a consequence of the algebra; **hypothesis** what
neither shows.

## Short answers

1. **The statistic** (A): the larger semi-axis `s` of the tilt's 2×2
   covariance as the tilted fit states it, **with the view-scatter
   scaling**. Under independent noise the stated `s` equals the Fisher σ
   at the truth to 1 % (20 views) or 0.55–1.08 (8 central views, deep in
   the undetermined region), so "determined iff `s ≤ θ`" errs only at the
   threshold: 8 of 1456 non-borderline draws at θ = 0.5°. With errors
   shared by a view's corners the unscaled σ is 2.3–5× short and calls 64
   of 160 undetermined draws determined; the scaled one, 0 (**measured**).
2. **Fixing the tilt at zero with `p1, p2` held is refuted** (B): in the
   cameras undetermined at θ = 0.5° it costs a median 2.0× and up to 31×
   the noise floor downstream (tilts 5–20°), and its frame is neither the
   lens axis nor the sensor normal. **Fixing it at zero with `p1, p2` free
   is confirmed** as the first-order equivalent: median 0.29×, above 2× in
   6 % of those cameras (up to 12.8×), and its frame is the sensor normal
   to 0.01–0.05° on weak lenses, 0.05–0.5° on moderate ones
   (**measured**). The claimed bound — the cost is the tilt × distortion
   term, small exactly when the tilt is undetermined — holds
   for the `p`-free fix to first order (`Δχ² ≈ (τ/σ(M2))²`), and fails for
   the `p`-held one, whose cost is `(τ/σ(M1))²` with `τ` the full mount
   angle.
3. **Where the `p`-free fix still costs** (B): narrow fields with strong
   distortion at a large tilt (the decentring `tan τ` exceeds the field,
   the first-order identity breaks: `Δχ²` up to 12× its prediction), and
   moderate distortion (−0.02…−0.05) at a 90° field and a 10–20° tilt. A
   likelihood-ratio guard — keep the tilted fit when the data reject the
   plain camera against tilt-plus-decentring at 99 % — leaves 8 of 1600
   draws above 2× the correct model's error (max 4.9×), seven of them with
   8 central views, where the training data cannot see what the full image
   shows (**measured**).
4. **A prior** (C): with `p1, p2` free under the prior the tilt comes from
   the prior (posterior σ ≈ σₚ wherever the data leave it undetermined) and
   a real decentring is measured, not reported as tilt: the 0.18° (6σ) bias
   of `p1 = 5e-4` drops to −0.024°. A wrong prior shows mostly as a wrong
   tilt: the decentring absorbs it, so predictions move ≤ 1.6× unless the
   conflict test flags it, and the data detect it only when the offset
   exceeds about `3·√(σ_data² + σₚ²)` (**measured**). The prior solve must
   reach the posterior: in the raw coordinates it stops a σ short (C0).
5. **End to end** (F): the plain camera through the pipeline, the tilted
   fit seeded from its decentring, the decision: the seed reaches the
   reference minimum in 512 of 512 draws (unbounded, tight stop); the
   procedure's camera is within 2.7× of the correct model's downstream
   error in every draw, median ≤ 1.4× per cell; the real-rig-like camera
   is estimated in 64 of 64 draws (**measured**).

## The four fits

| fit | tilt | `p1, p2` | what it is |
|---|---|---|---|
| M0 | held at 0 | held | the tilted camera's default fix list with the tilt fixed |
| M0p | held at 0 | free | the plain Brown-Conrady camera |
| M1 | free | held | today's tilted fit |
| M2 | free | free | tilt told apart from decentring only by the `r⁴` remainder |

Fits run from the truth (M0, M0p from the sensor-normal pinhole `K'` with
the poses in its frame) to a 1e-14 relative decrease, so the comparison is
of the problems, not the stopping rule. Sweep: HFOV 20–90° × corner
distortion −0.005…−0.15 × tilt 0, 2, 5, 10, 20° × {20 views over the whole
image, 8 views over its central half} × noise {0.2, 0.5} px, 4 draws each
(1600 draws) plus a noise-free fit per camera.

**Downstream metrics** (`downstream.rs`), each on noise-free held-out data
so it is the calibration's error alone, and none of them changed by a
choice of camera frame: held-out reprojection after a pose refit (12 views
over the whole image); a rig with a second, exactly known, untilted camera
(extrinsics solved with both cameras held, then held-out transfer of camera
1's pose into camera 2's pixels, and triangulation in camera 2's frame);
hand-eye with exact robot poses (solved, then predicting the corners at
held-out robot poses). *Floor*: the correct model's (M1's) RMS error under
noise. *Cost*: a fit's noise-free error over the floor, the worst of the
four metrics. Test `truth_predicts_exactly` pins every metric at zero
(< 1e-6) for the true camera.

## A. The decision statistic

**A1 — the stated σ is the Fisher σ** (median over tilts and draws;
stated / Fisher; RMS error / stated). 20 views, 0.2 px:

| corner distortion | 20° | 35° | 50° | 70° | 90° | σ with `p` free, 20° … 90° |
|---|---|---|---|---|---|---|
| −0.005 | 0.20° · 1.00 · 1.00 | 0.40 · 1.00 · 1.0 | 0.59 · 0.99 · 1.2 | 0.85 · 0.99 · 1.2 | 1.2 · 1.01 · 1.1 | 14 … 2.6° |
| −0.02 | 0.034 · 1.00 · 0.92 | 0.070 · 1.00 · 1.2 | 0.12 · 1.00 · 1.1 | 0.19 · 1.01 · 1.2 | 0.27 · 1.02 · 1.2 | 3.2 … 0.62° |
| −0.05 | 0.057 · 1.00 · 0.87 | 0.018 · 1.00 · 1.1 | 0.032 · 1.00 · 1.1 | 0.060 · 0.99 · 1.3 | 0.086 · 1.00 · 0.92 | 1.1 … 0.21° |
| −0.15 | 0.064 · 1.00 · 1.1 | 0.031 · 1.00 · 1.2 | 0.016 · 1.00 · 1.2 | 0.0078 · 1.00 · 1.1 | 0.0062 · 0.99 · 1.1 | 0.24 … 0.027° |

At 0.5 px every σ scales by 2.5; with 8 central views by 2–5. The stated σ
does not depend on the tilt, so the statistic needs no knowledge of it.
(The RMS/σ column pools 20 draws; its sampling spread is about ±0.3.)

**A2 — error rates of "determined iff `s ≤ θ`"** against the Fisher label
(determined when the Fisher σ ≤ θ/1.25, undetermined when ≥ 1.25·θ):

| θ | labelled determined | false undetermined | labelled undetermined | false determined | borderline (decided determined) |
|---|---:|---:|---:|---:|---|
| 0.1° | 380 | 0 | 1096 | 0 | 124 (77) |
| 0.25° | 700 | 8 | 716 | 0 | 184 (92) |
| 0.5° | 944 | 4 | 512 | 4 | 144 (74) |
| 1° | 1168 | 6 | 304 | 8 | 128 (66) |

With independent noise the statistic is as good as its label; the rate is a
property of the noise model, not the rule. What can fail it is a noise
model that does not hold:

**A3 — errors shared by a view's corners** (0.2 px plus a per-view
quadratic field of 0.3 px; 20 views, tilt 5°; 16 draws; label: the
Monte-Carlo σ). The library scaled the covariance in 318 of 320 draws, by
`k` = 7.5–19. Scaled, the stated σ is 0.71–1.57 of the Monte-Carlo σ (whose
own sampling error is about ±18 %); unscaled, 0.20–0.44 of it. At θ = 0.25°
the unscaled σ calls 64 of the 160 draws of undetermined cameras determined
(e.g. 20°, −0.005: Monte-Carlo σ 1.1°, unscaled 0.24°, 10/16), the scaled σ
none, and neither misses a determined camera (0 of 96). **The statistic
must be the scaled σ.**

Rejected statistics: σ with `p1, p2` free (A1's last column; ≤ 0.5° only
for 8 of 20 cameras, all strongly distorting, so nearly every
Scheimpflug lens would be "undetermined"; F1 below); the likelihood ratio
alone (F1: up to 20× the correct model's error, because it has no power at
a small tilt with a moderate lens while the fix still costs downstream);
the σ inflation `1/√(1 − R²)` (scales with `f·√N/σ_px`, not a precision).

## B. What fixing the tilt costs

**B1 — cost of M0 / M0p** (20 views, 0.2 px; `U`: Fisher σ > 0.5°):

| HFOV | distortion | σ(τ) M1 / M2 | tilt 2° | 5° | 10° | 20° |
|---|---|---|---|---|---|---|
| 20 | −0.005 | 0.19° / 14° | 4.0 / 0.048 | 8.4 / 0.091 | 12 / 0.27 | 15 / 0.94 |
| 20 | −0.15 | 0.065° / 0.24° | 12 / 3.2 | 30 / 12 | 60 / 30 | 122 / 71 |
| 35 | −0.02 | 0.068° / 1.8° | 10 / 0.38 | 22 / 0.71 | 34 / 1.3 | 47 / 4.0 |
| 50 | −0.005 | 0.58° / 5.5° U | 1.3 / 0.15 | 3.1 / 0.31 | 5.7 / 0.46 | 9.0 / 1.0 |
| 50 | −0.05 | 0.031° / 0.43° | 26 / 2.0 | 56 / 4.0 | 87 / 6.4 | 120 / 15 |
| 70 | −0.005 | 0.84° / 3.7° U | 0.87 / 0.19 | 2.1 / 0.42 | 4.0 / 0.70 | 7.1 / 1.5 |
| 90 | −0.005 | 1.1° / 2.7° U | 0.60 / 0.24 | 1.5 / 0.56 | 2.8 / 1.0 | 5.0 / 1.8 |
| 90 | −0.02 | 0.26° / 0.62° | 2.9 / 1.1 | 7.5 / 3.1 | 14 / 5.0 | 24 / 8.5 |
| 90 | −0.15 | 0.0062° / 0.028° | 189 / 61 | 634 / 182 | 849 / 388 | 1137 / 567 |

(The run prints all 20 cameras for each design and noise level.) Over every
camera undetermined at θ = 0.5° (Fisher label), tilt 5–20° (85 cells):

| fix | median cost | max | share above 2× | max with 20 views |
|---|---:|---:|---:|---:|
| M0 (`p` held) | 2.0× | 31× | 51 % | 11.7× |
| M0p (`p` free) | 0.29× | 12.8× | 6 % | 4.2× |
| M0, θ = 1° (54 cells) | 1.1× | 5.5× | 28 % | 5.0× |
| M0p, θ = 1° | 0.17× | 2.2× | 2 % | 1.8× |

M0p's costly cells: 20° field, −0.15, 8 central views, 0.5 px, tilt 10–20°
(3.8×, 12.8×); 90°, −0.02, 20 views at 0.5 px, tilt 10–20° (2.4×, 4.2×);
90°, −0.05, 8 central views at 0.5 px, tilt 20° (2.2×).

**B2 — the bound, as a likelihood ratio.** Fixing a coordinate costs its
squared z-score to first order, so M0 costs `Δχ² ≈ (τ/σ(M1))²` and M0p
`(τ/σ(M2))²` (**derived**). Measured on the noise-free fits (20 views,
0.2 px): at a 2° tilt the ratio of measured to predicted is 0.79–1.22 for
M0 in all 20 cameras and 0.7–1.3 for M0p in 15 (the exceptions: the 20°
field, where M0p's `Δχ²` is below 3, and the two −0.15 lenses at 20° and
90°, 1.6–1.75). At 20° M0's cost mostly saturates (0.09–0.6 of the
prediction: the other parameters take up part of the signal
nonlinearly; 0.9–1.35 for the strongest distortion on narrow fields), and
M0p's exceeds the prediction up to 12× in narrow fields (20°, −0.05:
3 276 against 276), where `tan τ` exceeds the field radius and the
decentring expansion fails. So the bound holds for the `p`-free fix at small and
moderate tilts, and the `p`-held fix's cost is `(τ/σ(M1))²`: large exactly
when the tilt is large, whatever its σ.

**B3 — per metric and the radial terms** (20 views, 0.2 px; `k1` bias in
units of its σ):

| camera | tilt | fit | held-out px | transfer px | triangulation ‰ | hand-eye px | `k1` |
|---|---|---|---:|---:|---:|---:|---:|
| 50°, −0.005 | floor | M1 | 0.015 | 0.046 | 0.13 | 0.056 | σ 8.3e-4 |
| | 20° | M0 | 0.13 | 0.42 | 0.96 | 0.51 | 9.5σ |
| | 20° | M0p | 0.016 | 0.027 | 0.043 | 0.029 | 1.3σ |
| 90°, −0.005 | floor | M1 | 0.018 | 0.056 | 0.14 | 0.054 | σ 2.6e-4 |
| | 20° | M0 | 0.089 | 0.27 | 0.54 | 0.26 | 2.9σ |
| | 20° | M0p | 0.032 | 0.047 | 0.10 | 0.049 | −0.47σ |
| 50°, −0.05 (determined) | 20° | M0p | 0.21 | 0.32 | 0.62 | 0.36 | 13σ |

The rig and hand-eye solves absorb the frame rotation exactly (their free
extrinsic and hand-eye rotations take it); what reaches them is the
distortion model's misfit, which is why the four metrics move together.

## C. A prior

Prior `τ ~ N(τ₀, σₚ²)` per axis, weighed against the corners at the plain
fit's pixel σ (both stated, so the covariance is the posterior); 16 draws.

**C0 — the solve must reach the posterior.** With `p1, p2` free, a weak
prior (σₚ = 1°) and the raw coordinates, the builder's Levenberg–Marquardt
stops near the truth from either start, 1° from the posterior mean, at a
cost 0.6 above the minimum (1798.72 vs 1798.10); in the gauge-aligned
coordinates (`K'` and sensor-frame poses) it reaches the minimum from both
starts in 9–16 iterations. The study's prior fits use those; the library's
must, or the σ-unit stopping rule (SOLVER-VALLEY-STOP) (**measured**).

**C1** (`tilt_x` bias / stated posterior σ, degrees; conflict: share of
draws with `|τ̂ − τ₀| / √(σ̂² + σₚ²) > 3`, `τ̂, σ̂` from the same free set
without the prior):

| camera | no prior: M1 / M2 | σₚ, offset | `p` held: bias / σ / conflict | `p` free: bias / σ / conflict |
|---|---|---|---|---|
| determined (50°, −0.05, 5°) | −0.005 ± 0.030 / −0.12 ± 0.43 | 0.25°, 0 | −0.005 / 0.030 / 0 | −0.023 / 0.22 / 0 |
| | | 0.25°, 1° | 0.009 / 0.030 / 16 | 0.73 / 0.22 / 2 |
| | | 0.25°, 3° | 0.039 / 0.030 / 16 | 2.3 / 0.22 / 16 |
| | | 1°, 3° | −0.003 / 0.030 / 7 | 0.39 / 0.39 / 6 |
| undetermined (70°, −0.005, 10°) | −0.08 ± 0.84 / −1.1 ± 2.8 | 0.25°, 0 | −0.001 / 0.24 / 0 | −0.005 / 0.25 / 0 |
| | | 0.25°, 1° | 0.92 / 0.24 / 2 | 0.99 / 0.25 / 0 |
| | | 0.25°, 3° | 2.7 / 0.24 / 9 | 3.0 / 0.25 / 1 |
| | | 1°, 1° | 0.39 / 0.64 / 0 | 0.85 / 0.97 / 0 |
| | | 1°, 3° | 1.2 / 0.64 / 5 | 2.8 / 0.98 / 1 |
| 8 central views (50°, −0.02, 10°) | 0.07 ± 0.44 / −0.1 ± 3.5 | 0.25°, 3° | 2.2 / 0.21 / 16 | 3.0 / 0.25 / 2 |
| decentred, determined (`p1` 5e-4) | **0.18** ± 0.030 / −0.12 ± 0.43 | 0.25°, 0 | **0.18** / 0.030 / 0 | **−0.024** / 0.22 / 0 |
| decentred, undetermined (`p1` 1e-4) | 0.75 ± 0.83 / −1.7 ± 2.9 | 1°, 0 | 0.46 / 0.64 / 0 | −0.099 / 0.96 / 0 |

- **Posterior σ.** Where the data leave the tilt undetermined the
  posterior σ is about the prior's (0.24–0.25 for σₚ = 0.25°; 0.64 with `p`
  held and 0.97 with `p` free for σₚ = 1°, against `√(1/(σ⁻² + σₚ⁻²))` of
  0.64 and 0.94). Where the prior dominates, the scatter over noise draws
  is smaller than that (RMS/σ 0.07–0.4 at offset 0) because the prior mean
  is the same in every draw: the posterior σ includes the prior's own
  uncertainty, as it should; the offset rows are the draws of that
  uncertainty (offset 1σₚ: RMS/σ 0.92–1.2).
- **Freeing `p1, p2` under a prior** removes the decentring bias (0.18° →
  −0.024°, `p1` recovered to 5e-5) and costs nothing downstream (median
  metric ratio against M1 0.998–1.14). With `p` held a determined lens
  ignores the prior (0.030° posterior, the bias stays 6σ).
- **A wrong prior** biases the tilt by `offset · σ_data² / (σ_data² + σₚ²)`
  (`σ_data` = σ(M1) with `p` held, σ(M2) free): 0.85° of 1° for σₚ = 1° and
  σ(M2) = 2.8°, predicted 0.89°. It is detectable when the offset exceeds
  about `3·√(σ_data² + σₚ²)`: with `p` held, 16/16 at 1° on the determined
  lens (threshold 0.76°), 9/16 at 3° on the undetermined one (2.6°), 2/16
  at 1°; with `p` free, almost never on an undetermined lens (0–1/16 even
  at 3°), because the decentring absorbs the offset. It then moves the
  predictions little (median ratio 1.0–1.6; 2.2 for a 3° offset with a
  0.25° prior on the determined lens, which the conflict test flags
  16/16): **a wrong prior with `p` free is a wrong frame, not a wrong
  camera.**

## D. The frame each branch reports

Rotation of the fitted camera frame from the lens-axis frame / from the
sensor-normal frame (noise-free fits; `frame_rotation`, the best rotation
between fitted and true training poses):

| camera | tilt | M0 | M0p | M1, M2 |
|---|---|---|---|---|
| 50°, −0.05 | 5° | 7.7 / 2.7 | 5.1 / 0.12 | 0 / 5.0 |
| 50°, −0.05 | 20° | 25 / 5.3 | 21 / 0.52 | 0 / 20 |
| 35°, −0.02 | 20° | 23 / 3.1 | 20 / 0.22 | 0 / 20 |
| 50°, −0.005 | 20° | 21 / 0.51 | 20 / 0.052 | 0 / 20 |
| 90°, −0.005 | 20° | 20 / 0.19 | 20 / 0.049 | 0 / 20 |
| 90°, −0.15 | 20° | 29 / 9.3 | 18 / 2.7 | 0 / 20 |

- **Estimated (and prior)**: the lens-axis frame — `z` along the lens axis,
  `(cx, cy)` where it meets the sensor, the distortion centre. A pose,
  extrinsic or hand-eye rotation is to that axis; comparing it with a
  drawing compares with the lens mount.
- **Fixed with `p` free**: the sensor-normal frame — `z` along the normal
  to the sensor through the projection centre, `(cx, cy)` its foot, the
  projective principal point of `K'`; the lens axis's offset is `p1, p2`.
  Within 0.01–0.05° on the weak lenses (whose tilt is undetermined),
  0.05–0.5° on moderate ones, 2.7° on the fisheye-like one (which the
  procedure estimates). Comparing extrinsics
  with a drawing compares with the sensor (body) mounting. Test
  `the_plain_camera_frame_is_the_sensor_normal` pins it (< 0.1° at 10°).
- **Fixed with `p` held**: neither frame (0.06–9° from the sensor normal,
  5–29° from the lens axis), depending on the distortion. Not a reportable
  state.
- A rig with free extrinsics and a hand-eye with exact robot poses absorb
  the difference (B3's metrics), so mixing branches between a rig's
  cameras is consistent as long as each camera's record says which frame
  it is in. Comparing the two frames is the tilt itself: with a `p`-free
  fix, the lens-axis direction is `R(τ)ᵀ·z` for a `τ` the data do not
  determine.

What the export should record (not implemented): per Scheimpflug camera
`tilt_source ∈ {estimated, estimated_imprecise, prior, fixed_undetermined}`,
`frame ∈ {lens_axis, sensor_normal}`, the statistic `s` and the threshold,
the likelihood ratio, and the σ with `p` free (whether the tilt is told
apart from a decentring at all).

## F. The procedure

**F1 — candidate rules over every draw of the sweep** (ratio: worst metric
of the picked fit over M1's in the same draw):

| rule | tilt | ratio > 1.5 | 99th pct | max | estimated |
|---|---|---:|---:|---:|---:|
| always estimate (M1) | any | 0 | 1.00 | 1.00 | 100 % |
| always fix, `p` free (M0p) | 2° / 20° | 29 / 71 % | 62 / 359 | 166 / 915 | 0 |
| `s ≤ 0.5°` → M1, else M0 | 5° / 20° | 24 / 28 % | 8.9 / 26 | 13 / 47 | 63–65 % |
| `s ≤ 0.5°` → M1, else M0p | 5° / 20° | 5.3 / 13 % | 2.2 / 7.5 | 4.1 / 21 | 63–65 % |
| `s ≤ 0.25°`, else M0p, LR guard | 0–20° | ≤ 6.6 % | ≤ 2.4 | 4.9 | 49–70 % |
| **`s ≤ 0.5°`, else M0p, LR guard** | 0–20° | ≤ 5.3 % | ≤ 1.9 | 4.9 | 63–76 % |
| `s ≤ 1°`, else M0p, LR guard | 0–20° | ≤ 2.5 % | ≤ 1.7 | 2.6 | 75–84 % |
| σ(M2) ≤ 0.5° → M2, else M0p, LR guard | 0–20° | 7–14 % | 2.5–4.8 | 21 | 18–68 % |
| LR guard alone | 2° | 10 % | 5.1 | 21 | 26 % |

LR guard: `(cost(M0p) − cost(M2))·dof / cost(M2) > 9.21` (χ²₂ at 99 %) keeps
the tilted fit. With θ = 0.5° and the guard, 8 of 1600 draws exceed 2×:
seven with 8 central views (four at 70–90°, −0.02…−0.05, tilt 10–20°,
`s` 0.8–1.1°, ratio 2.6–4.9; three at 2.05–2.16), one with 20 views at
0.5 px (90°, −0.02, 5°, 2.9). There the training data do
not show the misfit that the full-image held-out views do; no statistic of
the training data can, so a prior or views reaching the corners is the
remedy. Downstream, "always estimate" is never worse **on a lens with no
decentring**; the procedure trades at most this much prediction for a
frame claim the data support. θ is the precision asked of the lens-axis
frame: 0.5° is proposed as the default (1° gives the smaller cost, 0.25°
the stricter frame).

**F2 — end to end** (step 1 the plain camera through the pipeline's cold
route; step 2 the tilted fit seeded by the decentring identity, unbounded,
tight stop; θ = 0.5°, guard; 8 draws per cell, 32 for the real-rig-like
ones; E estimated / I estimated imprecise / Z fixed with `p` free):

| cell | E / I / Z | seed reached | `s` | ratio median / max | reported `tilt_x` error |
|---|---|---|---|---|---|
| 20 views, 35°, −0.005, tilt 0 / 5 / 20° | 7/0/1, 7/0/1, 8/0/0 | 8/8 each | 0.38–0.47° | 1.00 / 1.29 | ≤ 0.09° |
| 20 views, 50°, −0.005, tilt 0 / 5 / 20° | 0/0/8, 0/1/7, 0/4/4 | 8/8 | 0.56–0.66° | 1.03–1.40 / 2.19 | Z: −τ |
| 20 views, 90°, −0.005, tilt 0 / 5 / 20° | 0/0/8, 0/2/6, 0/8/0 | 8/8 | 1.2° | 1.00–1.18 / 1.65 | I at 20°: −0.50° |
| 20 views, −0.02, −0.15 (all fields, tilts) | all E | 8/8 | 0.006–0.27° | 1.00 / 1.00 | ≤ 0.17° |
| 8 central, 90°, −0.02, tilt 20° | 0/6/2 | 8/8 | 1.2° | 1.00 / 2.68 | −1.5° |
| 8 central, −0.005 (35°, 50°, 90°; each tilt) | 6–8 Z of 8, rest I | 8/8 | 1.4–5.7° | 0.89–1.11 / 1.64 | Z: −τ |
| 8 central, −0.02, 50° and 90°, tilt 0 / 5° | 7–8 Z of 8 | 8/8 | 0.55–1.3° | 1.00–1.28 / 1.98 | Z: −τ |
| 8 central, 50°, −0.02, tilt 20° | 6/1/1 | 8/8 | 0.48° | 1.00 / 1.00 | −0.09° |
| decentred, untilted, 50°, −0.05, `p1` 5e-4 | 8/0/0 | 8/8 | 0.032° | 1.00 / 1.00 | **0.19°** |
| decentred, untilted, 70°, −0.005, `p1` 1e-4 | 0/0/8 | 8/8 | 0.93° | 1.07 / 1.09 | 0 (exact) |
| real-rig-like (720×540, 35°, −0.07, (−5°, 0.3°), 22 views) | 32/0/0 | 32/32 | 0.034° | 1.00 / 1.00 | −0.008° |
| real-rig-like, 8 central views | 32/0/0 | 32/32 | 0.14° | 1.00 / 1.00 | −0.032° |

- **The decentring seed** reaches the reference minimum in every draw
  (512/512), tilts up to 20° and fisheye-like distortion included: the
  plain fit carries the tilt in `p1, p2` to first order, so it is a
  cold-start seed with no tilt sweep (**measured**; test
  `decentring_seed_recovers_the_tilt`, within 15 % at (5°, −3°)).
- **The cold plain route fails** (Zhang's linear init) at 35°, −0.15 in
  every draw (corner distortion 15 % defeats the undistorted homographies);
  the procedure then needs an intrinsics seed for step 1, as the seeded
  route already requires (**measured**, 24/24 draws in that camera).
- **A decentred lens with no tilt** is reported as tilted by `−p1/k1` =
  0.19° (6σ) when the distortion determines it, and as untilted with the
  decentring in `p1` when it does not. The first is the known limit
  (tilt-identifiability.md): no statistic of the data separates them
  except σ(M2); a prior with `p` free does.

## Hypotheses and open items

- **Real data**: the procedure's branch per camera and the held-out error
  against `Estimate` on the private Scheimpflug rigs, the decentring seed
  against the seeded route there. Not measured; the gate of the ADR.
- **Rig re-decision**: a rig's joint stage adds views but no information
  about the tilt beyond each camera's distortion (the extrinsic rotation
  absorbs it), so deciding per camera is expected to suffice (hypothesis).
- **θ per axis**: the rule uses the worse direction; a tilt determined on
  one axis and not the other (one-sided coverage) would fix both.
- The per-view error model (a quadratic field per view) is one choice; the
  scaled σ's 0.71–1.57 ratio to the Monte-Carlo σ is measured only on it.
- Robust losses, `k2`-dominated and rational lenses: not in the sweep.

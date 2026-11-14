# Tilt from focus — can marker blur determine the Scheimpflug tilt?

Family: single-camera intrinsics on a tilted sensor
([scheimpflug-intrinsics.md](scheimpflug-intrinsics.md)). Today the tilt is
determined only through lens distortion: without it a tilted pinhole is a
rotated plain pinhole, `K·H(τ) = K'(τ)·R(τ)` (the synthetic
tilt-identifiability study, on its own branch). Focus
is tied to the lens axis too. This note derives what the blur of each
marker says about the lens, tests it on exact synthetic optics and on the
six ring-grid cameras (an e-ink display target, per-marker blur from the
photometric centre fit), and asks whether it can check, constrain or
replace the distortion route. Dev-facing; aggregate numbers only.

**Derived** marks algebra (pinned by a test where it says so),
**measured** a number the harness prints, **hypothesis** what neither
shows.

Harness: bench example `scheimpflug_model_study`, `focus.rs` (model, fit,
inversion, the exact paraxial thick lens; unit-tested), `focus_study.rs`
(`--parts focus` real data, `--parts focussweep` synthetic) and
`centroid.rs` (`--parts centroid`, the signed-defocus residual test):

```text
cargo run --release -p vision-calibration-bench --features tier-b \
    --example scheimpflug_model_study -- \
    --registry crates/vision-calibration-bench/registry/private.json \
    --out <dir> --parts focus,centroid scheimpflug_rig_ringgrid_cam0 … cam5
cargo run … -- --registry <any> --out <dir> --parts focussweep --draws 40
cargo run … -- --registry <any> --out <dir> --parts synth,centroid \
    synth_plain_cam1 … synth_plain_cam6
```

## Short answers

1. **The relation** (derived). In the frame of the plain pinhole `K'`
   (its `z` axis the sensor normal), a marker at `X = z·(x, y, 1)` has
   blur `σ² = σ0² + κ²·(1/z − π·(x, y, 1))²`, `π·X = 1` the plane of sharp
   focus (PoF), `κ = D·f'/4` (`D` the entrance pupil, `f'` in px). The lens
   axis is `a ∝ π + e_z/v0`, `v0 = f'·pitch`: the tilt **direction** needs
   no metric datum, its **size** needs one (the pixel pitch, or the focal
   length in mm) — and, for a real (thick) lens, the pupil magnification
   `m_p`: the comparison with the distortion route determines `pitch/m_p`,
   not the pitch.
2. **Synthetic** (measured): from exact thick-lens blur the tilt comes
   back with ≤ 0.01° bias and 0.013° scatter at the studied rig's
   settings (≤ 0.09° bias at 15°), robust to magnification, base blur,
   the PSF combination and depth range; it fails only when the depth of field
   covers the scene (f/16) and degrades by the predicted factor when
   `m_p ≠ 1` (the implied pitch is `pitch/m_p` to 0.3 %).
3. **Real data** (measured): the 5-parameter focus model explains 91–96 %
   of the blur variance (a free 10-parameter quadratic 92–96 %), and the
   PoF agrees across the six cameras (`π_y` −14.7 … −16.0 /m, 78–79° to
   the sensor). The tilt's **sign agrees** with the distortion route on all
   six. The **implied `pitch/m_p` is 4.75–6.14 µm** (mean 5.29, 11 % sd,
   χ² 92 on 5 dof against the fold jackknife): plausible, but not one value
   for six units of one design, and at a common pitch the focus tilt and
   the routed tilt **anti-correlate** across cameras (−0.84).
4. **Verdict**: blur determines the PoF well and the tilt direction
   without any datum; it is (a) a usable **independent check** of the tilt
   direction and, given `pitch/m_p`, of its size at the 0.5° level; (b) a
   **prior** only once `pitch/m_p` is known from a datasheet or a reference
   unit, and with the unexplained ±0.5° unit-to-unit disagreement as its
   σ; (c) **not a replacement**. Details and what would settle the open
   points below.
5. **Hypothesis (i) tested** ([tilt-bias.md](tilt-bias.md)): refitting
   the route jointly with a defocus/depth law moves every camera's tilt
   towards its focus tilt (correlation with the gap +0.74 … +0.95) but
   closes only 13–51 % of the τx gap (by the law) and none of the
   direction (τy) gap; the focus tilt stays rejected by the distortion
   data. Partly supported, not the explanation.

## Derivation

**Thin lens, `K'` frame.** The projection centre `O` is the lens centre,
the sensor the plane `z = −v0` (normal `e_z`), the lens axis `a`
(unit), focal length `F`. A point `X` at axial depth `z_a = a·X` focuses
at `v` with `1/z_a + 1/v = 1/F`; its chief ray meets the sensor at axial
distance `v_s = v0·z_a/z` (`z = e_z·X`). The cone from the aperture
(diameter `D`) to the image point has, at `v_s`, the diameter
`c = D·|v_s − v|/v = D·v_s·|1/v − 1/v_s|`. Substituting,

```text
1/v − 1/v_s = (1/z_a)·(X·(a/F − e_z/v0) − 1)   ⇒   c = D·v0·|1 − π·X| / z,
π = a/F − e_z/v0 .
```

So the in-focus points form the plane `π·X = 1` (the PoF), and in pixels
`c/pitch = D·f'·|1/z − π·(x, y, 1)|` with `f' = v0/pitch` (**derived**).
The PoF contains the line `{a·X = 0, z = −v0}`: lens plane, sensor plane
and PoF meet in one line (Scheimpflug). Inverting: `a = F·(π + e_z/v0)`,
so `a ∝ π + e_z/v0` and `1/F = |π + e_z/v0|`. Test
`inversion_thin_exact_thick_scaled` recovers the tilt exactly (1e-12) from
the conjugate PoF.

**Any lens, object space.** The rays of `X` that pass the entrance pupil
(diameter `D`, at the projection centre) are imaged where they cross the
PoF, and a PoF point is imaged at its pinhole projection. So the spot is
the pinhole image of the pupil's shadow cast through `X` onto the PoF: an
angular diameter `D·|1/z − 1/z*|` along the ray, `z*` the PoF's depth
there. The blur model is therefore the same for a thick lens, with `D` the
*entrance* pupil and `π` the true PoF (**derived**; the exact simulator
`focus::Lens::spot_sigma` does this ray by ray, and test
`spot_follows_the_defocus_model` agrees with the formula within 3 %).
What a thick lens changes is where the PoF lies relative to the sensor.

**Thick lens.** With pupil magnification `m_p` (exit/entrance pupil),
image-side chief-ray slopes are object-side slopes over `m_p`; the
chief-ray map is `K_s·R(τ_s)·diag(1/m_p, 1/m_p, 1)`, whose RQ
decomposition gives (**derived**, tests
`calibrated_tilt_is_physical_over_pupil_magnification` and
`inversion_thin_exact_thick_scaled`)

```text
tan τ_d = tan τ_s / m_p                       (what distortion measures)
tan τ_f = tan τ_d · (m_p + μ) / (1 + μ + (1 − m_p)·tan² τ_d)
```

`τ_s` the sensor's physical tilt to the lens axis, `τ_d` the library's
(distortion-defined) tilt, `τ_f` the thin-lens inversion of the true PoF
with the pinhole's `v0 = f'·pitch`, `μ` the axial magnification at focus
(0.02–0.04 here). At small `μ`, `τ_f ≈ τ_s`: focus measures the physical
tilt, distortion the tilt over `m_p`. The pitch at which `τ_f = τ_d` is
the true pitch divided by `m_p` (the `μ` terms cancel to first order;
measured ratios 1.427, 1.175, 0.833, 0.714 for `m_p` 0.7, 0.85, 1.2,
1.4).

**What is determined, and from what.**

| quantity | needs | here |
|---|---|---|
| PoF `π` (3 numbers, 1/m), axial focus distance, PoF angle | blur + the `K'` poses | measured, all six cameras |
| tilt direction (azimuth of `π_xy`) and sign | nothing more | measured |
| entrance-pupil diameter `D = 4κ/f'` | the detector's σ being the disc's per-axis σ (`R/2`) | 4.75–5.00 mm (hypothesis on the factor) |
| tilt size `tan τ` | `v0 = f'·pitch` (or `F` via `1/F = |π + e_z/v0|`) and `m_p` | `pitch/m_p` from matching the distortion tilt |
| f-number | `F` and `D` | `N ≈ 1.2` at 5.5 µm, 0.5 at 2.2 µm |

**Assumptions and what each changes.**

| assumption | if wrong | evidence |
|---|---|---|
| thin lens (`m_p = 1`) | blur model unchanged; the inversion scales `tan τ` by `(m_p + μ)/(1 + μ)` | derived; synthetic rows `m_p` |
| Gaussian σ = disc σ (`c/4`) | rescales `κ` only; the PoF is the zero set of the defocus and does not move under any smooth positive factor (also vignetting's cat's-eye pupil) | derived; synthetic "combination exponent" |
| base blur in quadrature | `σ0` absorbs it; an exponent 1.5 or 3 moves the tilt 0.02–0.03° | measured |
| unsigned defocus | spherical aberration would make the two sides of focus differ | real: before − beyond ≤ 0.04 px (\|z\| ≤ 2.1) |
| a plane of focus (no field curvature) | `π` is the best plane over the covered field: curvature `c` shifts `π_xy` by `2c·(axis offset)`; detected by a curvature term | synthetic `c` = 3 /m: −0.06°, pitch −1.2 %; real `c` 0.7–5.7 /m (\|z\| 4–31) moves the implied pitch by 0–6 % |
| independent per-marker blur errors | the stated σ is short | real: fold jackknife/stated 0.8–3 for `π_y`; synthetic per-view 5 % jitter: scatter 4.8× stated. Use the view jackknife |
| the `K'` frame is right | the pitch-free part is gauge-invariant (`K'` poses do not depend on the tilt) | derived |

**Data requirement.** Blur well above `σ0` over the covered field (depth of
field much smaller than the depth range seen); depth variation *within*
views, which separates `κ` from `π_xy` (one frontal distance: σ 0.16°
instead of 0.013°); coverage of the image rows the PoF crosses. The capture
studied here (boards at 0.16–0.35 m, f/1.2-like aperture, blur 0.6–8 px)
is well inside this.

## Synthetic validation (measured)

Exact paraxial thick lens (`focus::Lens`): the PoF as the conjugate of the
sensor plane, the spot as the pupil's shadow on it (64 pupil samples),
720×540, `f' = 1145` px, tilt −5.16°, `F = 6` mm equivalent, f/1.22, focus
0.23 m, 20 views of a 14×15 ring grid at 0.17–0.33 m with ±0.45 rad board
tilt, base blur 0.72 px, noise 4 % + 0.06 px, the detector's 0.2 px floor;
40 draws of noise and view geometry per cell; the true pitch in the
inversion. `pitch*/pitch` is the pitch that matches the distortion tilt
over the true one.

| cell | τx error ° (mean ± sd) | stated σ ° | scatter/stated | pitch*/pitch | EV | \|z\| curvature |
|---|---|---:|---:|---|---:|---:|
| baseline | −0.003 ± 0.013 | 0.011 | 1.13 | 0.999 ± 0.003 | 0.99 | 2.7 |
| null, τ = 0 | −0.001 ± 0.005 (τy 0.000 ± 0.003) | 0.004 | 1.4 | — | 0.97 | 0.9 |
| tilt −2° / −10° / −15° | −0.002 / −0.029 / −0.088 (sd 0.006–0.094) | 0.006–0.065 | 1.0–1.4 | 0.994–0.999 | 0.99 | 1–11 |
| tilt (0°, 15°) | τy +0.135 ± 0.168 | 0.012 | 1.3 | 0.991 | 0.99 | 28 |
| tilt (−10°, 10°) | −0.060 ± 0.066, τy +0.061 ± 0.068 | 0.047 | 1.4 | 0.994 | 0.99 | 16 |
| F 4 / 12 / 25 mm (μ 0.018–0.12) | −0.002 … −0.024 (sd 0.014–0.026) | 0.013–0.018 | 1.1–1.4 | 0.995–1.000 | 0.98–0.99 | 1–3 |
| `m_p` 0.7 / 0.85 / 1.2 / 1.4 | +2.15 / +0.88 / −0.84 / −1.44 (sd 0.012) | 0.011 | 0.9–1.1 | 1.427 / 1.175 / 0.833 / 0.714 | 0.99 | 2–4 |
| f/2 / f/4 / f/8 | ±0.01 (sd 0.021 / 0.047 / 0.130) | 0.016–0.094 | 1.3–1.4 | 1.00 ± 0.004–0.026 | 0.99 / 0.95 / 0.72 | 1–2 |
| f/16 (depth of field ≥ scene) | +0.06 ± 0.44, 2 of 40 fail | 0.32 | 1.4 | 1.02 ± 0.09 | 0.19 | 0.9 |
| depth 0.23–0.27 / 0.245–0.255 m | 0.00 ± 0.039 / 0.055 | 0.030–0.041 | 1.3 | 1.00 ± 0.01 | 0.99 | 3 |
| one distance, frontal | +0.04 ± 0.16 | 0.15 | 1.1 | 1.01 ± 0.03 | 0.99 | 3 |
| 3 / 5 / 40 views | sd 0.068 / 0.034 / 0.009 | | 1.1–1.2 | 1.00 | 0.99 | 1–4 |
| noise 2 % / 10 % / 20 % | sd 0.005 / 0.023 / 0.047 | | 1.0 | 1.00 | 1.00 / 0.96 / 0.86 | 1–6 |
| + per-view 5 % blur jitter | −0.015 ± 0.078 | 0.016 | **4.8** | 1.00 ± 0.015 | 0.98 | 2.7 |
| base blur 0.3 / 1.2 px | sd 0.011 / 0.016 | | 1.1–1.2 | 1.00 | 0.99 | 2–3 |
| field curvature 1 / 3 / 6 /m (wrong model) | −0.021 / −0.062 / −0.132 (sd 0.02–0.09) | 0.012–0.024 | 1.7–3.6 | 0.996 / 0.988 / 0.975 | 0.97–0.99 | **21 / 58 / 113** |
| combination exponent 1.5 / 3 (wrong model) | −0.030 / +0.016 | 0.012 | 1.1–1.3 | 0.994 / 1.003 | 0.99 | 1.2 / 4.5 |
| `m_p` 0.7, tilt (−10°, 5°) (τ_d −14.1°, 7.0°) | +4.21, τy −2.15 | 0.036 | 1.5 | 1.428 | 0.99 | 13 |

Readings: the tilt is recovered from blur to a few hundredths of a degree
wherever the defocus exceeds the base blur, at every tilt and both axes;
the residual bias at 10–15° (−0.03 to −0.09°) is the spot's
second-order shape (the curvature statistic flags it). The direction
(azimuth of `π_xy`) is unbiased in every cell, `m_p` included. Field
curvature is the model error that matters, and the fit reveals it: its
`|z|` is 21–113 against 1–3 for a flat field. The stated σ is honest
(1.0–1.4) for independent blur errors and 4.8× short with view-correlated
ones.

## Real data (measured)

Six cameras, 17–22 views, 786–1206 markers each, blur 0.6–8.1 px, marker
depth 0.16–0.35 m. Each camera's route (Brown-Conrady `k1, k2` + tilt)
gives the poses; `X' = R(τ)·X` puts them in the `K'` frame. Folds: 4 by
view; "jackknife" is the delete-a-fold jackknife with each fold's own
route.

**Fit quality and the PoF.** Explained variance of σ (EV), robust fits on
σ; the free quadratic is the 10-coefficient quadratic form in
`(1/z, x, y, 1)` the focus model is a rank-one case of.

| camera | π (K', 1/m) | PoF angle to sensor | κ px·m | σ0 px | EV focus (5) | + curvature (6) | free (10) | held-out focus / free | curvature c /m |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| cam0 | (+0.51, −14.75, +2.98) | 78.6° | 1.39 | 0.77 | 0.958 | 0.959 | 0.962 | 0.957 / 0.961 | +0.7 ± 0.2 |
| cam1 | (−0.63, −15.98, +3.00) | 79.4° | 1.41 | 0.67 | 0.946 | 0.958 | 0.948 | 0.946 / 0.948 | +4.2 ± 0.2 |
| cam2 | (+1.30, −15.83, +2.99) | 79.4° | 1.43 | 0.67 | 0.959 | 0.967 | 0.961 | 0.958 / 0.961 | +3.2 ± 0.2 |
| cam3 | (+0.32, −14.70, +3.14) | 77.9° | 1.39 | 0.71 | 0.951 | 0.963 | 0.956 | 0.950 / 0.955 | +3.3 ± 0.2 |
| cam4 | (−0.28, −15.97, +3.11) | 79.0° | 1.35 | 0.80 | 0.911 | 0.953 | 0.924 | 0.910 / 0.923 | +5.7 ± 0.2 |
| cam5 | (−2.03, −15.09, +3.30) | 77.8° | 1.39 | 0.76 | 0.922 | 0.956 | 0.927 | 0.921 / 0.926 | +5.3 ± 0.2 |

Fold jackknife of `π`: 0.02–0.07 (x), 0.06–0.32 (y), 0.003–0.02 (z) /m,
0.8–3× the stated σ. In the routed (lens) frame the PoF is `≈ (π_x, −15,
+4.4)`: axial focus 0.22–0.23 m. Residual structure: robust scale 0.10–0.16
px; image-thirds means up to ±0.3 px, top rows positive and bottom corners
negative — the pattern of a field curvature, which a curvature term takes
(EV +0.01–0.04); depth terciles ≤ 0.05 px; before − beyond the PoF
−0.04 … +0.03 px. The physical restriction (5 against 10 parameters)
costs ≤ 1.3 points of EV; held-out EV equals training EV.

**The tilt as a function of the unknown pitch** (thin lens; ± the stated
σ, the fold jackknife at 5.5 µm is 0.0003–0.0020 rad):

| camera | 2.2 µm | 3.45 µm | 4.5 µm | 5.5 µm | 6.9 µm | routed τ (stated σ) |
|---|---|---|---|---|---|---|
| cam0 | −0.036 | −0.057 | −0.074 | −0.090 | −0.112 | −0.0996, +0.0038 (0.0014) |
| cam1 | −0.040 | −0.062 | −0.081 | −0.098 | −0.122 | −0.0882, −0.0006 (0.0015) |
| cam2 | −0.040 | −0.062 | −0.080 | −0.098 | −0.122 | −0.0872, +0.0021 (0.0016) |
| cam3 | −0.037 | −0.057 | −0.074 | −0.090 | −0.112 | −0.0954, +0.0039 (0.0018) |
| cam4 | −0.040 | −0.062 | −0.080 | −0.098 | −0.122 | −0.0847, −0.0015 (0.0012) |
| cam5 | −0.038 | −0.059 | −0.077 | −0.094 | −0.117 | −0.0855, −0.0101 (0.0015) |
| implied F (thin) | 2.5 mm | 3.9 mm | 5.0 mm | 6.1 mm | 7.6 mm | |
| implied f-number | 0.5 | 0.8 | 1.0–1.1 | 1.2–1.3 | 1.5–1.6 | |

(τx in rad; τy from focus is −0.010 … +0.016 at these pitches.) The
f-number column rests on the detector's σ being the disc's per-axis σ: if
it is, pitches below about 4 µm imply a lens faster than f/1, which no
such lens is (hypothesis-grade bound).

**The test that could have contradicted the idea.**

| camera | implied `pitch/m_p` µm (jackknife) | Δ azimuth focus − route (jackknife) | at 5.29 µm: focus τx − routed τx | focus τy − routed τy |
|---|---|---|---:|---:|
| cam0 | 6.14 ± 0.15 | +4.2° ± 0.2 | +0.0132 | −0.0068 |
| cam1 | 4.94 ± 0.07 | −2.6° ± 0.4 | −0.0061 | +0.0043 |
| cam2 | 4.90 ± 0.15 | +6.0° ± 0.3 | −0.0070 | −0.0098 |
| cam3 | 5.85 ± 0.12 | +3.6° ± 0.2 | +0.0086 | −0.0058 |
| cam4 | 4.75 ± 0.17 | −2.1° ± 0.6 | −0.0095 | +0.0032 |
| cam5 | 5.17 ± 0.08 | −14.4° ± 0.4 | −0.0046 | +0.0222 |

- **Sign: agrees** on all six (`π_y < 0` ⇔ `τx < 0`), with no datum.
- **Azimuth: agrees to 2–14°**, but the differences are 3.5–36 times
  their jackknife: the focus τy and the routed τy differ by 0.2–1.3°
  (σ ≈ 0.03° for either).
- **Implied pitch: plausible, not consistent.** 4.75–6.14 µm, mean 5.29,
  sd 0.57 (11 %), χ² 92 on 5 dof. A common `pitch/m_p` is expected for six
  units of one sensor and lens design (hypothesis: same lens model, fixed
  `m_p`). The curvature fits move each camera's value by 0–6 %, not
  enough to close it.
- **At a common pitch** the focus tilts are tighter across the six units
  (τx −0.0910, sd 0.0038) than the routed ones (−0.0901, sd 0.0060), and the
  two **anti-correlate** (−0.84): where the route says more tilt, focus
  says less. Two estimates of one physical tilt with independent errors
  would correlate positively or not at all.

So the idea survives the sign test and fails the consistency test at the
0.5° level: one of the two routes carries a unit-dependent systematic of
that size. Which one is open. **Hypotheses**, each with what would decide
it: (i) the routed tilt absorbs the defocus- and depth-coupled
displacement of the parametric note (it is mirror-symmetric about the tilt
plane, as the distortion centre's offset is) — the anti-correlation points
this way; refitting the route with that term and re-running this
comparison decides it (done: partly, see [tilt-bias.md](tilt-bias.md)); (ii) a decentred or tilted lens element tilts the
image field without moving the distortion centre the same way — a
different lens or a rotated lens (same unit) decides it; (iii) the six
lenses differ in `m_p` or are focused differently (internal focusing
changes `m_p`) — the datasheet decides it.

**Accuracy against the distortion route on this rig.** Statistical: focus
τx jackknife 0.0003–0.0020 rad, routed 0.0011–0.0018 rad, comparable.
Systematic: the focus route carries the unknown scale `pitch/m_p` (a
factor, not an offset) and an unexplained ±0.5° unit-to-unit
disagreement; the distortion route carries the latter equally, by this
data.

## Signed defocus and the residual (measured)

Hypothesis tested (coordinator's add-on): the defocus-coupled
displacement of the parametric note is an optical centroid shift
proportional to *signed* defocus — an asymmetrically illuminated pupil
moves the spot's centroid by `g·(a + B·n)`, `g = κ·(1/z − π·(x, y, 1))`
(px; the spot radius is `2|g|`), `n` the normalized image position (`n = 1`
at the side edge, 360 px), so `(a + B·n)/2` is the pupil-centroid offset
in pupil radii. Linearized held-out fits about the route, exactly the
`field` harness; `g` from each camera's PoF fitted to the full-data blur
(a covariate, never the residual). Held-out RMS px, Δ variance against
the route (0.1426):

| term (per camera, camera free) | ring-grid | Δ | null (plain synthetic) |
|---|---:|---:|---:|
| geometric blur (σ_geo − median) × deg 0–3 (parametric note) | 0.0986 | −52 % | +0.3 % |
| geometric blur × deg 0–3 + depth × deg 1–4 (parametric note) | 0.0833 | −66 % | +1.2 % |
| signed g × deg 0–1 (`a + B·n`) / 0–2 / 0–3 | 0.1187 / 0.1042 / 0.0973 | −31 / −47 / −53 % | +0.1–0.3 % |
| \|g\| × deg 0–1 / 0–3 | 0.1129 / 0.0983 | −37 / −53 % | +0.1–0.3 % |
| g² × deg 0–1 / 0–3 | 0.1192 / 0.1088 | −30 / −42 % | +0.1–0.5 % |
| depth (z̄/z − 1) × deg 0–1 / 0–3 | 0.1366 / 0.1196 | −8 / −30 % | +0.1–0.3 % |
| signed g × deg 0–3 + \|g\| × deg 0–3 | **0.0746** | −73 % | +0.9 % |
| depth × deg 0–3 + \|g\| × deg 0–3 | 0.0818 | −67 % | +0.9 % |
| signed g × deg 0–3 + geometric blur + depth | 0.0750 | −72 % | +1.6 % |
| *next to the image field φ ≤ 6 (0.1041), Δ vs φ:* | | | |
| φ + signed g × deg 0–1 / depth × deg 0–1 / \|g\| × deg 0–1 | 0.0953 / 0.0948 / 0.0890 | −16 / −17 / −27 % | +0.1 % |
| φ + signed g × deg 0–3 / depth × deg 0–3 / \|g\| × deg 0–3 | 0.0868 / 0.0867 / 0.0821 | −31 / −31 / −38 % | +0.3 % |
| φ + signed g + \|g\| (deg 0–3) / φ + depth + \|g\| | 0.0723 / 0.0723 | −52 / −52 % | +0.9 % |
| signed g × deg 0–1, shared by the six / deg 0–3 shared | 0.1249 / 0.1104 | −23 / −40 % | +0.0 % |
| \|g\| × deg 0–3, shared | 0.1043 | −47 % | +0.0 % |

**The sign is not separately testable from positions** (derived): at a
fixed image position `g` is affine in `1/z`, so `g·f(n) = κ·f(n)/z −
κ·f(n)/z*(n)` — a depth-coupled term plus an image-fixed field tied to the
PoF. The rows confirm it: alone, signed `g` beats depth at equal degree
(0.0973 against 0.1196) only by that image field; next to a free image
field the two are identical to 0.0005 px. Also `a·κ/z` is exactly a pose
translation (a gauge) and the radial `z`-term times `κ/z` is a translation
along the axis, so `a` and `Re B_z` are mostly what the gauge leaves.

**The unsigned blur is a separate term** (measured): `|g|` adds on top of
depth and of the image field (φ + depth −31 % → φ + depth + |g| −52 %),
and `|g|` from the fitted PoF reproduces the empirical "geometric blur"
covariate (0.0983 against 0.0986).

**Coefficients of `signed g × deg 0–1`** (per camera, ± stated σ):
`a_y` +0.57 … +1.72 (σ 0.11–0.29), across cameras +1.23 ± 0.48, fold
jackknife/σ 1.85; `Re z` −1.50 … +0.79, sign not stable; `Im z`, `Im z̄`
consistent with zero; `Re z̄` +0.06 … +0.24 (σ 0.003–0.008, jackknife/σ
1.95), the same sign on all six, shared value +0.109 ± 0.007. Read
literally, `|a|/2 = 0.61` is a pupil-centroid offset of 61 % of the pupil
radius on axis — implausible for vignetting, which also predicts `a ≈ 0`
here (the lens axis meets the sensor near the image centre) and a radial
`Re z`, not the `Re z̄` that is stable. So the radial-vignetting reading is
**not supported**. The stable `Re z̄·g` is, by the identity above, a
depth-dependent anamorphic magnification; one optical reading
(**hypothesis**): sagittal and tangential entrance pupils at different
axial positions (an astigmatic pupil of the tilted lens): a pupil at
axial offset `δ` moves a point by `−(pixel offset)·δ/z`, so `Re z̄ = c`
means `δ_x − δ_y = 2c·κ/360 px ≈ 0.85–1.2` mm (indicative: the
coefficient also carries the PoF-tied image part). A capture that changes focus or
aperture at fixed poses separates a defocus-coupled shift (it moves with
`g`) from a pupil (depth) term (it does not); views through focus at one
image position cannot.

## Verdict

- **Can blur determine the tilt?** It determines the plane of sharp focus
  to a few per cent and the tilt *direction* with no datum (measured, all
  six cameras, sign agreeing with the distortion route). The tilt's size
  needs `v0 = f'·pitch` and the pupil magnification; without them the
  comparison yields only `pitch/m_p` (derived; synthetic to 0.3 %).
- **Accuracy**: synthetic 0.013° at this rig's settings (0.005–0.05° over
  the sweep, 0.13–0.44° at f/8–f/16); real, per camera 0.02–0.11°
  statistical, but a unit-to-unit systematic of ±0.5° between the focus and
  distortion routes that neither model explains (measured).
- **(a) Independent check — yes**, now, for the direction, and for the
  size once `pitch/m_p` is fixed (the check that found the ±0.5°
  disagreement). **(b) Prior — conditional**: with `pitch/m_p` from a
  datasheet or a reference unit and a σ of 0.5° until the disagreement is
  explained. **(c) Replacement — no**: it needs `m_p` (or a calibrated
  reference) and has a systematic of its own size.
- **Required capture**: blur well above the in-focus blur over the field
  (a wide aperture, depth of field small against the depth range), depth
  variation within views, coverage of the rows the PoF crosses — what this
  capture has.

## What would settle the open points

From the maintainer: the sensor's pixel pitch, the lens's focal length and
f-number, and its pupil magnification (or the exit-pupil position) — with
them the focus route gives an absolute tilt to compare per unit. Captures:
the same poses at two apertures and at two focus settings (separates a
defocus-coupled centroid shift from a depth/pupil term and the detector's
blur response); one unit with the lens rotated 180° in its mount (a lens
element's field tilt rotates with it, a sensor tilt does not). The analysis
without new data (hypothesis (i)) is in [tilt-bias.md](tilt-bias.md): the
coupling accounts for part of the τx disagreement, not for the direction.

Library sketch: backlog `CAL-TILT-FOCUS`.

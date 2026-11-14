# Ring-grid ellipse-center bias — proof pack

Family: coded ring-grid targets on the seeded Scheimpflug intrinsics route
(`ScheimpflugIntrinsicsProblem`; detector `vision_calibration_detect::RinggridDetector`
wrapping the external `ringgrid` crate; bench accept path
`vision_calibration_bench::run::run_scheimpflug_intrinsics`).

**Finding:** the six ring-grid cameras floor at ~0.46 px mean reprojection
(committed baselines `scheimpflug_rig_ringgrid_cam*`: 0.42–0.51 px per camera), ~0.19 px
above the puzzleboard sibling on the same rig (`scheimpflug_rig_ref_cam*`, ~0.27 px).
That floor is not the projected ellipse-center bias of the ring markers: the
bias is ~0.09 px here and is removed inside the detector. The floor is
per-detection localization scatter under defocus blur, and the pipeline applies no correction.

## Mechanism

A coded ring marker is two concentric circles (radii `r_in`, `r_out`) on the
board plane. Setting lens distortion aside, the board plane → pixel map is a
single homography

```
H = K · H_tilt · [r1 r2 t]
```

with intrinsics `K`, the Scheimpflug tilt homography `H_tilt` (tilt acts on
normalized coordinates — ADR 0011), and the planar pose `[r1 r2 t]` from
`cam_se3_target`. A circle with conic matrix `C` images to

```
C' = H⁻ᵀ C H⁻¹.
```

The **ellipse center** of `C'` —
`center = −[[C'₀₀, C'₀₁],[C'₀₁, C'₁₁]]⁻¹ · [C'₀₂, C'₁₂]ᵀ` — is *not*
`H·center` of the circle under a genuinely projective (non-affine) `H`; the
offset is the ellipse-center bias. It vanishes for an affine `H` (bottom row
`[0 0 1]`) and grows roughly as `(marker radius in px)² × |bottom row of H|`.

Lens distortion displaces the observed edge, but to first order that
displacement is common to the inner and outer edges *and* to the model's
reprojection of the center, so it cancels from the residual; the remainder is
second order in the distortion gradient across the ~4 px marker.

## Closed-form magnitude

`crates/vision-calibration-bench/src/ringgrid_bias.rs` holds the conic-center
math: `circle_conic`, `project_conic` (`C' = H⁻ᵀ C H⁻¹`), `conic_center`, and
`predicted_center_bias(H, cx, cy, r) = center(H⁻ᵀ C H⁻¹) − H·(cx, cy)`.
`diagnose` evaluates it per (view, marker) from an exported model + pose,
using the no-distortion homography above.

On the six private ring-grid cameras, with the calibrated model and poses
(outer radius 4.8 mm, inner 3.2 mm; measured with `ringgrid` 0.7), the
predicted single-conic (outer-edge) bias is:

| camera | n_obs | bias mean | median | p95 | max (px) |
|--------|------:|----------:|-------:|----:|---------:|
| cam0 | 1196 | 0.107 | 0.090 | 0.273 | 0.420 |
| cam1 | 1041 | 0.109 | 0.092 | 0.294 | 0.434 |
| cam2 |  576 | 0.094 | 0.092 | 0.154 | 0.241 |
| cam3 | 1193 | 0.114 | 0.099 | 0.300 | 0.452 |
| cam4 |  746 | 0.091 | 0.089 | 0.159 | 0.232 |
| cam5 |  572 | 0.084 | 0.087 | 0.135 | 0.155 |

~0.09 px mean (max ≤ 0.45 px). The measured image scale is ≈ 3.4–3.5 px/mm
(`diagnose floor`), so the 4.8 mm outer radius is ≈ 16.6 px, not a few
pixels; the bias is small because the board plane is only mildly foreshortened
and it scales with the square of the pixel radius times the foreshortening.
That is well below both the floor and the gap to the puzzleboard sibling, so
even uncorrected it could not explain the gap.

## Why the pipeline applies no correction

The `ringgrid` detector's default `CircleRefinementMethod::ProjectiveCenter`
removes the *projective* part of this bias at the observation level: it fits
the inner and outer edge conics and recovers their common projected center
with an intrinsics-free two-conic pencil (Wang et al. 2019) that is exact
under a homography — it cancels exactly the `C'`-center-vs-`H·center` offset
above, for perspective and Scheimpflug tilt alike. `RinggridDetector` runs
the detector with that default.

A model-based re-correction in the pipeline would therefore either
double-correct the removed bias (strictly harmful) or reproduce the
detector's correction, whose only unmodeled remainder is the second-order
distortion term (negligible here). The conic-center math ships as a tested
diagnostic module, not a pipeline stage.

## Evidence

`diagnose` regresses the calibration residual `r = observed − projected` on
the predicted bias `b` (outer conic). If the detector reported *naive*
ellipse centers, `r` would contain `b` and the coefficient `Σ(r·b)/Σ|b|²`
would be ≈ +1. Measured (`ringgrid` 0.7):

| camera | residual mean | median | p95 | max (px) | resid/bias coeff | mean cos | corr(bias, radius) |
|--------|-------------:|-------:|----:|---------:|-----------------:|---------:|-------------------:|
| cam0 | 0.460 | 0.361 | 1.084 | 6.270 | −0.035 | 0.019 | −0.027 |
| cam1 | 0.501 | 0.347 | 1.292 | 5.616 | +0.086 | −0.021 | +0.055 |
| cam2 | 0.485 | 0.366 | 1.325 | 5.165 | +0.003 | 0.048 | +0.065 |
| cam3 | 0.463 | 0.355 | 1.050 | 7.189 | −0.031 | 0.030 | −0.011 |
| cam4 | 0.425 | 0.350 | 0.993 | 3.563 | +0.023 | 0.086 | −0.017 |
| cam5 | 0.487 | 0.410 | 1.092 | 2.985 | −0.104 | 0.065 | +0.053 |

The residual means matched the `calib-bench accept` per-camera means of the
same run to three decimals, validating the diagnostic. The coefficient is
≈ 0 (|·| ≤ 0.104) and the mean cosine between `r` and `b` is ≈ 0
(|·| ≤ 0.086): **the predicted projective bias is absent from the residual
field**, and what remains is orthogonal noise. `corr(bias, radius) ≈ 0`
shows no peripheral-growth pattern.

Conclusion: the projective ellipse-center bias is absent from the residual
field. What the floor is made of is measured in the next section.

## What the floor is made of

`calib-bench diagnose floor --dataset scheimpflug_rig_ringgrid_cam0 …
scheimpflug_rig_ringgrid_cam5` (module `floor.rs`; needs `--features tier-b`
and the private data) measures it on the six cameras together. The smooth-fit
part is calibration-free: per view, a quartic polynomial of the board position
fitted to each image coordinate, outliers rejected, residuals scaled for the
degrees of freedom. A camera model is a smooth map and cannot absorb what
such a fit leaves. Aggregate results (5422 inlier detections, 79 views):

| quantity | value |
|----------|-------|
| image scale | 3.4–3.5 px/mm (outer radius 4.8 mm ≈ 16.6 px) |
| smooth-fit (quartic) RMS / median per camera | 0.33–0.44 / 0.22–0.30 px (pooled 0.39 / 0.26) |
| cubic RMS (convergence check) | 0.37–0.52 px (pooled 0.44) |
| calibrated RMS / median on the same points | 0.47–0.62 / 0.31–0.36 px (pooled 0.54 / 0.34) |
| smooth systematic gap, `sqrt(calib² − smooth²)` | 0.31–0.45 px |
| per-marker repeatable part (board frame) | 22 µm of 105 µm total, 4.4 % of variance |
| split-half correlation of marker offsets | 0.29 |
| neighbour correlation of fit residuals | −0.20 … 0 (the fit itself induces ≈ −0.1) |
| σ_u / σ_v, radial / tangential | 0.27 / 0.28, 0.28 / 0.27 px: isotropic |
| camera-fixed field (6×4 cells), variance explained | 3–10 %, cell means ≤ 0.2 px |
| board-fixed field per camera / pooled | 2–9 % / 1.7 % |
| between-camera correlation of board-fixed means | mean −0.02 over 15 pairs |
| per-view mean vector | none (noise-corrected variance ≤ 0) |

Reading. Most of the error is per-detection scatter that a quartic in board
position cannot fit: isotropic, uncorrelated between neighbours, and of the
same size in every direction about the image centre (so not a radial or
tilt-related effect). Only ≈ 4 % of it repeats per board point, and the
per-point offsets of different cameras are uncorrelated, so a printing error
of the board (which all cameras share) is not what is seen. The
camera-fixed field is a few percent, with cell means within their error bars
mostly below 0.2 px: a better lens model would remove at most that, not the
floor. The calibrated residual exceeds the smooth-fit scatter by 0.3–0.45 px
RMS; that gap is the part a better smooth model or per-view pose freedom could
in principle reach, and it is what remains to explain, but even closing it
completely leaves ≈ 0.4 px of scatter.

Conclusion: the ring-grid floor is dominated by per-detection localization
scatter under defocus blur, not by marker size (the markers are ≈ 17 px in
radius), print error or the projective centre bias, which is absent (above).
The detector's own documentation reports ≈ 0.28 px centre error at 3 px blur
against ≈ 0.05 px on sharp synthetic images, which matches the measured
0.27 px per axis. Lower floors need a detector change (sharper images, or an
estimator less sensitive to blur), not a camera-model or target change.

- **Gate:** `calib-bench accept` gates each `scheimpflug_rig_ringgrid_cam*` entry at
  0.7 px (~1.4× the worst committed per-camera mean, 0.508 px).
- **Diagnostic module + tests:** `ringgrid_bias.rs`
  (`conic_center_matches_sampled_ellipse_fit`,
  `bias_is_nonzero_and_equals_center_minus_projection`,
  `affine_homography_has_no_bias`); the `#[ignore]`d
  `diagnose_private_ringgrid` reproduces the tables above from the private
  dataset under `--features tier-b`.
- **Related:** ADR 0011 (seeded Scheimpflug route),
  [scheimpflug-intrinsics.md](scheimpflug-intrinsics.md),
  [planar-intrinsics.md](planar-intrinsics.md).

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
| between-camera correlation of board-fixed means (image frame, see below) | mean −0.02 over 15 pairs |
| per-view mean vector | none (noise-corrected variance ≤ 0) |

Reading. Most of the error is per-detection scatter that a quartic in board
position cannot fit: isotropic, uncorrelated between neighbours, and of the
same size in every direction about the image centre (so not a radial or
tilt-related effect). Only ≈ 4 % of it repeats per board point, and the
per-point offsets of different cameras are uncorrelated, so a printing error
of the board (which all cameras share) is not what is seen. (That
correlation was taken in image pixels, which understates a shared board
offset when the cameras see the board turned; `diagnose floor` now takes it
in the board frame, see [display-target-residual.md](display-target-residual.md).
The 4 % repeatable share above is board-frame and stands.) The
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
  [planar-intrinsics.md](planar-intrinsics.md),
  [scheimpflug-literature.md](scheimpflug-literature.md).

## Where the blur-gradient bias goes

A photometric centre under a blur gradient lands `b ≈ 1.24·σ·∇σ` towards
the sharper side (`ringgrid` ADR-022 / ADR-029; σ the fitted PSF width,
∇σ its image gradient). **Derived**
([scheimpflug-literature.md](scheimpflug-literature.md), *Ring centres under
a blur gradient*): through the focus model, `b` is to first order a view
translation, an `fy` scale and a `cy` shift, so the calibration absorbs it
and the residual does not show it.

**Method** (a study part of the bench example `scheimpflug_model_study`,
deleted when the study closed, AGENTS.md §6). ringgrid is called directly (photometric centres, the accept
path's layout) for each marker's σ; ∇σ is the least-squares plane through
the marker and its 8 nearest accepted neighbours (ringgrid's
`attach_blur_gradient_bias` rule: ≥ 4 fits, non-degenerate), and
`b = −1.24·σ·∇σ`. Three solves per camera with identical seeds: raw,
corrected (`obs − b`) and sign-flipped (`obs + b`), each the seeded route
then a tight polish (`k1, k2`, tilt, intrinsics and every pose free;
`k3, p1, p2` held). The response is the antisymmetric part
`(corr − anti)/2`. This applies the bias *law* as a correction, a proxy
for ringgrid's archived per-marker gradient term, which removes ~90 % of
the same bias.

**Measured** (4921 observations, 110 views, six cameras; "/σ" is the
response over the raw solve's stated σ):

| camera | median / p95 \|b\| px | fy/fx raw | Δ(fy/fx) | /σ | Δfy px | Δcy px | Δτx ° (/σ) | share of \|b\|² left in the residual |
|---|---|---:|---:|---:|---:|---:|---|---:|
| cam0 | 0.020 / 0.068 | 1.00808 | +3.5e-4 | +1.91 | +0.62 | +0.10 | +0.028 (+0.44) | 0.067 |
| cam1 | 0.025 / 0.090 | 1.00913 | +4.5e-4 | +1.69 | +0.67 | −0.15 | +0.029 (+0.36) | 0.082 |
| cam2 | 0.017 / 0.086 | 1.00744 | +4.6e-4 | +1.18 | +0.62 | −0.14 | +0.026 (+0.20) | 0.091 |
| cam3 | 0.020 / 0.062 | 1.00886 | +3.1e-4 | +0.90 | +0.63 | −0.08 | +0.031 (+0.25) | 0.061 |
| cam4 | 0.011 / 0.047 | 1.00760 | +3.5e-4 | +1.47 | +0.65 | −0.23 | +0.011 (+0.14) | 0.088 |
| cam5 | 0.014 / 0.060 | 1.00948 | +3.7e-4 | +0.76 | +0.36 | −0.21 | +0.008 (+0.05) | 0.105 |

Pooled: 92.1 % of `|b|²` (5.63 px²) is taken up by the camera and the
poses and 7.9 % stays in the residual. The pooled mean residual moves by
−0.003 px (0.109 → 0.106 px; the sign flip +0.004 px). The response is
linear: the symmetric part is 2–4 orders below the antisymmetric one, the
two backends agree to 3–4 figures, and the three solves end in one basin.
A first-order regression of `b` on the Jacobian of every free parameter
and pose reproduces the absorbed share to the fourth digit and the
parameter changes within a few per cent (10–25 % for `cx, cy`). On poses
alone it explains 72 %; on `cx, cy, fx, fy` and poses 91 %, but with a
different split (it flips `Δfx` and halves `Δfy`): the distortion and the
tilt take part of it. Per view: lateral translations up to 0.05 mm, along
the axis 0.04–0.08 mm, rotations ≤ 0.024°.

**Readings.**

- The bias is nearly a gauge, as derived: correcting it does not change
  the residual, which agrees with ringgrid's gradient-term on/off finding.
- The `fy` scale holds quantitatively. Predicted ≈ 4e-4 relative; measured
  `Δ(fy/fx)` +3.1 … +4.6e-4, the same sign on all six, 0.8–1.9 σ. This is
  the parameter it moves most.
- The `cy` prediction does not hold as stated. `Δcy` is −0.23 … +0.10 px
  with no common sign: the principal point trades with the poses, the
  distortion and the tilt, and only the full set fixes the split.
- The tilt moves ≤ 0.44 σ (≤ 0.031°). For geometric calibration on this
  rig the correction is not worth the accept path's change. It matters
  where the aspect ratio does (CAL-TILT-PUPIL) and when the blur itself
  is an observation (CAL-TILT-FOCUS).

**Pinned** by `blur_gradient_bias_is_absorbed_with_an_aspect_ratio_signature`
(`crates/vision-calibration/tests/planar_intrinsics_scheimpflug.rs`): an
exact Scheimpflug scene of the studied rig's geometry (720×540, `f` 1145,
`k1` −0.41, `k2` 0.2, τx −5°, 16 views at 0.17–0.33 m, 2418 points) with
the derived field `b = (k·κ²/fx)·π_xy·D` added (rms 0.075 px). Measured
there: 8e-5 of `|b|²` stays in the residual (asserted < 2e-3);
`Δ(fy/fx)` −4.26e-4 against the first-order −k·κ²·π_y²/fx² = −4.17e-4
(ratio 1.02, asserted within 15 %); tilt moves −0.006° (asserted
< 0.03°). The real data leave 6–10 % in the residual against 8e-5 here:
the derived field is smooth, the real per-marker bias also carries the
neighbourhood-plane scatter (**hypothesis**).

**Open.** (1) The study's pooled raw residual (mean 0.109, RMS 0.129 px)
is not the 0.137 px quoted from ringgrid's documents: a different solve or
residual definition; ringgrid's own harness on these captures decides it.
(2) **Hypothesis**: the raw `fy/fx` is 1.0074–1.0095 on all six, 20–50 σ
from square. Three candidates: the e-ink target's x and y scales differ
(display pixel pitch, row and column snapping;
[display-target-residual.md](display-target-residual.md)); the `d = c`
model absorbing Steger's aspect family for a tilt about a pixel axis
(0.85 % at `τ_d` = 5.2° would need `m_p ≈ 1.75`;
[scheimpflug-literature.md](scheimpflug-literature.md)); or non-square
pixels. The `fy/fx` of a printed, non-display target on the same rig, and
the datasheet pitch against tilt-from-focus's `pitch/m_p` = 5.29 µm
(`m_p ≈ 1.75` would need a 9.3 µm pitch), decide between them.

**The display candidate, checked** (measured 2026-11-22, `calib-bench run
--dataset <id>` per entry, default configs, `fy/fx` from the record's
intrinsics; σ not extracted). Untilted three-camera device on the same
kind of e-ink display, anti-aliased rendering:

| entries | target | `fy/fx`, cam0 / cam1 / cam2 | mean residual px |
|---|---|---|---|
| `rtv3s_rg_small_cam0..2` | ring grid | 1.00088 / 1.00083 / 1.00137 | 0.03 |
| `rtv3s_rg_large_cam0..2` | ring grid | 1.00095 / 1.00061 / 1.00140 | 0.03 |
| `rtv3s_pb_b_cam0..2` | PuzzleBoard | 1.00085 / 1.00096 / 1.00160 | — |
| `ipgs_cam1` | precision print | 1.00011 | 0.034 |

Each camera repeats across the three display captures to about 3e-4, and
the camera on the precision print is square to 1e-4. So the display's x/y
scale, together with those cameras' own aspect, is at most +0.06 … +0.16 %:
it cannot carry the +0.74 … +0.95 % of the tilted cameras. About 0.6–0.85 %
is left to the tilted devices themselves, which leaves two candidates:
Steger's aspect family under the `d = c` model (0.7 % at `τ_d` = 5.2° is
`τ_s ≈ 8.5°`, `m_p ≈ 1.66`, a pitch of 8.8 µm by tilt-from-focus's
`pitch/m_p`), or non-square pixels on that sensor. The sensor's model name
(its pitch) decides: a pitch near 8.8 µm supports `m_p`, a square pitch far
from it does not. Caveats: the untilted device is a different camera and
lens, so this bounds the display, not the tilted sensor; and the display
captures compared are anti-aliased, the tilted ones aliased (snapping moves
points by a fraction of a display pixel and keeps the mean scale).

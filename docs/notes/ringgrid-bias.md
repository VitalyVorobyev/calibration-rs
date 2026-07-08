# Ring-grid ellipse-center bias — proof pack

Family: coded ring-grid targets on the seeded Scheimpflug intrinsics route
(`ScheimpflugIntrinsicsProblem`; detector `vision_calibration_detect::ringgrid`
wrapping the external `ringgrid` crate; bench accept path
`vision_calibration_bench::run::run_scheimpflug_intrinsics`).

Question (backlog Q3-RINGGRID-BIAS): the ring-grid cameras floor at ~0.47 px
mean reprojection, ~0.17 px above the puzzleboard sibling (`rtv3d_ref`, ~0.30 px)
on the same rig. Is that gap the projected ellipse-center bias of the ring
markers — a systematic, correctable projective effect — or detector noise?

**Answer: it is not the projective bias.** That bias is ~0.09 px here and is
already removed at the detector level; the residual floor is small-marker
localization noise. No pipeline-side correction was added. Details below.

## Mechanism

A coded ring marker is two concentric circles (inner/outer radii `r_in`,
`r_out`) on the board plane. Set lens distortion aside for a moment: the board
plane → pixel map is then a single homography

```
H = K · H_tilt · [r1 r2 t]
```

with intrinsics `K`, the Scheimpflug tilt homography `H_tilt` (tilt is a
homography acting on normalized coordinates — ADR 0022), and the planar-pose
homography `[r1 r2 t]` from `camera_se3_target`. A circle with symmetric conic
matrix `C` on the plane images to the conic

```
C' = H⁻ᵀ C H⁻¹.
```

The **ellipse center** of `C'` — the stationary point of its quadratic form,
`center = −[[C'₀₀, C'₀₁],[C'₀₁, C'₁₁]]⁻¹ · [C'₀₂, C'₁₂]ᵀ` — is *not* the image
`H·center` of the circle's plane center under a genuinely projective (non-affine)
`H`. That offset is the ellipse-center bias. It vanishes exactly for an affine
`H` (bottom row `[0 0 1]`) and grows with the perspective foreshortening across
the marker's pixel extent, i.e. roughly as `(marker radius in px)² × |bottom row
of H|`.

Because a real lens applies (non-homographic) distortion, the observed edge is
displaced by the distortion field; to first order that displacement is common to
the marker's inner and outer edges *and* to the model's reprojection of the
center, so it cancels out of the residual — the leftover is second order in the
distortion gradient across the ~4 px marker.

## Closed-form magnitude

`crates/vision-calibration-bench/src/ringgrid_bias.rs` is the single source of
truth for the conic-center math: `circle_conic`, `project_conic`
(`C' = H⁻ᵀ C H⁻¹`), `conic_center`, and `predicted_center_bias(H, cx, cy, r) =
center(H⁻ᵀ C H⁻¹) − H·(cx, cy)`. `diagnose` evaluates it per (view, marker) from
an exported model + pose, using the no-distortion homography above.

Evaluated on the six private ring-grid cameras with the *calibrated* model and
poses (outer radius 4.8 mm, inner 3.2 mm), the predicted single-conic
(outer-edge) bias is:

| camera | n_obs | bias mean | median | p95 | max (px) |
|--------|------:|----------:|-------:|----:|---------:|
| cam0 | 1196 | 0.107 | 0.090 | 0.273 | 0.420 |
| cam1 | 1041 | 0.109 | 0.092 | 0.294 | 0.434 |
| cam2 |  576 | 0.094 | 0.092 | 0.154 | 0.241 |
| cam3 | 1193 | 0.114 | 0.099 | 0.300 | 0.452 |
| cam4 |  746 | 0.091 | 0.089 | 0.159 | 0.232 |
| cam5 |  572 | 0.084 | 0.087 | 0.135 | 0.155 |

The bias is ~0.09 px mean (max ≤ 0.45 px). It is small because the markers span
only a few pixels (outer radius ≈ 3–4 px at this working distance and focal),
and the bias scales with the square of the marker's pixel radius. This is an
order of magnitude below the ~0.47 px residual floor and below the ~0.17 px gap
to the puzzleboard sibling — so even entirely uncorrected it could not explain
the gap.

## Correction: why not

The `ringgrid` detector (≥ 0.7, `CircleRefinementMethod::ProjectiveCenter`, on by
default) already removes the *projective* part of this bias at the observation
level, before calibration sees a center. It fits both the inner and outer edge
conics and recovers their common projected center with an intrinsics-free
two-conic pencil (Wang et al. 2019) that is exact under a homography — i.e. it
cancels exactly the `C'`-center-vs-`H·center` offset derived above, for both
perspective and Scheimpflug tilt. The detector's own path applies it in
`finalize_premerge` / `run`; `vision_calibration_detect::RinggridDetector` uses
that default.

A model-based re-correction in the pipeline would therefore either (a)
double-correct the already-removed bias (strictly harmful), or (b) merely
reproduce the detector's projective correction, whose only unmodeled remainder
is the second-order distortion term (negligible here). Per the workspace's
honesty-over-heroics rule, **no correction was wired**. The conic-center math
ships as a tested diagnostic module, not a pipeline stage; `FactorKind`, the
optimization IR, the detector wrapper, and the pipeline are untouched.

## Evidence

`diagnose` also regresses the calibration residual vector `r = observed −
projected` on the predicted bias vector `b` (outer conic). If the detector
reported *naive* ellipse centers, the residual would *contain* `b` and the
regression coefficient `Σ(r·b)/Σ|b|²` would be ≈ +1. Measured:

| camera | residual mean | median | p95 | max (px) | resid/bias coeff | mean cos | corr(bias, radius) |
|--------|-------------:|-------:|----:|---------:|-----------------:|---------:|-------------------:|
| cam0 | 0.460 | 0.361 | 1.084 | 6.270 | −0.035 | 0.019 | −0.027 |
| cam1 | 0.501 | 0.347 | 1.292 | 5.616 | +0.086 | −0.021 | +0.055 |
| cam2 | 0.485 | 0.366 | 1.325 | 5.165 | +0.003 | 0.048 | +0.065 |
| cam3 | 0.463 | 0.355 | 1.050 | 7.189 | −0.031 | 0.030 | −0.011 |
| cam4 | 0.425 | 0.350 | 0.993 | 3.563 | +0.023 | 0.086 | −0.017 |
| cam5 | 0.487 | 0.410 | 1.092 | 2.985 | −0.104 | 0.065 | +0.053 |

The residual means reproduce the `calib-bench accept` per-camera means (0.4601,
0.5008, 0.4852, 0.4625, 0.4248, 0.4867 px) to three decimals, validating the
diagnostic. The regression coefficient is ≈ 0 (|·| ≤ 0.104) and the mean cosine
between `r` and `b` is ≈ 0 (|·| ≤ 0.086): **the predicted projective bias is
absent from the residual field** — the detector's two-conic pencil removed it —
and what remains is orthogonal noise. `corr(bias, radius) ≈ 0` shows no
peripheral-growth pattern in the (already tiny) bias.

Conclusion: the ~0.47 px ring-grid floor is small-marker ellipse-fit /
localization noise (ring markers a few pixels across, with the pencil amplifying
edge-fit noise), not correctable projective ellipse-center bias. The Q3 outcome
is a documented diagnostic plus a gate tightening, not a correction.

- **Before/after:** detection, model, and residuals are unchanged (no
  correction wired); the six per-camera accept means stay 0.4601 / 0.5008 /
  0.4852 / 0.4625 / 0.4248 / 0.4867 px. The acceptance gate tightened from the
  provisional 1.0 px to 0.7 px (~1.4× the worst mean); baselines are unchanged.
- **Diagnostic module + tests:**
  `crates/vision-calibration-bench/src/ringgrid_bias.rs`
  (`conic_center_matches_sampled_ellipse_fit`,
  `bias_is_nonzero_and_equals_center_minus_projection`,
  `affine_homography_has_no_bias`; the `#[ignore]`d `diagnose_private_ringgrid`
  reproduces the tables above from the private dataset under `--features
  tier-b`).
- **Related:** ADR 0022/0023 (seeded Scheimpflug route),
  `docs/notes/scheimpflug-intrinsics.md`, `docs/notes/planar-intrinsics.md`
  (the puzzleboard sibling's noise floor).

# What the Scheimpflug camera model leaves behind — a model study

The seeded Scheimpflug route (Brown-Conrady `k1, k2` under a tilted sensor,
Huber 1 px; [scheimpflug-intrinsics.md](scheimpflug-intrinsics.md)) leaves
a systematic residual on a real six-camera ring-grid rig. This note measures
what it is, which model terms remove it, and what the solver and the
parameterization have to do with it. Dev-facing; aggregate numbers only.

## Data and yardstick

`scheimpflug_rig_ringgrid_cam0..5`: six cameras of one lens type, 720×540,
`f ≈ 1150 px`, `tilt_x ≈ −0.09 rad`, `k1 ≈ −0.41`, ring-grid board with the
photometric centre method: 17–21 views of 22 poses and about 800–1200
markers per camera, mostly in the lower two thirds of each image. Working distance 150–340 mm, so the board is
close and the fitted blur σ runs from 0.7 to 3.4 px.

The yardstick is the calibration-free smooth-fit floor of `calib-bench
diagnose floor` (a quartic in board coordinates per view): **RMS 0.073,
median 0.051 px** pooled. A camera model has nothing left to explain when
its residual reaches it.

The harness is the bench example `scheimpflug_model_study`
(`crates/vision-calibration-bench/examples/scheimpflug_model_study/`):

```text
cargo run --release -p vision-calibration-bench --features tier-b \
    --example scheimpflug_model_study -- \
    --registry crates/vision-calibration-bench/registry/private.json \
    --out <dir> [--parts lib,gen,basin,blur,xcam] <dataset-id>...
```

Every variant starts from the route's solution and is converged tightly
(relative decrease 1e-14). **Held-out** scoring is 4-fold over views: the
route and the variant are refitted on three folds, and only the fourth
fold's poses are fitted with the camera held. Library variants go through
`optimize_planar_intrinsics` with fix masks; the others are written as a
user residual through `optim::problem` (`model.rs`), whose baseline
configuration reproduces the library's projection exactly (it starts at
the route's optimum and stops after one iteration with the same cost).

## Results

Pooled over the six cameras; RMS / median in px. *Gap* is
`sqrt(rms² − floor²)` on held-out views, *cam field* the share of the
held-out residual variance explained by a 6×4 image-cell mean, *spread* the
standard deviation of the estimate across the six cameras, *fold sd* the
mean within-camera standard deviation over the four training folds.

| model | train | held-out | gap | cam field | spread fx / cx / cy px, tilt_x mrad | fold sd fx / cy px, tilt_x mrad | iters |
|---|---|---|---:|---:|---|---|---:|
| bc5 `k1 k2` (default) | 0.137 / 0.105 | 0.138 / 0.106 | 0.117 | 18 % | 8.5 / 16 / 6.5, 6.0 | 1.5 / 0.19, 0.9 | 6–7 |
| + `k3` | 0.136 / 0.104 | 0.137 / 0.106 | 0.117 | 18 % | 8.3 / 16 / 6.6, 5.9 | 1.6 / 0.19, 1.0 | 7–9 |
| + `p1 p2` | 0.134 / 0.104 | 0.135 / 0.105 | 0.114 | 17 % | 8.1 / 63 / 31, 24 | 3.1 / 20, 17 | 23–43 |
| rational8, all free | 0.132 / 0.100 | 0.134 / 0.102 | 0.112 | 16 % | 9.2 / 80 / 35, 28 | 4.3 / 21, 18 | 90–240 |
| thin prism, `k1 k2 s1–s4` | 0.127 / 0.098 | 0.128 / 0.099 | 0.105 | 12 % | 15 / 22 / 181, 29 | 2.9 / 38, 6.3 | 31–110 |
| thin prism, all free | 0.109 / 0.082 | 0.111 / 0.084 | 0.085 | 6 % | 47 / 190 / 280, 174 | 26 / 114, 68 | 83–142 |
| `p s` free, tilt held | 0.129 / 0.100 | 0.131 / 0.102 | 0.109 | 14 % | 15 / 22 / 123, 6.0 | 3.5 / 26, 0.9 | 15–90 |
| distortion centre `≠` pp | 0.133 / 0.101 | 0.134 / 0.102 | 0.113 | 16 % | 11 / 19 / 69, 55 | 3.2 / 34, 29 | 46–90 |
| tilt before distortion | 0.135 / 0.103 | 0.136 / 0.104 | 0.115 | 18 % | 8.5 / 17 / 6.5, 6.2 | 1.6 / 0.19, 1.0 | 4–5 |
| tilt `× (1 + η r²)` (pupil) | 0.132 / 0.103 | 0.133 / 0.104 | 0.111 | 19 % | 8.4 / 17 / 7.1, 5.2 | 1.8 / 0.27, 1.3 | 5–7 |
| depth-dependent `k1` | 0.136 / 0.104 | 0.137 / 0.105 | 0.116 | 19 % | 8.3 / 16 / 6.5, 6.0 | 1.6 / 0.16, 1.0 | 4–5 |
| board out of plane, degree 3 | 0.118 / 0.089 | 0.125 / 0.093 | 0.102 | 12 % | 7.6 / 17 / 6.5, 5.1 | 1.7 / 0.72, 1.2 | 6–7 |
| image field, degree 2 (6) | 0.107 / 0.082 | 0.110 / 0.083 | 0.082 | 11 % | 60 / 10 / 148, 89 | 20 / 52, 25 | 33–153 |
| image field, degree 3 only (8) | 0.105 / 0.077 | 0.108 / 0.078 | 0.079 | 9 % | 6.7 / 6.7 / 3.0, 5.2 | 2.8 / 0.27, 2.4 | 8–24 |
| image field, degrees 2–3 (14) | 0.078 / 0.059 | 0.082 / 0.061 | 0.038 | 5 % | 11 / 9.3 / 6.9, 209 | 5.6 / 4.9, 6.4 | 40–108 |
| image field, degrees 2–4 (24) | 0.065 / 0.047 | 0.068 / 0.049 | 0 | 5 % | 13 / 6.5 / 18, 24 | 2.5 / 3.2, 9.2 | 62–159 |
| image field, degrees 2–6 (50) | 0.057 / 0.039 | 0.062 / 0.041 | 0 | 2 % | 132 / 11 / 84, 272 | 12 / 11, 33 | 102–328 |

The *image field* is a polynomial displacement in normalized pixel
coordinates added to the projection, shared by every view of a camera: an
upper bound for what any central-camera model refinement can reach. The
covariance is available for every variant (no bound, no rank deficiency),
which says nothing about its honesty; see below.

Per-camera mean reprojection error, held-out views (px):

| model | cam0 | cam1 | cam2 | cam3 | cam4 | cam5 |
|---|---:|---:|---:|---:|---:|---:|
| bc5 `k1 k2` (default) | 0.146 | 0.091 | 0.116 | 0.115 | 0.139 | 0.125 |
| thin prism, all free | 0.123 | 0.084 | 0.087 | 0.096 | 0.102 | 0.101 |
| image field, degrees 2–3 | 0.085 | 0.056 | 0.061 | 0.065 | 0.094 | 0.072 |
| image field, degrees 2–4 | 0.071 | 0.053 | 0.052 | 0.058 | 0.063 | 0.060 |

### The gap is a camera-fixed, smooth model error

A free image field of degree 4 brings the held-out residual to the
smooth-fit floor (0.068 against 0.073 px; the floor is slightly
conservative because it spends 15 terms per view). The field is fitted on
other views than it is scored on, so it is a property of the camera, not
of a view. Nothing else comes close:

- **The parametric distortion family is not the missing piece.** `k3`,
  `p1 p2`, the rational terms and the distortion centre each lower the
  held-out RMS by at most 3 %. The thin-prism model with everything free
  closes a quarter of the gap, but its `p, s` terms span the tilt's keystone to
  first order, so the tilt and the principal point stop being identified:
  across the six cameras `tilt_x` spreads 0.17 rad and `cy` 280 px, and
  within one camera the fold scatter is 20–40 times the stated σ. With the
  tilt held the same terms recover little (0.131).
- **The order and form of the tilt are not it.** Tilting before the
  distortion, a field-dependent tilt (an exit pupil that moves with the
  field angle) and a separate distortion centre lower it by 1–4 %.
- **Depth-dependent distortion is real and small.** `k1 + kz·(z₀/Z − 1)`
  fits `kz` between −0.021 and −0.035 on all six cameras (σ 0.003–0.008:
  same sign, significant), and per view the residual's `r²` coefficient
  correlates −0.76 with `z̄/Z − 1`; the term lowers the RMS by about 1 %.
- **The board is flat enough.** An out-of-plane board shape lowers the
  training residual, but its coefficients disagree between the six
  cameras that look at the same board (the `x²` term ranges from +0.01 to
  −0.65 mm), it gains half as much on held-out views, and with the image
  field present it makes held-out views worse (0.071 against 0.068). The
  in-plane print error is 2.5 % of the variance (`diagnose floor`). A
  shape shared by all six cameras in one joint fit, and the display-target
  case with in-plane offsets and a cover layer, are measured in
  [display-target-residual.md](display-target-residual.md).
- **Neither the tangential nor the third-order terms alone suffice**: the
  degree-2 field (the span of the tilt, `p` and the `r²` prism terms) and
  the third-order field each lower the held-out RMS by about a fifth;
  together (14 terms) they close two thirds of the gap's RMS and 90 % of
  its variance. The fourth-order terms close the rest.

The field of one camera, held, partly transfers to the others: among
cameras 0–3 a borrowed field lowers the residual from 0.11–0.17 to
0.08–0.12 px (own field 0.06–0.08), so a good part of it belongs to the
lens type. Cameras 4 and 5 do not take or give a field; the degree-4
polynomial is a poor carrier for transfer, since it trades with `k1` (it
settles at `k1 ≈ −8.7` with the field cancelling most of it).

### Part of it depends on the blur

The photometric detector fits a Gaussian PSF per marker. Splitting the
markers of each camera at the median blur, and separately at the median
of the blur *relative to the blur typical of the marker's image position*
(a quadratic surface over the image, so both halves cover the image
alike), and training on one half, scoring the other (poses refitted,
camera held), pooled RMS in px:

| split | model | sharp → sharp | sharp → blurred | blurred → blurred | blurred → sharp |
|---|---|---:|---:|---:|---:|
| random halves (control) | bc5 default | 0.136 | 0.137 | 0.137 | 0.136 |
| random halves (control) | field degree 4 | 0.064 | 0.066 | 0.063 | 0.067 |
| relative to position | bc5 default | 0.093 | 0.146 | 0.117 | 0.107 |
| relative to position | field degree 4 | **0.034** | 0.111 | 0.059 | 0.058 |
| absolute | field degree 4 | 0.032 | 0.180 | 0.062 | 0.059 |

On markers sharper than typical for their position, the field brings the
residual to 0.034 px: the detector's floor on sharp markers. A field
learned there does not carry to blurrier markers at the same image
positions (0.111 against their own 0.059), while one learned on the
blurrier markers serves the sharp ones at 0.058. So the blurrier markers
carry an extra displacement that the sharp ones do not, and it is not a
function of image position alone. It is the PSF-mismatch bias suspected
of the Gaussian photometric model, or a depth effect (the relative split
also separates views near the plane of focus from views away from it);
this dataset cannot tell the two apart. Simple regressions of the
residual on the blur, its gradient and the depth explain at most 3 % of
its sum of squares (about 3 % jointly), so the dependence is not a
simple linear one.

### Not the solver

The route's solution against a solve from it with a relative-decrease
tolerance of 1e-14 and 3000 iterations, on both backends: the cost agrees
to all printed digits and `fx` moves by less than 0.01 px, `tilt_x` by
less than 5 µrad. The route stops 6–7 iterations in at the minimum; the
stopping rule does not leave anything on the table here.

### Reparameterizing the tilt and principal point

On this dataset the tilt is well determined: `corr(cy, tilt_x)` is
0.005–0.115 (the +0.995 of the two-camera synthetic scene in
[scheimpflug-intrinsics.md](scheimpflug-intrinsics.md) does not occur with
16 well-spread views), `corr(cx, tilt_y)` 0.82–0.90 with a valley slope
`dcx/dtilt_y = 1.26–1.39 fx`. Measured, from the optimum with one tilt
perturbed by up to ±0.15 rad (±8.6°), six cameras, 16 starts each:

| parameterization | converged | iterations (tilt_x / tilt_y starts) | `corr(cy, tilt_x)` |
|---|---:|---|---:|
| `cx, cy` (current) | 96 / 96 | 3–5 / 3–5 | 0.005–0.115 |
| `c̃ = c − f·tan(tilt)` | 96 / 96 | 9–23 / 5–11 | −0.90 to −0.93 |

Both reach the same minimum from every start; the substituted coordinates
take two to four times the iterations and, since the valley they are
built for is absent, create the correlation they were meant to remove.
Gauss-Newton is invariant to a linear change of variables and the
Marquardt diagonal is the only thing the substitution touches, so no gain
was to be expected where the valley is linear. A substitution changes
which combination the covariance reports, not the information in it.

### The parameters are not unstable

The default model's parameters differ between the six cameras by much
more than their σ (`fx` spreads 8.5 px against σ ≈ 2–3 px, `cx` 16 px
against 0.9–1.2 px), but within a camera the delete-one-fold jackknife
scatter matches the stated σ: ratio 0.5–1.6 for `fx`, 0.5–1.1 for `cx`,
0.7–1.2 for `tilt_x` (4 folds, so each ratio is itself uncertain to about
±40 %). The six units differ; the estimates are stable. Freeing more terms
destroys that: with the thin-prism model the same ratio reaches 20–40.

## Conclusions

1. The 0.117 px gap is a smooth, camera-fixed displacement field that the
   Brown-Conrady + Scheimpflug family cannot represent, plus a smaller
   blur- or depth-dependent part. A free image field of degree 4 on top
   of the current model reaches the smooth-fit floor on held-out views and
   puts every camera's mean error at 0.052–0.071 px.
2. No change of the defaults is supported: the fix list (only `k1, k2`
   free under a tilt), the stopping rule and the parameterization are each
   measured above and none is the limit. Freeing the thin-prism or
   tangential terms lowers the residual at the price of an unidentified
   tilt and principal point, which no default should do.
3. A residual-field model term is the change the evidence asks for; it is
   proposed, not implemented (backlog `CAM-RESIDUAL-FIELD`). Its design
   problems are measured here: a plain monomial basis trades with `k1`
   and the intrinsics (`k1 → −8.7`), a polynomial extrapolates badly
   outside the observed region (the markers cover the lower two thirds),
   and its stated σ is up to 4 times optimistic (jackknife ratio 0.9–3.7). It needs
   a basis orthogonal to the parametric terms, a bounded support (a
   B-spline grid over the image, or a field that decays outside the data)
   and a selection rule on held-out views (`CAM-SELECT`).
4. The blur-dependent part belongs to the detector: the sharp markers show
   a 0.034 px floor that the blurred ones do not reach under any camera
   model.

## What would settle the rest

- Views that put the board at the same image positions at different
  depths, and the same depth at different positions: blur and depth then
  vary independently, which separates the detector's PSF bias from a
  depth-dependent distortion.
- Coverage of the top third of the image, so a field is fitted where it is
  used rather than extrapolated.
- A detector-side test: rendered markers blurred with a measured (not
  Gaussian) defocus PSF through the photometric fit, to size the bias the
  Gaussian model leaves.

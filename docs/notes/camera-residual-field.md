# A camera residual field for Scheimpflug rigs — structure, artefact, design

What camera term removes the camera-fixed part of the residual that the
Brown-Conrady + tilt model leaves on the two private six-camera Scheimpflug
rigs, and how a generic field has to be built to be sound. Dev-facing;
aggregate numbers only. The design decision is
[ADR 0030](../adrs/0030-camera-residual-field.md); the earlier studies are
[scheimpflug-residual-model.md](scheimpflug-residual-model.md) and
[display-target-residual.md](display-target-residual.md), whose
image-field numbers this note corrects. A later study,
[parametric-residual-term.md](parametric-residual-term.md), finds that most
of the field measured here is coupled to defocus and depth rather than
fixed in the image, which puts ADR 0030 on hold.

## Conclusions

1. **The earlier image-field gain was mostly an artefact.** The polynomial
   field of those studies was evaluated at the *observed* pixel. That is
   not a camera model: once `k1` is driven far enough (−6 to −9) the field's
   Jacobian approaches the identity over the observed region, and the
   residual `u_obs − u_par − F(u_obs)` contracts the detection noise
   instead of explaining it. Held-out views do not catch it, because their
   pose-only fit evaluates the field at their observations too. On
   synthetic data from the plain model (noise 0.099 px RMS) it reports
   0.088 px held-out, 11 % *below* the noise, with `k1` at −4.1 against a
   true −0.415. Evaluated at the model's own projection, the same field
   gains nothing on those data, and on the real rigs it gains a third of
   what was reported: ring-grid 0.143 → 0.109 px held-out RMS (reported
   0.068), display 0.315 → 0.265 (reported 0.186).
2. **No structured term with a handful of identified parameters exists on
   this evidence.** Every physical candidate either gains under 5 % or
   gains by trading with the tilt and principal point, whose values then
   differ between folds by tens of pixels and between the six cameras by
   hundreds (table below).
3. **A designed generic field does reach the goal on the ring-grid rig.**
   A bicubic B-spline over the image, evaluated at the projection, held
   orthogonal to what the camera parameters already span, with a weak
   smoothness penalty, takes the held-out RMS from 0.143 to 0.100 px and
   every camera's held-out mean to 0.068–0.099 px (from 0.091–0.146),
   while the camera parameters stay within their fold scatter of the
   route's values (`fx` +0.8 px, `cy` −1.1 px, `tilt_x` −2.3 mrad on
   average). On the display rig it takes 0.315 to 0.248 (means
   0.183–0.239), still above that rig's detector-and-target floor
   (smooth fit 0.213 RMS), so 0.1 px is not reachable there by any camera
   term; there the field also doubles the fold-to-fold scatter of `fx` and
   `tilt_x`, beyond their stated σ.
4. **Its stated σ is honest only if the smoothness term is a penalty.**
   Counted as known information (the library's convention for a stated
   prior), the control points' σ is 5–6 times too large; as a penalty the
   Monte-Carlo scatter matches it (ratio 0.89–1.03).

## The artefact

With the field at the observation the residual is
`r = u_obs − P(θ, X) − F(u_obs)`. Write `u_obs = u + n`. To first order
`r = [u − P − F(u)] − (I − ∂F/∂u)·n`: the noise enters through
`I − ∂F/∂u`, so a field whose Jacobian is near `I` where the points are
shrinks the noise term, and the cost rewards it. A large negative `k1`
compresses the parametric image; the field then expands it back, with a
Jacobian of `3|k1|r²` ≈ 1 at the typical `r²` ≈ 0.05: the optimum at
`k1 ≈ −8.7` (ring-grid) and −6 (display) is where that holds on average.
With the camera held (no `k1` freedom) both forms give the same numbers
(ring-grid camera 0: 0.1429 / 0.1429; display camera 0: 0.3228 /
0.3230), which locates the effect in the `k1` runaway. Synthetic check,
plain-model data, 20 views, 4-fold held-out, two seeds:

| model | held-out RMS | `k1` | `tilt_x` |
|---|---:|---:|---:|
| plain (truth: −0.415, −0.096) | 0.0988 / 0.0980 | −0.4151 | −0.0961 |
| field 2–4, at the projection | 0.0989 / 0.0981 | −0.4152 | −0.0960 |
| field 2–4, at the observation | **0.0881 / 0.0893** | −4.12 | −0.044 |

Since the loop stops on the distance to the minimum in σ (`lm-stopping.md`),
the pinned test of this (`a_field_at_the_observation_contracts_the_noise`,
10 views, field degree 4) shows the field at the projection trading with
`k1` along a flat valley: `k1` −0.474 against −0.415 after 100 iterations
without reaching the minimum, the cost 0.3 % lower (*measured*). The old
rule had stopped it near its start. The contrast stands (−0.06 against
more than 1 at the observation); whether the 20-view row above moves
under the new rule is not re-measured (*hypothesis*: little, the valley
being shorter with more views; rerun `--parts gen` to decide).

The joint target study had the same form; its corrected rows are below.

## Held-out results

Harness: the bench example `scheimpflug_model_study` (`--parts gen,lib`),
4 folds by view index per camera, poses of held-out views fitted with the
camera held. RMS / mean / median over all points in px, then the mean per
camera. *Spread* is the standard deviation over the six cameras of the
full-data fit, *fold sd* the mean within-camera standard deviation over
the four training folds (`fx` and `cy` in px, `tilt_x` in mrad).

### Ring-grid rig (`scheimpflug_rig_ringgrid_cam0..5`; floor 0.073 RMS)

| model | held-out | cam0 | cam1 | cam2 | cam3 | cam4 | cam5 | spread fx / cy, tilt_x | fold sd fx / cy, tilt_x |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| route `k1 k2` + tilt | 0.143 / 0.121 / 0.109 | 0.146 | 0.091 | 0.116 | 0.115 | 0.139 | 0.125 | 8.5 / 6.5, 6.0 | 1.5 / 0.19, 0.9 |
| + `k3 p1 p2` | 0.140 / 0.119 / 0.107 | 0.144 | 0.088 | 0.112 | 0.113 | 0.138 | 0.122 | 6.9 / 39, 31 | 4.7 / 21, 19 |
| thin prism, all free | 0.117 / 0.099 / 0.087 | 0.123 | 0.084 | 0.087 | 0.096 | 0.102 | 0.101 | 47 / 280, 174 | 26 / 114, 68 |
| depth-dependent `k1` | 0.141 / 0.119 / 0.108 | 0.145 | 0.089 | 0.116 | 0.111 | 0.139 | 0.123 | 8.3 / 6.5, 6.0 | 1.6 / 0.16, 1.0 |
| tilt first + distortion centre | 0.140 / 0.119 / 0.107 | 0.144 | 0.089 | 0.114 | 0.112 | 0.137 | 0.124 | 8.5 / 6.5, 6.1 | 1.6 / 0.20, 1.0 |
| `k3` + distortion centre | 0.139 / 0.118 / 0.105 | 0.142 | 0.089 | 0.113 | 0.111 | 0.135 | 0.122 | 11 / 65, 52 | 3.0 / 14, 12 |
| sensor height deg 2–3 @ foot | (cam5 diverges) | 0.135 | 0.084 | 0.103 | 0.102 | 0.127 | — | 158 / 380, 259 | 7.8 / 11, 5.1 |
| sensor height deg 2–3 @ free centre | 0.114 / 0.097 / 0.084 | 0.130 | 0.077 | 0.086 | 0.091 | 0.100 | 0.092 | 14 / 34, 70 | 7.3 / 13, 15 |
| poly field 2–4 @ observation (artefact) | 0.072 / 0.060 / 0.051 | 0.071 | 0.053 | 0.052 | 0.058 | 0.063 | 0.060 | 13 / 18, 24 (`k1` −8.7) | 2.5 / 3.2, 9.2 |
| poly field 2–4, camera free | 0.109 / 0.092 / 0.080 | 0.117 | 0.074 | 0.082 | 0.089 | 0.098 | 0.092 | 106 / 29, 82 | 40 / 51, 76 |
| poly field 2–4, camera held | 0.116 / 0.099 / 0.087 | 0.123 | 0.076 | 0.089 | 0.095 | 0.114 | 0.102 | route | route |
| spline 6×5, free, σ 10 | 0.168 / 0.090 / 0.071 | 0.105 | 0.079 | 0.095 | 0.086 | 0.093 | 0.085 | 50 / 34, 9.5 | 23 / 18, 4.7 |
| spline 6×5, camera held, σ 10 | 0.103 / 0.085 / 0.071 | 0.105 | 0.070 | 0.077 | 0.081 | 0.093 | 0.083 | route | route |
| spline 6×5, orthogonal, σ 10 | 0.105 / 0.087 / 0.075 | 0.110 | 0.071 | 0.079 | 0.082 | 0.099 | 0.087 | 7.9 / 6.4, 4.3 | 2.0 / 0.23, 1.6 |
| spline 6×5, orthogonal, σ 30 | 0.103 / 0.084 / 0.070 | 0.105 | 0.070 | 0.076 | 0.078 | 0.094 | 0.082 | 7.9 / 6.4, 4.3 | 2.1 / 0.23, 1.7 |
| **spline 10×8, orthogonal, σ 10** | **0.100 / 0.079 / 0.064** | 0.099 | 0.068 | 0.072 | 0.075 | 0.087 | 0.077 | 7.8 / 6.4, 4.3 | 2.0 / 0.23, 1.6 |

An elliptic radial metric (`r² + e1(x² − y²) + 2 e2 xy` in `k1`) and a
radial sensor-plane bowl centred on the foot of the sensor normal change
camera 0 by less than 0.001 px; the bowl at the principal point is `k1`
again. The spline's smoothness σ is that of each control point's second
difference with the reprojection residuals at unit weight (the problem
builder's convention): σ 1 → 0.116, 3 → 0.108, 10 → 0.105, 30 → 0.103,
100 → 0.104 (6×5); cells 4×3 → 0.107, 8×6 → 0.102, 10×8 → 0.100 (σ 10–30).

### Display rig (`scheimpflug_rig_ref_cam0..5`; floor 0.213 RMS)

| model | held-out | cam0 | cam1 | cam2 | cam3 | cam4 | cam5 | spread fx / cy, tilt_x | fold sd fx / cy, tilt_x |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| route `k1 k2` + tilt | 0.315 / 0.268 / 0.238 | 0.310 | 0.230 | 0.237 | 0.261 | 0.303 | 0.273 | 5.5 / 5.0, 2.1 | 1.4 / 0.18, 0.7 |
| + `k3 p1 p2` | 0.308 / 0.263 / 0.234 | 0.306 | 0.227 | 0.231 | 0.256 | 0.294 | 0.271 | 12 / 75, 64 | 3.2 / 20, 17 |
| thin prism, all free | 0.284 / 0.241 / 0.214 | 0.279 | 0.216 | 0.207 | 0.236 | 0.254 | 0.258 | 47 / 267, 193 | 6.5 / 52, 17 |
| depth-dependent `k1` | 0.313 / 0.267 / 0.236 | 0.309 | 0.229 | 0.235 | 0.260 | 0.301 | 0.270 | 5.6 / 4.9, 2.2 | 1.6 / 0.17, 0.9 |
| tilt first + distortion centre | 0.311 / 0.265 / 0.236 | 0.306 | 0.226 | 0.233 | 0.257 | 0.301 | 0.271 | 5.6 / 5.0, 2.2 | 1.5 / 0.18, 0.8 |
| sensor height deg 2–3 @ foot | 0.297 / 0.254 / 0.228 | 0.296 | 0.213 | 0.222 | 0.244 | 0.290 | 0.265 | 5.6 / 11, 7.5 | 1.8 / 1.2, 1.3 |
| sensor height deg 2–3 @ free centre | 0.277 / 0.235 / 0.208 | 0.277 | 0.213 | 0.208 | 0.232 | 0.247 | 0.239 | 67 / 94, 138 | 36 / 30, 54 |
| poly field 2–4 @ observation (artefact) | 0.183 / 0.151 / 0.127 | 0.167 | 0.137 | 0.138 | 0.139 | 0.163 | 0.159 | 9.8 / 19, 47 (`k1` −6.0) | 1.9 / 2.6, 8.9 |
| poly field 2–4, camera free | 0.265 / 0.225 / 0.199 | 0.266 | 0.199 | 0.195 | 0.220 | 0.239 | 0.233 | 135 / 98, 166 | 53 / 27, 84 |
| poly field 2–4, camera held | 0.278 / 0.236 / 0.207 | 0.277 | 0.208 | 0.206 | 0.230 | 0.256 | 0.243 | route | route |
| spline 6×5, camera held, σ 10 | 0.250 / 0.210 / 0.182 | 0.247 | 0.189 | 0.184 | 0.193 | 0.225 | 0.221 | route | route |
| spline 6×5, orthogonal, σ 30 | 0.250 / 0.210 / 0.182 | 0.245 | 0.189 | 0.185 | 0.195 | 0.225 | 0.220 | 5.8 / 5.0, 1.9 | 2.4 / 0.26, 2.0 |
| **spline 10×8, orthogonal, σ 10** | **0.248 / 0.207 / 0.177** | 0.239 | 0.188 | 0.183 | 0.188 | 0.223 | 0.219 | 5.8 / 5.0, 2.0 | 2.3 / 0.26, 2.0 |

The polynomial fields of degree 3 and the `k3` + centre model put single
held-out points tens of pixels off on the display rig (pooled RMS 21 px):
an unbounded basis extrapolates.

### The target term with a proper field (display rig, joint fit)

The joint six-camera fit of [display-target-residual.md](display-target-residual.md),
folds by capture, with the field at the projection:

| model | held-out | per-camera mean |
|---|---|---|
| baseline | 0.315 / 0.268 / 0.238 | 0.230–0.311 |
| + offsets per point | 0.269 / 0.228 / 0.202 | 0.205–0.254 |
| + poly field 4 | 0.270 / 0.229 / 0.200 | 0.194–0.273 |
| + offsets + poly field 4 | 0.221 / 0.184 / 0.159 | 0.170–0.196 |
| + offsets + poly field 4, camera held | 0.240 / 0.201 / 0.174 | 0.179–0.218 |
| + offsets + poly field 4 @ observation (reported before) | 0.153 / 0.124 / 0.103 | 0.116–0.134 |

The offsets' own gain (0.315 → 0.269) stands; on top of a proper field it
is 0.270 → 0.221.

## Why the free fields fail and what fixes each

- **Spanning the camera.** A cubic B-spline reproduces every polynomial of
  degree ≤ 3 in `u` and `v`, so a free spline contains `fx, fy, cx, cy`
  and the tilt's keystone exactly: the covariance reports the intrinsics
  unobservable, and the folds put `fx` 23 px apart. Holding the spline to
  `Σ_points J_jᵀ B(u) · c = 0` for the eight camera parameters and a
  common rotation (the camera's pixel directions against the basis, at the
  route's solution) leaves the camera where the route has it: the
  orthogonal fit (10×8, σ 10) moves `fx` by +0.8 px, `cy` by −1.1 px and
  `tilt_x` by −2.3 mrad on average over the ring-grid cameras (fold sd
  2.0 px, 0.23 px, 1.6 mrad), and the
  covariance is available.
- **The constraint must not include a common translation.** A translation
  of every pose moves a point by its inverse depth, which an image field
  cannot copy; including it forbids directions the data use (ring-grid
  camera 1: 0.0978 with it, 0.0850 without, at σ 10).
- **Camera held or orthogonal: equal on held-out views** (ring-grid 0.103
  against 0.103–0.105, display 0.250 against 0.250). The route's residual
  is already orthogonal to the camera's directions, so a field fitted to it
  needs little from them on the points. Off the points it differs: the
  camera-held spline carries camera-like terms into the uncovered border
  cells, with control values up to 20–96 px (Lipschitz bound up to 0.12),
  against 1.7–7.4 px (bound ≤ 0.046) for the orthogonal one. The
  orthogonal form is also the one that survives a later joint refinement
  of the camera (rig, hand-eye).
- **Where the orthogonality is measured.** Over the fitted points (their
  density as the weight) or uniformly over the covered image cells (a
  24 × 18 occupancy grid): the same held-out numbers (ring-grid 0.100,
  display 0.248 at 10×8, σ 10), and the covered-image form ties the split
  to the coverage rather than to where the points cluster; its fold
  scatter of `tilt_x` and `k1` is the smaller of the two (below).
- **Bounded.** The spline's argument is clamped to the image, so a point
  outside takes the border value; no spline fit put a held-out point off,
  against 21 px RMS for the cubic polynomial on the display rig.
- **Weak smoothing.** The gain grows up to σ 30 and is flat beyond; finer
  grids help a little (10×8 over 6×5: 0.002–0.003 px). The term
  needs a selection rule (held-out folds over a short ladder), not a
  tuned default.

## Stability of the camera on real views

Delete-one-fold jackknife standard deviation over the four training
folds, divided by the stated σ, root-mean-square over the six cameras
(1 when σ is honest; each camera's value is uncertain to about ±40 %):

| rig, model | `fx` | `cx` | `cy` | `tilt_x` | `k1` |
|---|---:|---:|---:|---:|---:|
| ring-grid, route | 1.84 | 1.41 | 0.71 | 1.60 | 1.17 |
| ring-grid, spline 10×8 orthogonal over the points | 2.19 | 0.73 | 0.90 | 2.86 | 1.85 |
| ring-grid, spline 10×8 orthogonal over the covered image | 2.18 | 0.74 | 0.37 | 2.05 | 0.44 |
| display, route | 2.29 | 1.31 | 0.82 | 1.66 | 1.80 |
| display, spline 10×8 orthogonal over the points | 4.74 | 1.61 | 1.33 | 6.29 | 2.64 |
| display, spline 10×8 orthogonal over the covered image | 4.65 | 1.67 | 0.92 | 4.96 | 1.57 |

The route itself scatters up to twice its σ between folds on real views
(the per-view error ADR 0026's view-scatter scaling exists for; these
user-residual solves do not apply it). On the ring-grid rig the field
leaves that unchanged; on the display rig it doubles the scatter of `fx`
and `tilt_x` (fold sd 2.3 px and 2.0 mrad, against 1.4 px and 0.7 mrad):
there the field and the tilt share more than the constraint at the
route's solution separates, and the stated σ does not show it.

## What the camera-fixed field looks like

The route's camera with a degree-4 field at the projection (camera held)
is taken as the truth and its projections replace the detections
(`--parts mimic`); on that noise-free copy the route leaves 0.069–0.093 px
RMS (ring-grid) and 0.120–0.189 px (display). Per camera:

- **Cubic and quartic, not radial.** Complex monomials `z^p z̄^q` about
  the principal point, each group fitted alone: degree 3 explains 25–52 %
  of it, degree 4 36–60 %, degree 2 1–14 %; the radial (`m = 0`, real)
  part none, the radial twist ≤ 4 %. The largest single rotation order is
  `m = −4` (`z̄³`, 14–37 %), a pattern none of the lens, tilt or decentring
  terms produces to first order.
- **What the candidates take from it**, fitted to the noise-free copy
  (route 0.069–0.093 / 0.120–0.189 px): distortion centre, `k3` + centre,
  elliptic metric, depth `k1`, prism with the tilt held: 0.055–0.093 /
  0.097–0.189; a sensor height at the foot: 0.043–0.081 / 0.077–0.172;
  the identifying-tilt-losing ones go further (all nine distortion terms
  0.023–0.043 / 0.055–0.169, sensor height with a free centre
  0.025–0.048 / 0.050–0.114), and a cubic-only field (8 terms) leaves
  0.022–0.047 / 0.044–0.085.
- **Similar across the ring-grid units, not across the display ones.**
  The cell maps of the six ring-grid cameras correlate +0.53 to +0.87,
  those of the display rig +0.13 to +0.84. A field fitted on one
  ring-grid camera and held on another lowers the other's residual for
  some pairs (camera 2: 0.135 → 0.110 with camera 0's) and raises it for
  others, so part of it belongs to the lens type; a shared field was not
  measured in the projection form.

## Monte-Carlo and synthetic truth

Scene: the typical camera (`f` 1140/1149 px, `k1` −0.415, `k2` 0.24,
`tilt` −0.096/0.004 rad), 20 views of a 24 × 18 grid at 170–330 mm, noise
0.07 px per axis; truth field a spline 6×5 with its camera-parallel part
removed (0.15 px RMS on the points). 40 noise draws, fitted from the truth
with the orthogonal spline at σ 10 (TinySolver; a user residual runs on
that backend only):

| parameter | constraint σ 1e-4, smoothing as known information | 1e-4, smoothing as a penalty | 1e-6, as a penalty | plain model on plain data |
|---|---:|---:|---:|---:|
| `fx`, `fy` | 0.61–0.63 | 0.74–0.79 | 0.81–0.83 | 0.85 |
| `cx`, `cy` | 0.34–0.36 | 0.34–0.37 | 0.87–1.09 | 0.92–1.07 |
| `k1`, `k2` | 0.92–0.94 | 0.95–0.97 | 0.96–0.98 | 0.97–0.99 |
| `tilt_x`, `tilt_y` | 0.78–0.90 | 0.88–0.97 | 0.88–0.97 | 0.89–1.01 |
| four central control points | 0.16–0.19 | 0.89–1.03 | 0.92–1.06 | — |

(MC sd / stated σ; 1 is honest.) Every estimate is unbiased to well
inside its scatter. Two things decide the σ, both about what the
covariance takes as known information. The smoothing term, stated, adds
its σ to the gradient's scatter although the true field is fixed: the
control points' σ comes out 5–6 times too large; entered as a penalty (a
family whose scatter is estimated) it is honest. The orthogonality
constraint, as a stated residual, leaks its σ into the parameters it
separates the field from: at 1e-4 px `cx, cy` state three times their
scatter (at 1e-2, 100–10 000 times), at 1e-6 they match the plain model;
but at 1e-6 a solve started from the route rather than the truth accepts
no step (ring-grid camera 1 stops at iteration 1), so no penalty σ is
right for both. A hard constraint (a subspace manifold) has neither
problem. The real-data runs
above used 1e-4; their held-out numbers do not depend on it (ring-grid
camera 1: identical at 1e-2 and 1e-4).

The same scenes scored like the datasets (4-fold held-out RMS, px; the
noise alone gives 0.099):

| truth | route | poly field 2–4 | spline 6×5 orth. σ 10 | spline 10×8 orth. (covered image) σ 10 |
|---|---:|---:|---:|---:|
| plain model | 0.0988 | 0.0989 | 0.0993 | 0.0997 |
| plus a 0.15 px orthogonal spline | 0.1704 | 0.1042 | 0.0993 | 0.0997 |
| plus a quartic field outside the spline's span | 0.1041 | 0.0989 | 0.0993 | 0.0997 |

On plain data the spline costs 0.5–0.9 % on held-out views, which is what
a selection rule has to reject.

The bench example's tests (`cargo test -p vision-calibration-bench
--features tier-b --example scheimpflug_model_study`) pin three facts on
these scenes: the observation-evaluated field contracts the noise and
moves `k1` by more than 1, the projection-evaluated one gains under 2 %
and keeps `k1` within 0.05; the orthogonal spline recovers its truth
(camera to 1 px / 1 mrad; the field, 0.149 px RMS on the points, to 0.007 px); on plain data
it gains under 3 % and leaves the camera where the plain model puts it.

## Open

- **The camera's σ on real views with the field.** Honest on synthetic
  data (above); on the display rig the field doubles the fold scatter of
  `fx` and `tilt_x` beyond σ. Whether ADR 0026's view-scatter scaling
  covers it is for the implementation to measure.
- **What is left is per view.** With the field, the ring-grid rig sits at
  0.100 px RMS against a 0.073 smooth-fit floor: `sqrt(0.100² − 0.073²)
  = 0.068` px of smooth per-view error that no camera-fixed term removes
  (the blur-dependent part of
  [scheimpflug-residual-model.md](scheimpflug-residual-model.md)). On the
  display rig, 0.248 against 0.213: 0.127 px per view, plus the target
  offsets.
- **The field's physical origin is not identified.** Views that put the
  same image region at different depths, and the top third of the image
  covered, would test a depth-dependent or non-central origin and remove
  the extrapolation the current coverage forces on the border cells.

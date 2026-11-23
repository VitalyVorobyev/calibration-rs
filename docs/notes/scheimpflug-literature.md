# Scheimpflug literature — what is published, and where our results stand

Family: tilted-sensor cameras, blur and focus
([scheimpflug-intrinsics.md](scheimpflug-intrinsics.md),
[tilt-identifiability.md](tilt-identifiability.md),
[tilt-determination.md](tilt-determination.md),
[tilt-from-focus.md](tilt-from-focus.md), [ringgrid-bias.md](ringgrid-bias.md)).
A literature review (2026-11-21) checked our tilt, blur and focus results
against published work: 18 sources fetched, the top 25 claims verified by
three independent reads each, 23 confirmed, 2 refuted. This note keeps
what matters for the library: the published statement of each result we
rely on, the two derivations the review added, and the gaps nobody has
published. Dev-facing; the published review page is
<https://claude.ai/artifact/J4qgtHd8KQ1RYgKgxo2qFK> (private).

**Published** marks a claim checked against the source; **derived** algebra
from this review, not yet run; **hypothesis** what neither shows;
**pointer** a reference recalled but not verified in the review.

## Short answers

1. **Best formalism** (published). Steger's tilt-lens model (IJCV 2017, the
   HALCON `area_scan_tilt_*` models): the tilt is a homography from an
   untilted virtual image plane to the sensor, after distortion, with the
   tilt `τ`, the tilt-axis azimuth `ρ` and the image-side distance `d`, plus
   image- and object-side telecentric variants. OpenCV's `τx, τy`
   (`CALIB_TILTED_MODEL`) and our Scheimpflug sensor slot are its `d = c`
   case.
2. **The tilt is observed only through the distortion** (published). Our
   `K·H(τ) = K'·R(τ)` is stated in print, in fewer places than expected and
   mostly in passing (below).
3. **Steger's `d` is our pupil magnification** (derived): his `d = c`
   example obeys our `tan τ_d = tan τ_s / m_p`. It gives a geometric route
   to `m_p`, second order in `τ` for a tilt about a pixel axis.
4. **A blur gradient across a ring** biases its centre by `1.24·σ·∇σ`
   towards the sharper side (measured in ringgrid; no published
   counterpart). Fed through the focus model the bias is nearly a gauge of
   `cy`, `fy` and the view translations (derived). Measured on the six
   ring-grid cameras: 92 % of it is absorbed, the residual moves by
   0.003 px, and `fy/fx` moves by +3.1 … +4.6e-4 (0.8–1.9 σ) as predicted;
   the `cy` shift has no common sign ([ringgrid-bias.md](ringgrid-bias.md),
   *Where the blur-gradient bias goes*).
5. **Blur as a calibration signal** is published (2000, 2014, 2020), with
   less scope than [tilt-from-focus.md](tilt-from-focus.md); none of the
   published methods has the pupil magnification.
6. **The depth of field follows from the focus model** (derived): the near
   and far limits are planes through the hinge line, the view-camera
   wedge.
7. **A moving entrance pupil** (Kumar & Ahuja, ICCV 2015) is a cheap
   near-central candidate for the depth-coupled residual (hypothesis; it
   predicts a radial term, the data's stable one is anamorphic).

## The gauge, as published

- Kumar & Ahuja (ICPR 2014), abstract: under ideal perspective imaging "a
  geometric calibration of tilt is impossible as tilt parameters are
  correlated with the principal point location parameter". Earlier
  non-frontal calibration avoided it only by tying the principal point to
  the centre of radial distortion; they cite Gennery (IJCV 2006): with a
  non-frontal sensor, radial distortion about the optical centre models all
  the distortion. This is our `p1 ≈ −k1·tan τx` read the other way.
- Steger 2017, Theorem 2: every finite projective camera is an image-side
  telecentric tilt camera with square pixels; Corollary 2: every projective
  camera is some tilt camera. Without distortion the tilt is not
  observable.
- OpenCV (`computeTiltProjectionMatrix`, `matTilt = Pz·Ry·Rx`): the
  review's verifiers derived `K·matTilt = K'·R` from the source; OpenCV
  documents only the trapezoidal effect.
- Pointer: Wang, Shi, Zhang & Liu, Pattern Recognition 41 (2008) 607–615,
  model the tangential terms as a small image-plane rotation (bibliography
  confirmed, content from memory). If so, the closest published form of
  `Δτ = −p1/k1`.
- Kumar & Ahuja, "Generalized Pupil-Centric Imaging and Analytical
  Calibration for a Non-frontal Camera" (CVPR 2014; abstract checked): a
  thick-lens (entrance and exit pupil) imaging model for a lens-sensor tilt
  about any axis, three years before Steger's. It solves the sensor
  rotation analytically *as a function of the centre of radial
  distortion*, then finds that centre by a radial alignment constraint:
  the decentring identity again, used as an initializer.

Distortion does not always fix the tilt. Steger's degeneracies, which
remain with distortion present:

| case | not unique | Steger's fix | here |
|---|---|---|---|
| any tilt model | `sx, sy, c` overparameterize (Remark 1) | hold `sy` | `fx, fy` is that gauge choice |
| image-side telecentric + tilt | `c` (or `m`), `ρ`, `τ`, `sx`, `sy` (Prop. 6): the untilted-to-tilted map is affine | also hold `sx` | no telecentric projection yet: CAM-TELECENTRIC |
| entocentric, tilt about a pixel axis | `τ, d, sx, sy`: `sy₂ = sy₁·cos τ₁/cos τ₂`, `d₂ = d₁·tan τ₂/tan τ₁` (Prop. 5) | hold the pixel aspect ratio | our rig (`τy ≈ 0`) |
| `d` held at `c` | low RMS, wrong tilt: true `d = 3c`, `τ = 5°` fits `1.67°`, `sy` −0.34 % | free `d` | implicit `d = c` |
| `τ → 0` | `d` undefined, ill-conditioned near zero | hold `d` | — |

Steger leaves geometry-dependent degeneracies to the user; the published
procedures found have nothing like ADR 0031's data-driven decision or its
`separable_sigma`.

## Steger's `d` and the pupil magnification (derived)

Steger's example: `tan 1.67° / tan 5° = 0.0292 / 0.0875 = 1/3 = c/d`. Our
thick-lens result ([tilt-from-focus.md](tilt-from-focus.md), *Thick lens*):
`tan τ_d = tan τ_s / m_p`. So `d/c` plays the role of `m_p`, the ratio of
object- to image-side chief-ray slopes. Consequences:

- The library's tilt is `atan(tan τ_s / m_p)`: the physical sensor tilt only
  when `m_p = 1`. The book should say so (BOOK-TILT-MEANING).
- A free `d` with the pixel aspect held could measure `m_p` from geometry,
  the datum the focus route lacks. For a tilt about a pixel axis only
  through `cos τ`; at `τ_d = 5.16°` (the studied rig):

  | `m_p` | 1.0 | 1.1 | 1.5 | 3 |
  |---|---:|---:|---:|---:|
  | `τ_s = atan(m_p·tan τ_d)` | 5.16° | 5.67° | 7.72° | 15.1° |
  | aspect change `1 − cos τ_s/cos τ_d` | 0 | 8.5e-4 | 5.0e-3 | 3.1e-2 |

  10 % in `m_p` needs the aspect ratio to a few 1e-4, free of other
  aspect biases; the blur-gradient bias below is of that size. Prop. 5 is
  specific to an axis-aligned tilt, so a tilt with `ρ` near 45° may do
  better (hypothesis). Study: CAL-TILT-PUPIL.
- Parameterization for that study: `q = c/d = 1/m_p` as `q = exp(η)`
  (positive, `η = 0` at the `d = c` model); fit the plain tilt with `q = 1`
  first and free `q` only after; test small tilts explicitly, where `q`
  and `τ` become strongly correlated and `q` is undefined at `τ = 0`.
  Kumar & Ahuja's pupil-centric model (CVPR 2014, above) is the other
  published thick-lens tilt model to compare against.
- A thick lens modelled through its principal (nodal) points, as some
  Scheimpflug papers do (Guo, Cui & Wu, Sensors 2024), adds nothing a
  calibration can see. The separation of the principal planes only moves
  object space along the axis, which the poses absorb. And the nodal ray
  is right only for points in focus: a point off the plane of focus lands
  where the chief ray through the pupils meets the sensor. So the
  thick-lens quantity the image geometry sees is `m_p`, not the
  principal-plane separation.

## Ring centres under a blur gradient

Published pieces (none quantifies the gradient bias inside one marker):

- Calvet et al. (CVPR 2016): concentric-ring centre from a 2-DOF
  rectifying homography and a symmetry cost between rectified radial
  cross-sections. The cost assumes equal blur on opposite sides, so it
  carries the gradient bias; ringgrid's forward model (ADR-022 there) is
  the better design.
- Song et al. (CVPR 2024, arXiv 2403.04583): the first moment of a
  projected, radially distorted circle in closed form. A centroid is
  immune to the gradient under scatter blur (ringgrid's
  `blur_gradient_ink_centroid` shows it); limits: radial distortion only, a
  binary disk.
- Heikkilä (PAMI 2000), Mallon & Whelan (PRL 2007), Calvet 2016: the
  perspective centre bias, a separate effect, removed by ringgrid's
  two-conic centre ([ringgrid-bias.md](ringgrid-bias.md)).
- CircleFlow (preprint, arXiv 2512.00796): circles probe an anisotropic
  PSF better than checkerboards (the boundary normal sweeps every
  direction); pooled per region, not per ring.
- Shi, "Generic Camera Calibration using Blurry Images" (arXiv
  2603.05159, 2026; abstract checked): estimating feature positions and a
  spatially varying PSF together leaves a global translational ambiguity,
  resolved by aligning to a parametric camera calibrated from a few sharp
  images. Any PSF whose centroid is free trades against the position. A
  centred PSF (ringgrid's Gaussian) avoids it by construction. The
  defocus-coupled centroid shift of [tilt-from-focus.md](tilt-from-focus.md)
  (*Signed defocus*) *is* a PSF with an off-centre centroid. Sharp
  reference views, Shi's anchor, are a second way to separate it, next to
  the two-aperture, two-focus capture.
- Cledat, Rufener & Cucci, "Compensating over- and underexposure in
  optical target pose determination" (Pattern Recognition 2021; abstract
  checked): saturation together with the PSF moves the edges of
  concentric-circle targets with exposure, a bias they estimate from the
  target's own geometry. In ringgrid it mostly changes the stroke width,
  which the per-image printed-radii fit probably absorbs; a symmetric width
  change does not move the centre (hypothesis). Check: do the fitted radii
  follow exposure or contrast across images?

Measured in ringgrid (ADR-022, ADR-029 there): a one-σ fit lands
`b ≈ 1.24·σ·∇σ` towards the sharper side, independent of the marker
radius; a fitted gradient term removes ~90 % on synthetic scenes and left
the pooled calibrated residual on the six ring-grid cameras at 0.137 px
either way. First-order edge theory gives the law with coefficient 1: a
curved Gaussian-blurred edge moves inward by `σ²/2R`, so the centre moves
by `−½∇σ²`, whatever `R`.

**The bias is nearly a gauge** (derived). With the focus model of
[tilt-from-focus.md](tilt-from-focus.md), `σ² = σ0² + κ²D²`,
`D = 1/z − π·(x, y, 1)` in the `K'` frame, and `∇_px D ≈ −π_xy/f'` within a
marker:

```text
b ≈ −(k/2)·∇σ² = (k·κ²/f')·π_xy·D
  = (k·κ²/f')·π_xy·[ 1/z − π_x·x − π_y·y − π_z ]
```

`c/z` is a lateral view translation (exactly, as the `a·κ/z` term of
tilt-from-focus), the `y` term scales `fy`, the `x` term is a skew a
zero-skew `K` cannot take (small: `π_x ≈ 0` here), the constant moves `cy`.
What is left is the variation of `1/z` inside a marker. At the studied
rig's values (`κ ≈ 1.4` px·m, `π ≈ (0, −15, +3)` /m, `f' ≈ 1145` px,
`k = 1.24`): coefficient 0.032 px·m, `cy` ≈ 0.1 px, `fy` relative ≈ 4e-4,
≈ 28 µm per view. Invisible in the residual; it matters for anything
aspect-sensitive (CAL-TILT-PUPIL) and for blur used as an observation
(CAL-TILT-FOCUS).

**Measured** (the full tables and the pinned synthetic test are in
[ringgrid-bias.md](ringgrid-bias.md), *Where the blur-gradient bias goes*):
92.1 % of `|b|²` is absorbed by the camera and the poses (89.5–93.9 % per
camera), and the pooled residual moves by 0.003 px. `Δ(fy/fx)` is
+3.1 … +4.6e-4, the same sign on all six (0.8–1.9 σ), against the 4e-4
predicted. `Δcy` is −0.23 … +0.10 px with no common sign: the principal
point shares the bias with the distortion, the tilt and the poses, so the
`cy` term of the decomposition is not what the full solve chooses. View
translations move up to 0.05 mm laterally and 0.04–0.08 mm along the
axis, and the tilt ≤ 0.44 σ.

## Blur as a calibration signal

| work | signal | result | limits |
|---|---|---|---|
| Aggarwal & Ahuja, ICPR 2000 | relative blur per patch, chart perpendicular to the axis, `σ = a·x + b·y + c` | sensor tilt from blur, ~0.12° detectable against ~0.74° geometrically (25 mm, f/2, 400 mm, 10 µm) | one experiment, no ground truth; the linear model breaks where the PoF crosses the field; front/back ambiguity of unsigned blur |
| Kumar & Ahuja, ICPR 2014 | `1 − NCC` of the observed corner patch against the sharp patch blurred by a thin-lens Gaussian, added to the reprojection cost | tilt σ 6.66° → 0.34° (1 image), 2.86° → 0.17° (5) | focal-stack capture; `k1, k2` already in the baseline; σ are the estimator's |
| Labussière et al., CVPR 2020 | feature `(u, v, ρ)`, `ρ` the signed blur radius of the thin-lens circle of confusion | blur in the residual makes otherwise unobservable optical parameters identifiable | plenoptic; `ρ` derived from depth, not measured |
| active alignment (e.g. US9210306B1) | plane fit to SFR over the field | industrial sensor alignment, same principle | patent, no calibration model |
| Kobayashi, Kawamata & Nishi, Optical Review 30 (2023) 342–349 | defocus (amplitude) and distortion (phase) of one chart image, separated in the frequency domain | sensor tilt from both together, the focus and distortion routes combined | optical-design simulation only (abstract checked) |

[tilt-from-focus.md](tilt-from-focus.md) goes further than all four: the
exact thick lens, the PoF in the `K'` frame, the direction with no datum,
the size needing `pitch/m_p`. The review offers three readings of its
±0.5° unit-to-unit gap, none new: field curvature taken for tilt (flagged
by Aggarwal & Ahuja; our curvature statistic detects it), a decentring the
distortion route reports as tilt and the focus route does not (fits the
−0.84 anti-correlation), and `m_p` varying per unit with internal focusing
(CAL-TILT-PUPIL could measure it).

## A moving entrance pupil: a near-central candidate (hypothesis)

Kumar & Ahuja, "On the Equivalence of Moving Entrance Pupil and Radial
Distortion for Camera Calibration" (ICCV 2015; abstract checked): in a
thick lens the entrance pupil moves along the axis with the field angle,
and that motion is equivalent to radial distortion. The equivalence holds
at one object depth: the pupil's axial offset `δ(θ)` displaces a point by
about `r_px·δ(θ)/z`, so beyond the part a radial term takes up at the mean
depth there is a depth-coupled remainder `r_px·(δ(θ) − δ₀)·(1/z − 1/z̄)`.
With the studied capture's depths (0.16–0.35 m, `1/z` 2.9–6.3 /m), a 1 mm
change of `δ` across the field leaves about 0.5 px at the image edge
(rough estimate): the size of the depth-coupled residual in
[parametric-residual-term.md](parametric-residual-term.md). A moving pupil
is a one- or two-parameter near-central model (`g(ρ) = g₂ρ² + g₄ρ⁴`, below), far
cheaper than a generic non-central camera. Caution: it predicts a *radial*
depth term, while the stable coefficient in
[tilt-from-focus.md](tilt-from-focus.md) is the anamorphic `Re z̄` and the
radial `Re z` changes sign between cameras, so a held-out comparison may
reject it. Backlog CAM-FIELD-ORIGIN.

Tezaur, Kumar & Nestares (CVPR Workshops 2022; read in full) built and
tested such a model for fisheye lenses. Three of its choices carry over:

- **Parameterize by the image radius, not the field angle.** The axial
  shift of the viewpoint is `g(ρ) = g₂ρ² + g₄ρ⁴` of the ideal-image radius
  `ρ`. Unprojection stays closed form (the ray starts at `(0, 0, g(ρ))`);
  projection solves one scalar equation in `ρ`; the plane-based linear
  initialization extends to `g` unchanged. A shift written as `δ(θ)` needs
  a nested solve both ways.
- **No constant term.** A constant shift is the view translation (a gauge).
- **Judge it on held-out points.** On a real 16-camera fisheye array the
  calibration residual moved only 0.39 → 0.30 px. The error predicting
  points triangulated by neighbouring cameras fell 2.13 → 0.93 px
  (2.04 → 0.52 px with six distortion terms). The paper's reading:
  intrinsic and extrinsic errors offset each other in the calibration
  residual.

Their shifts exceed 1 cm between 0° and 90° of field. The shift grows
about as `θ²`, so at our field angles it is far smaller: the 1 mm above is
an assumption to measure, not a value the paper supports. The same paper
models the sensor misalignment as a homography applied after the
distortion, our tilt form. On the array, that homography alone lowered
the held-out error from 2.49 to 2.13 px.

## Depth of field (derived)

For an acceptable blur `σ_max`, `ε = √(σ_max² − σ0²)/κ`, `X = z·(x, y, 1)`:

```text
|1 − π·X| ≤ ε·(n·X)/s     (n the sensor normal, s the spot scale)
near / far limits at s = 1:  (π + ε·n)·X = 1,   (π − ε·n)·X = 1
both contain        {π·X = 1} ∩ {n·X = 0}   (the hinge line)
```

`n·X = 0` is the plane through the projection centre parallel to the
sensor (Merklinger's PTF plane); its intersection with the PoF is the
hinge line, about which the view-camera literature's depth-of-field wedge
opens (Merklinger, *Focusing the View Camera*; Evens, tilt and depth of
field — pointers, the near/far formulas were not retrieved). The fitted
`π, κ, σ0` therefore give the wedge with no further model, exact where
the spot scale `s` is 1 (an untilted camera without distortion) and the
hinge-rule approximation elsewhere (`s` 0.97–1.05 across the image at an
11.5° × 5.7° tilt: the pupil lies in the lens plane, not square to `n`);
`analysis::fit_focus` reports it (`DepthOfField`), the planes with their
covariance, `σ_max` the measured blur including the pixel term
(`ε = √(σ_max² − σ0² − 1/12)/κ`).

Per-feature blur by target: in the library, chessboard and PuzzleBoard
corners give the full PSF covariance `Σ` (`measure_blur`, a fit of the
blurred print around each corner); ring grids give σ and the PSF
anisotropy from ringgrid 0.18's photometric fit, mapped to the same `Σ`
(`Feature::blur` on photometric centres). Alternatives in the literature:
an isotropic σ from the edge profiles, as Kumar & Ahuja used, or per
square with the ISO 12233 slanted edge (pointer). Requirement
([tilt-from-focus.md](tilt-from-focus.md), *Data requirement*): blur well
above `σ0` over the field, depth variation within views, coverage of the
rows the PoF crosses.

## The blur field of a planar target (derived)

The gradient the bias law needs is available without a camera model,
because one image of a planar target has a blur field of known form.

Take rays `r` in the lens frame and `x̃ = r / r_z`. A target point on the
plane `n·X = d` has `1/z = n·x̃ / d`, and the PoF term is `π·x̃`. So the
signed defocus

```text
D = 1/z − π·x̃ = (n/d − π)·x̃
```

is affine in `x̃`. With `x̃` affine in undistorted, untilted pixels, one
image's blur is

```text
σ²(u, v) = σ0² + (α·u + β·v + γ)²
```

That is three coefficients per image and a `σ0` shared by a capture. No
intrinsics, poses or tilt enter. The bias law needs no sign: `b = 1.24·σ·∇σ
= 0.62·∇(σ²)`, towards the sharper side. So the unsigned `σ` a detector
measures is enough.

**Where it is approximate.**

- **Tilt.** On a tilted sensor, `x̃ = (K·H(τ))⁻¹·p` normalized, so `D` in
  pixels is a ratio `l·p / m·p` of affine functions. The denominator varies
  across the image by about `tan τ·tan θ`: about 3 % at the field edge for
  5° of tilt. That is negligible for a bias correction. Two more
  coefficients make it exact.
- **Distortion.** It breaks the affinity in raw pixels. Fit in the
  detector's undistorted working frame when it has one.
- **Other blur.** Field curvature adds a radial term, and aberration-driven
  PSF changes are not modelled. The fit's residual shows when either
  matters.

**Why this, rather than a local plane.** ringgrid estimates `∇σ` today
(ADR-029, `attach_blur_gradient_bias`) as a least-squares plane through `σ`
of a marker and its few nearest neighbours, keeping the magnitude only. That
serves a confidence. A correction needs more:

- **Less noise.** The per-image field pools every marker of the image, where
  the neighbour plane uses a handful. That is what matters at low
  resolution.
- **A direction.** The field gives the gradient's direction, not just its
  size.
- **The fold.** Near the line where the target crosses the PoF, `σ` is
  V-shaped. A linear plane flattens the fold; the field models it.

**What ringgrid measured** (its ALGO-018/019, closed as rejected on
2026-11-22; its numbers, synthetic, `blur_gradient_bias_matrix`). With the
neighbour-plane gradient, subtracting the law leaves 0.50 of the bias and
a refit with the gradient fixed leaves 0.46. With the true gradient the
law leaves 0.29 and the refit 0.16, at 2.1–2.3× the fit time and up to
2.6× the centre scatter. So the gradient estimate is the limit, and even
a perfect field removes about 70 % by the law. ringgrid keeps neither the
refit nor a prior input.

**The split this gives** (recorded in CAL-TILT-FOCUS):

- **The detector reports, it does not correct.** It gives the centre and
  the blur (`σ`, the anisotropy, their covariance). The bias law belongs
  to the estimator: 1.24 for ringgrid's one-σ fit, none for a moment
  centre under scatter blur, unmeasured for X-corners.
- **The bias stays in the points.** It is a near-gauge of `fy` and the
  poses (92 % absorbed, [ringgrid-bias.md](ringgrid-bias.md)), so a
  calibration cannot estimate it from geometry, and for the residual it
  does not matter. On the studied rig it is a systematic of about 4e-4 in
  `fy/fx`.
- **A caller that needs the aspect ratio subtracts the law** `0.62·∇(σ²)`
  with a smooth gradient: the per-image field above, or the fitted focus
  model's (`(α, β, γ)` is `κ·(n_j/d_j − π)` mapped through the camera).
  That is a study step (CAL-TILT-PUPIL), not a library default.
- **The calibration uses blur as an observation.** The PoF `π`, `κ`, `σ0`,
  the tilt direction and the depth of field need the poses and intrinsics.
  The calibration takes the per-feature blur with its covariance and never
  models the bias law.

A correction is checked through `fy/fx` and the poses: the calibration
residual does not move.

## Refuted in the review

- "For image-side telecentric tilt the principal point is fixed only by the
  distortion and stays at its initial value" (0–3): a misreading of Steger;
  the rule is to hold `sx`.
- "Scheimpflug models before 2016 ignore pupil magnification" (PeerJ
  preprint 1887, 0–3): refuted for want of an accessible text, not shown
  false. Guo, Cui & Wu (Sensors 2024) describe that preprint (Sinharoy,
  Rangarajan & Christensen, 2016) as a pupil-centred ray-tracing model,
  which points the other way. The pupil magnification matters here
  regardless.

## Not found in the literature

- The ring-centre bias under a blur gradient (ringgrid's study is the
  reference).
- Field curvature separated from tilt in a blur-field fit.
- A Scheimpflug procedure that decides from the data whether the tilt is
  determined (ADR 0031).

## Sources

Verified in the review:

- C. Steger, "A Comprehensive and Versatile Camera Model for Cameras with
  Tilt Lenses," IJCV 123, 2017 —
  <https://link.springer.com/article/10.1007/s11263-016-0964-8>.
- A. Kumar, N. Ahuja, tilted-sensor calibration from a focal stack with a
  defocus term, ICPR 2014 —
  <https://migrate2wp.web.illinois.edu/html-files-to-import/publications/Kumar_FocalStackCalibration_ICPR_2014.pdf>.
- M. Aggarwal, N. Ahuja, "Estimating Sensor Orientation in Cameras," ICPR
  2000 —
  <https://migrate2wp.web.illinois.edu/html-files-to-import/publications/Estimating%20Sensor%20Orientation%20in%20Cameras.pdf>.
- OpenCV calib3d, `computeTiltProjectionMatrix`, `undistortPoints` —
  <https://codebrowser.dev/opencv/opencv/modules/calib3d/src/undistort.dispatch.cpp.html>.
- L. Calvet, P. Gurdjos, C. Griwodz, S. Gasparini, "Detection and Accurate
  Localization of Circular Fiducials under Highly Challenging Conditions,"
  CVPR 2016.
- Song et al., first moment of projected distorted circles, CVPR 2024 —
  <https://arxiv.org/abs/2403.04583>.
- M. Labussière et al., "Blur Aware Calibration of Multi-Focus Plenoptic
  Camera," CVPR 2020 — <https://arxiv.org/abs/2004.07745>.
- CircleFlow, preprint, December 2025 — <https://arxiv.org/abs/2512.00796>.
- A. Legarda, A. Izaguirre, N. Arana, A. Iturrospe, Scheimpflug camera
  calibration with the Scheimpflug angle as an intrinsic, IWECMS 2011
  (abstract only) — <https://ieeexplore.ieee.org/document/5952376>.

Checked later the same day (bibliography and abstract only), from a
second review of the same questions:

- A. Kumar, N. Ahuja, "Generalized Pupil-Centric Imaging and Analytical
  Calibration for a Non-frontal Camera," CVPR 2014, pp. 3970–3977 —
  <https://migrate2wp.web.illinois.edu/html-files-to-import/publications/Kumar_Calibration_CVPR_2014.pdf>.
- A. Kumar, N. Ahuja, "On the Equivalence of Moving Entrance Pupil and
  Radial Distortion for Camera Calibration," ICCV 2015 —
  <https://openaccess.thecvf.com/content_iccv_2015/html/Kumar_On_the_Equivalence_ICCV_2015_paper.html>.
- Z. Shi, "Generic Camera Calibration using Blurry Images," arXiv
  2603.05159, 2026 — <https://arxiv.org/abs/2603.05159>.
- E. Cledat, M. Rufener, D. A. Cucci, "Compensating over- and
  underexposure in optical target pose determination," Pattern
  Recognition, 2021, DOI 10.1016/j.patcog.2021.107930.
- K. Kobayashi, T. Kawamata, K. Nishi, "Measurement of image sensor tilt
  based on amplitude and phase of image," Optical Review 30, 2023,
  342–349 — <https://link.springer.com/content/pdf/10.1007/s10043-023-00802-2.pdf>.
- R. Tezaur, A. Kumar, O. Nestares, "A New Non-Central Model for Fisheye
  Calibration," CVPR Workshops 2022, pp. 5218–5227 (read in full,
  2026-11-22) —
  <https://openaccess.thecvf.com/content/CVPR2022W/OmniCV/papers/Tezaur_A_New_Non-Central_Model_for_Fisheye_Calibration_CVPRW_2022_paper.pdf>.
- D. Guo, J. Cui, Y. Wu, "Linear-Structured-Light Measurement System
  Based on Scheimpflug Camera Thick-Lens Imaging," Sensors 24 (2024) 5124
  (text read, equations not) — <https://doi.org/10.3390/s24165124>.
- Not confirmed: "Beyond Reprojection Error: Camera Calibration with 3D
  Targets," arXiv 2608.05066 (entry exists; authors and abstract not
  checked).

Pointers (not verified): Gennery, IJCV 2006; Wang et al., Pattern
Recognition 2008; Merklinger, *Focusing the View Camera*; L. Evens,
<https://sites.math.northwestern.edu/~len/photos/pages/tilt_dof.pdf>;
Heikkilä, PAMI 2000; Mallon & Whelan, PRL 2007; Kee et al., ICCP 2011
(spatially varying blur); Mannan & Langer, CRV 2016 (blur calibration for
depth from defocus); Delbracio et al., IJCV 2012 (sub-pixel PSF);
Louhichi et al., Meas. Sci. Technol. 2007; Astarita, Exp. Fluids 2012;
Fasogbon et al. 2015; ISO 12233.

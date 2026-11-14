# Tilt bias — does the distortion-derived tilt absorb the defocus-coupled displacement?

Family: single-camera intrinsics on a tilted sensor
([scheimpflug-intrinsics.md](scheimpflug-intrinsics.md)). On the six
ring-grid cameras the routed tilt (Brown-Conrady `k1, k2` + two-axis tilt,
identified only through the distortion) and the focus-derived tilt
([tilt-from-focus.md](tilt-from-focus.md), at a common `pitch/m_p` of
5.29 µm) disagree by about ±0.5° per camera, and anti-correlate across
cameras (−0.84). The residual carries a displacement coupled to defocus
blur and depth that the camera model lacks
([parametric-residual-term.md](parametric-residual-term.md)). Hypothesis
(i) of the focus note: the routed tilt absorbs part of that displacement.
This note tests it on the capture and maps, on synthetic cameras, where a
distortion-derived tilt is contaminated by such a displacement.
Dev-facing; aggregate numbers only.

**Derived** marks algebra, **measured** a number the harness prints,
**hypothesis** what neither shows.

Harness: bench example `scheimpflug_model_study`, `tiltbias.rs` (`--parts
tiltbias`: the laws, the iterated joint fit, the gauge transport, the
real-data study) and `tiltbias_synth.rs` (`--parts tiltsweep`: the
synthetic bias map and its Monte-Carlo), unit-tested:

```text
cargo run --release -p vision-calibration-bench --features tier-b \
    --example scheimpflug_model_study -- \
    --registry crates/vision-calibration-bench/registry/private.json \
    --out <dir> --parts tiltbias scheimpflug_rig_ringgrid_cam0 … cam5
cargo run … -- --registry <any> --out <dir> --parts tiltsweep --draws 16
```

(about 4 minutes and 1 minute.)

## Short answers

1. **Mechanism** (derived, then measured). A camera parameter's pixel
   direction depends only on the image position; a pose's also on depth.
   So the route takes, of an unmodelled displacement, mainly its
   image-fixed part, and the tilt — determined only by the small
   distortion-tilt interplay — is the camera parameter that part moves
   most in units of its σ. A law for the displacement fixes its
   image-fixed part from its depth-varying part; this capture cannot test
   that part, so the size of the correction depends on the law.
2. **Hypothesis (i): partly supported, not sufficient** (measured). Every
   law that leaves the tilt identified moves each camera's tilt towards
   its focus tilt: across the six, the move correlates with the
   focus − route gap at +0.74 … +0.95 (+0.74 for the depth-only law, which
   uses no blur). It closes 13–51 % of the gap (by the law), shrinks the
   routed tilts' spread across cameras from 0.0060 to 0.0017–0.0044 rad
   and the implied `pitch/m_p` spread from 11 % to 6–9 %. It leaves the
   tilt *direction* gap (τy) untouched, a τx gap of 0.26–0.41° RMS (χ²
   120–245 on 5 dof) and the anti-correlation (−0.65 … −0.92). Holding
   the tilt at the focus value costs 42–84 % of held-out variance under
   every law: the distortion data reject it either way.
3. **Signed-defocus laws leave the tilt unidentified** (measured): a
   centroid shift ∝ signed defocus (`g × deg 0–1` or `0–3`) multiplies
   σ(tilt) by 12–30 at the route; its nonlinear fit walks to a tilt of
   the opposite sign, overfits and fails held out. If the coupling is of
   that form, this capture cannot give a distortion tilt at all.
4. **Synthetic, general** (measured): a displacement of the measured form
   and size (0.10 px left in the residual) biases the plain route's tilt by
   0.06–0.17° (5–13 noise σ) at a 2–15° tilt, 0.44–1.8° with weak
   distortion, and almost nothing at zero tilt; random laws of the same
   visible size bias it by a median 3.5–15 noise σ in every regime, and so
   does a random image-fixed field (5–14σ). The absolute bias scales with
   σ(tilt): a regime that determines the tilt well is not protected, only
   less biased in degrees. The library's view-jackknife scale inflates the
   stated σ about 4× on such data, leaving 2–3 stated σ.
5. **For the tilt-determination rule**: a small σ is not enough. Add a
   systematic-sensitivity term (the tilt's shift under a nuisance
   depth/blur law, which needs no datum) and, where the detector reports
   blur, the focus-direction check; export the tilt with both and a
   determination status that can say "determined, systematics dominate".

## The fit

The route's camera (`fx fy cx cy k1 k2 τx τy`), every pose and a law's
coefficients, from the route's solution, by Gauss-Newton with step halving
iterated to convergence (camera steps below 0.001σ, or a stall below
0.05σ), each iterate re-linearized (`linfield`, poses eliminated per view,
then back-substituted), with Huber weights at 1 px as the route.

- **A law** is a per-point covariate times complex monomials of the image
  position about the centre (scale: half the width). Covariates: `g =
  κ·(1/z − π·(x, y, 1))`, the signed defocus about the plane of sharp focus
  (PoF) fitted to the detector's blur in the `K'` frame (`focus.rs`);
  `|g|` (minus its median, or raw); the geometric blur (blur² quadratic in
  image position and inverse depth); depth `z̄/z − 1`; and the
  depth-varying parts of `g`, `|g|` (each minus its degree-4 regression on
  the image position). The monomials are evaluated at the current
  projection; their dependence on the camera and poses enters the
  Jacobian.
- **Covariates per point, held.** The blur model and the PoF are fitted
  to the blur at the start's geometry, and each point's covariates are
  held through the fit. Re-evaluating them at each iterate couples the
  covariate to the poses being fitted (`a·κ/z` follows a translation
  exactly as the projection does, the gauge of the focus note), and the
  fit then exploits that coupling: training residual below the
  calibration-free floor, held-out 2–5× worse (measured on the first
  attempts). Covariates are measurements here (blur and a geometry known
  far better than the term needs). As a sensitivity, the covariates are
  refitted at the solution and the fit repeated: the camera moves ≤ 0.31σ
  for every identified law (≤ 0.21σ but one).
- **Gauge.** The coefficients carry a ridge of σ = 1 (displacement per px
  of covariate at the image edge), which binds only the near-gauge
  combinations. Holding the tilt (the reverse experiment) moves the start
  along the exact gauge of a distortion-free tilt: `K'` and the
  `K'`-frame poses kept, `fx, fy, cx, cy` and the lens-frame poses
  re-expressed (`tiltbias::transport`, the rotation composed from axis
  rotations; test `transport_is_a_gauge_without_distortion`, 1e-8 px).
- **Checks** (measured): the plain model re-converged by this fit equals
  the library route to ≤ 0.01σ on all six; held-out (4 folds by view, the
  covariate models from the training folds) reproduces the linearized
  field harness (geometric blur + depth: 0.0824 against 0.0833 px).

## Real data (measured)

### Held out, and which laws leave the tilt identified

| law | held-out pooled px | Δ var vs route | σ(τx) with / without the law |
|---|---:|---:|---|
| route (plain) | 0.1426 | | |
| `g × deg 0–3` | 0.3598 | +537 % | 14–18 (22–30 at the route): not identified |
| `g + |g| × deg 0–3` | 0.4807 | +1036 % | 11–16 (17–24): not identified |
| `g × deg 0–1 + |g| × deg 0–3` | 0.2891 | +311 % | 11–17 (12–18): not identified |
| `(|g| − med) × deg 0–3` | 0.0993 | −52 % | 0.54–0.86 |
| `|g| × deg 0–3` | 0.0914 | −59 % | 0.69–0.97 |
| geometric blur 0–3 + depth 1–4 | **0.0824** | −67 % | 0.52–0.71 |
| depth × deg 1–4 | 0.1237 | −25 % | 0.80–0.97 |
| `g, |g|` depth-varying only × deg 0–3 | 0.1051 | −46 % | 0.69–0.81 |

A signed-defocus law's image-fixed part, `−κ·π·(x, y, 1)` times the
monomials, reproduces the tilt's distortion signature: with it free the
tilt walks by +0.09 … +0.22 rad, to +0.003 … +0.12 (from −0.09), `fx` drops 30–67 px, the
training residual falls to 0.035–0.06 px (below the 0.073 floor) and
held-out fails. The linearized one-step fit of the focus note (0.0746) sat
in that flat valley next to the route. The rows below use only the
identified laws.

### The camera's change (law − route, full data, ± fold jackknife of the change)

Geometric blur × deg 0–3 + depth × deg 1–4 (the parametric note's law,
the best held-out):

| camera | Δτx rad | Δτy rad | Δcx px | Δcy px | Δk1 | Δfx px |
|---|---|---|---|---|---|---|
| cam0 | +0.0083 ± 0.0019 | +0.0021 ± 0.0003 | +3.8 ± 0.3 | −1.9 ± 0.3 | +0.015 ± 0.004 | +3.4 ± 2.7 |
| cam1 | +0.0014 ± 0.0014 | −0.0010 ± 0.0006 | −1.3 ± 1.1 | +0.2 ± 0.3 | +0.011 ± 0.002 | −2.8 ± 2.4 |
| cam2 | +0.0012 ± 0.0008 | +0.0028 ± 0.0008 | +5.9 ± 1.6 | −1.2 ± 0.4 | +0.017 ± 0.004 | −4.3 ± 1.6 |
| cam3 | +0.0040 ± 0.0013 | +0.0009 ± 0.0001 | +1.6 ± 0.2 | −1.9 ± 0.2 | +0.020 ± 0.004 | −0.8 ± 2.6 |
| cam4 | −0.0010 ± 0.0010 | +0.0007 ± 0.0010 | +0.7 ± 1.2 | −4.9 ± 0.3 | +0.022 ± 0.002 | −6.4 ± 2.8 |
| cam5 | −0.0004 ± 0.0021 | +0.0010 ± 0.0006 | −0.2 ± 1.0 | −2.0 ± 0.4 | +0.026 ± 0.002 | −7.4 ± 2.0 |

(Route: stated σ of τx 0.0012–0.0018 rad, fold jackknife of τy 0.0003–0.0006 rad.) `k1` moves by
+0.011 … +0.026 on every camera (k1 ≈ −0.41; 2–6 %, 3–12 jackknife σ) and
`cy` by −1 … −5 px on five; τx by up to 0.0083 rad (0.48°, 4.6 stated σ).

Δτx (rad) by law — what each assumes of the image-fixed part:

| camera | focus − route at 5.29 µm | `|g| − med` | `|g|` | geo + depth | depth | depth-varying only |
|---|---:|---:|---:|---:|---:|---:|
| cam0 | +0.0132 | +0.0085 | +0.0059 | +0.0083 | +0.0047 | +0.0073 |
| cam1 | −0.0061 | +0.0002 | −0.0019 | +0.0014 | +0.0030 | +0.0034 |
| cam2 | −0.0070 | −0.0004 | −0.0032 | +0.0012 | +0.0018 | +0.0044 |
| cam3 | +0.0086 | +0.0036 | +0.0018 | +0.0040 | +0.0035 | +0.0069 |
| cam4 | −0.0095 | −0.0013 | −0.0083 | −0.0010 | +0.0006 | +0.0038 |
| cam5 | −0.0046 | −0.0009 | −0.0051 | −0.0004 | −0.0009 | +0.0018 |
| slope on the gap, corr | | +0.36, +0.95 | +0.51, +0.93 | +0.31, +0.93 | +0.13, +0.74 | +0.13, +0.85 |

The direction of the move agrees with the gap on every law (this could
have come out uncorrelated: the laws are fitted to positions, the gap is
from blur); the size does not. Cameras 0 and 3 move by +0.002 … +0.009 under
every law, camera 4 between −0.008 and +0.004 depending on the law's
centring: equal held-out, different tilts. With six cameras a correlation
of 0.93 is p ≈ 0.007, 0.74 is p ≈ 0.09.

### The focus comparison with the refitted tilt

| law | `pitch/m_p` µm, sd | χ² (5 dof) | focus − dist τx RMS | τy RMS | sd across cameras: focus / dist τx | corr |
|---|---|---:|---:|---:|---|---:|
| route | 5.29, 11 % | 92 | 0.0086 | 0.0108 | 0.0038 / 0.0060 | −0.84 |
| `|g| − med` | 5.19, 7 % | 167 | 0.0052 | 0.0112 | 0.0035 / 0.0024 | −0.78 |
| `|g|` | 5.38, 6 % | 129 | 0.0045 | 0.0111 | 0.0036 / 0.0017 | −0.65 |
| geo + depth | 5.15, 7 % | 245 | 0.0056 | 0.0111 | 0.0035 / 0.0028 | −0.86 |
| depth | 5.16, 9 % | 186 | 0.0072 | 0.0107 | 0.0036 / 0.0044 | −0.92 |
| depth-varying only | 5.02, 9 % | 120 | 0.0066 | 0.0109 | 0.0034 / 0.0042 | −0.82 |

The χ² grows because the refitted tilts' jackknife shrinks faster than
the pitch spread. With the blur laws the routed tilts become *more*
uniform across the six units than the focus tilts (sd 0.0017–0.0028
against 0.0035): if six units of one design have similar tilts
(hypothesis), the larger spread has moved to the focus side, where the
unit's `m_p`, its focus setting and the field curvature enter (the focus
note's hypotheses (ii), (iii)).

### Reverse: the tilt held at the focus value

Both tilt axes held at the focus-derived value (common `pitch/m_p`
5.29 µm), the rest refitted from the gauge-transported route; held-out
pooled px:

| law | tilt free | tilt held at focus | Δ var held vs free |
|---|---:|---:|---:|
| route (plain) | 0.1426 | 0.1701 | +42 % |
| `|g| − med` | 0.0993 | 0.1299 | +71 % |
| geo + depth | 0.0824 | 0.1117 | +84 % |
| depth-varying only | 0.1051 | 0.1332 | +61 % |
| `g × deg 0–1 + |g| × deg 0–3` | (not identified) | 0.1327 | |

Held, the principal point follows the tilt along its gauge (`Δcx` −15 …
+32 px, the largest on cam5, whose τy differs by 1.3°). The laws do not
make the focus tilt acceptable to the distortion data; they make it
relatively *less* acceptable, because they sharpen the distortion tilt.
The direction (τy) half of the disagreement needs no pitch and no law
touches it.

### Verdict on hypothesis (i)

**Partly supported, and not the explanation** (measured): the
defocus/depth coupling biases the distortion-derived tilt on this capture,
in the direction of the focus tilt on every camera, by up to 0.5° (cam0),
and accounts for 13–51 % of the τx disagreement and none of the τy one.
The size is law-dependent because the law's image-fixed part is not
testable with this capture; with a signed-defocus law it is not even
bounded. The remaining ±0.3–0.4° (τx) and the 0.2–1.3° direction
disagreement stay unexplained; the routed tilts after the correction are
the more uniform set.

## Synthetic: the bias map (measured)

The library's camera with a thin lens of the same tilt for the PoF
(`focus::Lens`, F = 6 mm, f/1.22, focus at the harmonic middle of the
depth range), 720×540, an 8 × 7 board (pitch scaled with the field of
view), 20 views ±0.45 rad, 4 geometries per cell. Displacement: the
`(|g| − med) × deg 0–3` law with the rig-mean coefficients measured above,
scaled so the plain route leaves 0.10 px RMS of it (the measured
`√(0.1426² − 0.0993²)`). Bias by one Gauss-Newton step at the truth
(linear in the displacement; the iterated fit agrees to 1 % in tilt, test
`linear_response_matches_the_iterated_route`). σ: noise only, 0.07 px per
axis. Baseline: tilt −5.2°, HFOV 35°, corner distortion −0.065, depth
0.17–0.33 m.

| cell | σ(τx) | measured form: τx bias | τy | cy px | k1 | random laws \|τx\|/σ median / 90 % | \|τy\|/σ |
|---|---:|---|---|---|---|---|---|
| baseline | 0.013° | −0.135° (−11σ) | −0.006° | −2.7 (−6σ) | −0.004 | 10 / 20 | 3.2 / 7 |
| tilt 0° | 0.012° | +0.016° (+1σ) | −0.020° | −0.2 | −0.007 | 4.3 / 9 | 4.2 / 10 |
| tilt −2° / −10° / −15° | 0.012–0.015° | −0.060 / −0.164 / −0.169° (−5 … −12σ) | ≤ 0.03° | −1.2 / −3.3 / −3.5 | ≈ 0 | 6–11 / 16–20 | 3–4 |
| tilt (0°, 5°) | 0.011° | +0.008° | +0.037° (+3σ) | −4.0 (−10σ) | −0.027 (−17σ) | 3.5 / 8 | 11 / 26 |
| tilt (−10°, 10°) | 0.013° | −0.090° (−7σ) | +0.090° (+8σ) | −4.0 | −0.011 | 12 / 24 | 11 / 23 |
| corner distortion −0.005 / −0.02 | 0.29 / 0.055° | −1.76 / −0.44° (−6 / −8σ) | +1.09 / +0.20° | −35 / −9 | ≈ 0 | 5–7 / 13–16 | 12 / 21 |
| corner distortion −0.15 | 0.020° | −0.059° (−3σ) | −0.060° | −1.0 | ≈ 0 | 4 / 10 | 11 / 21 |
| HFOV 20° / 60° / 90° | 0.037 / 0.026 / 0.051° | −0.04 / −0.26 / −0.34° (−1 / −10 / −7σ) | −0.12 / +0.13 / +0.34° | −2.1 / −2.6 / −2.0 | ≈ 0 | 3.5–7.5 / 8–16 | 9–16 / 17–28 |
| depth 0.23–0.27 / 0.12–0.5 m | 0.013 / 0.015° | −0.163 / −0.069° (−13 / −5σ) | ≈ 0 | −3.6 / −0.9 | ≈ 0 | 11 / 7 (median) | 3–4 |
| 5 / 10 / 40 views | 0.026 / 0.017 / 0.008° | −0.12 / −0.14 / −0.12° (−4.5 / −8 / −14σ) | ≈ 0 | −1.9 / −2.7 / −2.4 | ≈ 0 | 6.5 / 9 / 15 (median) | 3–5 |

**Controls** (same visible size, random forms, |τx bias|/σ median / 90 %):
an image-fixed field of degree 0–4: 5–14 / 13–34 in every cell; the
depth-varying parts of `g`, `|g|` only: 2–6 / 6–14. The route hides most
of a displacement: its total RMS is 1.8–4.6 times the visible part for the
`|g|` law, 7–9 times for an image field.

**Monte-Carlo** (16 draws per cell, a new geometry and noise each, the
library's seeded route then tight refines on both backends, which agree
to every printed digit):

| cell | plain route, null | plain route, measured form: τx | cy px | joint fit with the law, measured form |
|---|---|---|---|---|
| baseline | ≤ 0.4σ, scatter/stated 0.72–1.10 | −0.153 ± 0.018° (linear −0.146), −3.3 stated σ | −2.74 ± 0.33 (linear −2.61) | ≤ 0.5σ, 0.78–1.24 |
| tilt 0° | ≤ 0.4σ | −0.016 ± 0.022° (−0.020) | −0.57 ± 0.33 (−0.58) | ≤ 0.4σ |
| corner distortion −0.02 | ≤ 0.4σ | −0.50 ± 0.05° (−0.45), −2.3 stated σ | −10.0 ± 1.0 (−8.8) | ≤ 0.3σ |
| HFOV 60° | ≤ 0.4σ | −0.274 ± 0.021° (−0.272), −2.7 stated σ | −2.70 ± 0.26 (−2.71) | ≤ 0.4σ |

The library's stated σ is 3.7–4 times the noise-only σ on displaced
data: its view-jackknife scale (the covariance's `k`) responds to a
displacement that differs from view to view. It covers part of the bias,
leaving 2–3 stated σ in tilt.

**Regimes** (measured, then the reading):

- The bias of the *measured* form is along the tilt's own axis and grows
  with the tilt (0 at 0°, 0.06° at 2°, 0.16° at 10–15°): the blur
  pattern's image-fixed average is mirror-symmetric about the tilt plane,
  and so is the tilt's signature. A random law biases both axes at any
  tilt.
- In noise σ the bias is about regime-independent (median 4–15σ for
  0.10 px visible against 0.07 px noise); in degrees it scales with
  σ(tilt): 0.04–0.34° with strong distortion, 0.44–1.8° with weak (corner
  −0.02 … −0.005), up to 0.34° at a 90° field. More views do not help
  (the bias stays, σ shrinks: −4.5σ at 5 views, −14σ at 40); a wide depth
  range does (−5σ against −13σ for a narrow one), because more of the
  displacement is then depth-varying.
- Rule of thumb (from the scaling, hypothesis beyond the sweep): `|bias|/σ
  ≈ ρ·(δ/σ_n)·√(2N)`, `δ` the visible systematic RMS, `σ_n` the noise per
  axis, `N` the points (about 1000 here), `ρ` the overlap with the tilt's
  signature (0.05–0.2 here, median about 0.15). At `N ≈ 1000` a
  systematic of a tenth of the noise is already most of a σ.

So a distortion-derived tilt is materially contaminated by an
unmodelled defocus-coupled displacement whenever the tilt is non-zero and
the residual carries any visible systematic part; not specifically in
some regimes. Where the tilt is poorly determined the contamination is
large in degrees; where it is well determined it is small in degrees but
many σ.

## For the tilt-determination rule

The rule under design (estimate the tilt when the data determine it, else
a prior, else zero) reads "determined" from the tilt's σ or the Fisher
information. By the above, σ alone does not protect: the bias in σ units
is about the same in every regime. Safeguards that need no metric datum,
in order of cost (recommendations; evidence as cited):

1. **A systematic-sensitivity term** (data only). Refit the camera with a
   nuisance displacement law and take the tilt's shift as `σ_sys`: the
   depth law (`z̄/z − 1` × monomials, geometry only, any detector) and, when
   the detector reports blur, the `|g|` law. On this capture `σ_sys`
   would be 0.001–0.008 rad against a stated 0.0012–0.0018, and its sign
   agrees with the focus gap. "Determined" then needs `√(σ² + σ_sys²)`
   below the threshold, not σ. A signed-defocus law cannot serve: it makes
   the tilt unidentified (itself a finding worth reporting).
2. **A residual-systematics statistic**: the held-out gain of the nuisance
   law (here 25–67 % of the variance). Where it is significant, the stated
   σ of the tilt is a lower bound.
3. **The focus-direction check** (when the detector reports blur): the
   azimuth of the PoF against the tilt's azimuth, no pitch needed. Here it
   flags all six cameras (2–14°, 3.5–36 jackknife σ) and no law reduces
   it. It checks the tilt's direction only, with a sensitivity of `|τ|`
   times the azimuth σ; the size needs `pitch/m_p`.
4. **Export**: the tilt with its stated σ, `σ_sys` and the law it came
   from, the residual-systematics statistic, the focus-direction
   difference when available, and a determination status with a third
   value, "determined, systematics dominate", beside "determined" and
   "not determined". The view-jackknife scale already in the covariance
   (`k`) catches part of it (about 4× on the synthetic displaced data) and
   should be reported with it.

## Open, and what would settle it

- **The image-fixed part of the coupling** decides the size of the
  correction and is not testable here: the same poses at two apertures
  (a defocus-coupled term scales with the blur, a chief-ray term does not)
  measure the coupling directly; then the law need not be assumed.
- **The τy (direction) disagreement** is untouched by every law: a lens
  rotated 180° in its mount (a lens element's field tilt turns with it, a
  sensor tilt does not) and the datasheet `m_p` (focus note).
- The synthetic map uses the measured `|g|` law; a signed law would bias
  more (it can mimic the tilt outright) — not mapped, since it cannot be
  fitted here.

Backlog: `CAL-TILT-SYSTEMATIC` (the safeguard), `CAL-TILT-FOCUS` (updated).

# Tilt identifiability — a synthetic study of the Scheimpflug fit

Family: single-camera intrinsics on a tilted sensor (the planar route of
[scheimpflug-intrinsics.md](scheimpflug-intrinsics.md)). Question: is the
tilted fit stable, does it need a change of variables, and how do the tilt
and the tangential (and thin-prism) terms trade off? Everything here is
synthetic, over a sweep of camera, lens, tilt and view design, so no claim
rests on one dataset. The gauge is published (Kumar & Ahuja 2014; Steger
2017, Theorem 2), with Steger's degeneracies that remain under distortion:
[scheimpflug-literature.md](scheimpflug-literature.md).

Harness: `crates/vision-calibration-bench/examples/tilt_identifiability/`
(`scene.rs` cameras and view designs, `model.rs` the camera in four
parameterizations and the Fisher information, `solve.rs` the library solves,
`study.rs` the tables). Every number below is printed by

```text
cargo run --release -p vision-calibration-bench --example tilt_identifiability -- \
    [--parts analytic,fisher,mc,basin,solver] [--quick]
```

(about 25 minutes in full on 12 cores, 20 of them the basin; `--quick` is a
smoke run of a few minutes). Its tests
(`cargo test -p vision-calibration-bench --example tilt_identifiability`)
pin the analytic results below, and
`vision-calibration-core::models::sensor::tests::tilt_is_a_rotated_pinhole`
pins the factorization in the crate that defines the homography.

**Measured** marks a number the harness prints; **derived** a result of the
algebra below that the harness verifies; **hypothesis** what neither shows.

## Short answers

1. *Is the fit stable?* Locally, yes wherever the data determines the tilt
   at all: from the truth the solve's scatter matches the stated σ on both
   backends (ratio 0.84–1.15 in every cell with a well-determined tilt), and
   the seeded route reaches the reference minimum from every seed up to
   30 % / 10° / 100 px off (table B); past that its own bounds, not the
   problem, limit it. Where the lens distortion is weak the tilt is poorly
   determined, because it is determined only through the distortion (below), and the
   default stopping rule stops along the valley (scatter 0.7–0.8 of σ from a
   start at the truth, up to 0.66σ short of the minimum from a seed).
2. *Do we need a substitution?* No. The tilt's marginal σ is invariant under
   any change of the other coordinates; the substitutions move the
   correlation from one coordinate to another, and which coordinates are the
   well-conditioned ones depends on the regime (F5). The gauge-aligned one
   (P3) cuts iterations in the weak-distortion valley but loses elsewhere
   and adds four wrong-basin failures; no substitution reaches a minimum
   the raw coordinates miss (their only misses are two stopping-rule
   valley stops).
3. *Tilt ↔ tangential?* Degenerate to leading order, for a reason that is
   exact algebra (a decentred radial distortion *is* a tangential term), so
   freeing `p1, p2` with the tilt multiplies σ(tilt) by 3.4–100; the default
   (hold `p1, p2` under a tilt) is right. Its price: a real decentring is
   reported as tilt, `Δτx = −p1/k1`.

## The exact factorization: without distortion the tilt is a gauge

The library's camera is `pixel = K·H(τ)·d(π(X))` with
`H(τ) = P_z(R)·R(τ)`, `R = Ry(τy)·Rx(τx)` (OpenCV's convention,
`vision_calibration_core::models::sensor`). Multiplying out,

```text
K·H(τ) = K'(τ)·R(τ),   K' = [ fx·R22   0        cx − fx·R02 ]
                            [ 0        fy·R22   cy − fy·R12 ]
                            [ 0        0        1           ]
```

with `R02 = −sin τy cos τx`, `R12 = sin τx`, `R22 = cos τx cos τy`
(**derived**; **measured** to 1e-12 px over the image for tilts up to
(20°, 20°), and pinned in core). `K'` is a zero-skew pinhole whose
principal point is the foot of the perpendicular from the projection centre
to the sensor; `(cx, cy)`, the library's principal point, is where the lens
axis meets the sensor, which is also the distortion centre.

So the camera is a pinhole `K'` looking through a lens whose frame is
rotated by `R(τ)`, and the distortion is applied in the lens frame. With no
distortion `R` commutes into every view pose: a tilt changes nothing that a
common rotation of all poses and a change of `K` (focal lengths ×`R22`,
principal point by `f·sin τ`) do not undo. **The tilt is unobservable
without distortion, from any view geometry**: the Fisher information is
singular (F1, first row), and in the gauge-aligned coordinates the tilt's
Jacobian column is zero (test `without_distortion_the_tilt_is_unobservable`).

## First order: the tilt is the offset of the distortion centre

In the sensor frame the lens distortion is centred at
`a = ((cx − cx')/fx', (cy − cy')/fy') = (−tan τy, tan τx)` (normalized), and
taken in a plane rotated by `R`. Expanding a radial distortion decentred by
`a`, `a + (x − a)(1 + k1|x − a|²)`, to first order in `a`:

```text
x + k1·x·r² − k1·(a·r² + 2x(x·a))
```

The last term is exactly Brown's tangential term with `p1 = −k1·a_y`,
`p2 = −k1·a_x` (Conrady's decentring), so (**derived**)

```text
p1 ≈ −k1·tan τx,   p2 ≈ k1·tan τy.
```

What is left after a tangential term absorbs that: the rotation of the
distortion plane, `k1·r²·x(x·a)` (order `k1·τ·r⁴`), the decentring of `k2`
(order `k2·τ·r⁴`), and `O(τ²)`. Two more identities of the same order: a
rotation of the camera adds `ω + x(x·ω)` to the normalized point, and
`(r², 0) = (r² + 2x², 2xy) − 2(x², xy)`, so the thin-prism `r²` terms are
the tangential ones up to a pose rotation and a principal-point shift:
`s1 ≡ p2`, `s3 ≡ p1` to the same order.

Verification (**measured**, analytic part): noise-free tilted data fitted
with the tilt held at zero. With `p1, p2` held, the tilt's signal is left
in the residual; freed, they absorb it, with the predicted value.

| HFOV | corner distortion | tilt | RMS, `p` held | RMS, `p` free | RMS, `s` free | `p1` fitted / predicted |
|---|---|---|---:|---:|---:|---:|
| 30° | −0.02 | 1° | 0.062 px | 0.0014 px | 0.0034 px | 1.01 |
| 30° | −0.02 | 5° | 0.23 | 0.0049 | 0.0092 | 1.00 |
| 30° | −0.10 | 5° | 0.45 | 0.049 | 0.16 | 1.04 |
| 60° | −0.10 | 5° | 0.98 | 0.11 | 0.26 | 1.08 |
| 90° | −0.10 | 5° | 0.59 | 0.23 | 0.33 | 1.34 |
| 90° | −0.10 | 20° | 1.5 | 0.82 | 0.97 | 1.14 |

The tangential term takes 90–98 % of the tilt's signal at a 30° field and
60 % at 90°, where the `r⁴` remainder grows; the prediction holds to a few
per cent on a moderate field. (The run prints 24 cells, both-axis tilts
included; `p2` follows `k1·tan τy` likewise.)

### What breaks which degeneracy

| combination | kind | broken by |
|---|---|---|
| tilt ↔ principal point, focal lengths, a common pose rotation | exact, all orders | nothing in the data: only lens distortion, which the tilt moves relative to the sensor |
| tilt ↔ `p1, p2` | first order in `τ` and in field angle | the `k1·τ·r⁴`, `k2·τ·r⁴` remainder: strong distortion seen at a wide field angle; not depth or board tilt (F3) |
| tilt ↔ `s1..s4` | first order | as above |
| `p1, p2` ↔ `s1..s4` | first order, needs only free poses and principal point | as above; `corr(p1, s3) = −0.995 … −1.000` everywhere in F6 |
| tilt ↔ focal length | second order (`fx' = fx·cos τ`) | grows with the tilt: `corr(fx, τx)` 0.12 at 5°, 0.68 at 20° (F2) |
| tilt ↔ `k1` | none to first order | `|corr| ≤ 0.14` everywhere |

### The mechanism of σ(tilt)

To first order the tilt is `(c − c')/f`: the offset between the distortion
centre `c` (determined by the distortion) and the projective principal point
`c'` (determined by perspective, as for any pinhole). It is lost when either
is: weak distortion (σ ∝ 1/(f·|k1|·r²), every field of view), or a narrow
field (the pinhole's own principal point is weak, σ(τ) floors at σ(c')/f;
the 10° column of F1). The sign of `corr(cy, τx)` tells which: near +1 when
`c'` is the well-determined one (weak distortion), negative when `c` is
(narrow field). This is why one cell can show `corr(cy, τx) = 0.995` and
another, with 16 good views on a 35° lens, 0.005–0.115.

## The map (Fisher information at the truth)

Noise 0.2 px per coordinate (σ scales linearly), poses eliminated, default
free set `fx, fy, cx, cy, k1, k2, τx, τy`. Baseline: 1280×960, HFOV 50°
(f = 1372 px), corner distortion `k1·r_c² = −0.05` (`k1 = −0.147`), tilt
(5°, 0), 20 views, board tilt ±30°, depth 1.5:1, full coverage, roll ±15°,
12×9 corners (1936 points). The Fisher σ equals the library's stated σ to
two digits in every Monte-Carlo cell (table M), so these are the library's
numbers.

**F1 — σ(τx) in degrees: field of view × corner distortion** (`unobs.`: σ
above 5°, singular for no distortion)

| corner distortion | 10° | 20° | 35° | 50° | 70° | 90° |
|---|---:|---:|---:|---:|---:|---:|
| 0 | unobs. | unobs. | unobs. | unobs. | unobs. | unobs. |
| −0.005 | 0.070 | 0.19 | 0.39 | 0.57 | 0.83 | 1.1 |
| −0.02 | 0.13 | 0.034 | 0.067 | 0.12 | 0.19 | 0.26 |
| −0.05 | 0.14 | 0.056 | 0.018 | 0.030 | 0.057 | 0.081 |
| −0.15 | 0.14 | 0.065 | 0.031 | 0.016 | 0.0079 | 0.0062 |

`corr(cy, τx)` on the same grid is 0.98–1.000 wherever σ(τx) > 0.1°, and
negative (−0.8 … −0.87) in the narrow-field cells. (The 10° column needs
`k1` up to −12 to reach those corner distortions; read it as the
narrow-field limit, not as a lens.)

**F1′ — the same with `p1, p2` free** (σ(τx) in degrees; `corr(p1, τx)`
is −0.90 … −1.000 in every cell)

| corner distortion | 10° | 20° | 35° | 50° | 70° | 90° |
|---|---:|---:|---:|---:|---:|---:|
| −0.005 | unobs. | unobs. | unobs. | unobs. | 3.6 | 2.7 |
| −0.02 | unobs. | 3.3 | 1.8 | 1.2 | 0.82 | 0.61 |
| −0.05 | 2.3 | 1.1 | 0.64 | 0.43 | 0.28 | 0.19 |
| −0.15 | 0.48 | 0.23 | 0.13 | 0.082 | 0.052 | 0.021 |

**F2 — tilt magnitude** (baseline): σ(τx) = 0.028–0.033° and σ(cy) =
1.1–1.4 px for every tilt from 0 to (20°, 20°): the size of the tilt does
not matter, only the distortion. What grows is the focal coupling,
`corr(fx, τx)` 0.12 (5°) → 0.34 (10°) → 0.68 (20°), `corr(fy, τx)` up to
0.81, and with it the σ inflation of τx by its correlations (3.0 → 4.75)
and the scaled condition number of the camera block (245 → 591 on one axis,
992 at (20°, 20°); in the gauge-aligned coordinates 190–258 throughout).

**F3 — view design, one factor at a time** (baseline camera; last column
the same views on a weak lens, corner distortion −0.01)

| factor | level | σ(τx) ° | σ(fx) px | σ(cy) px | `corr(cy, τx)` | σ(τx) ° with `p` free | σ(τx) °, weak lens |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | | 0.030 | 0.68 | 1.3 | 0.91 | 0.43 | 0.27 |
| views | 5 / 10 / 40 | 0.065 / 0.048 / 0.022 | 1.7 / 1.1 / 0.42 | 3.0 / 2.1 / 0.92 | 0.86–0.92 | 1.0 / 0.61 / 0.32 | 0.61 / 0.43 / 0.18 |
| board tilt | ±3° / ±10° / ±45° | 0.025 / 0.025 / 0.035 | 2.0 / 1.5 / 0.51 | 1.5 / 1.3 / 1.5 | 0.84–0.93 | 0.43–0.44 | 0.26–0.30 |
| depth ratio | 1 / 2 / 3 | 0.029 / 0.030 / 0.031 | 0.65–0.71 | 1.3 | 0.90–0.92 | 0.42–0.46 | 0.26–0.27 |
| coverage | central 50 % | 0.072 | 1.3 | 3.0 | 0.94 | 1.9 | 0.58 |
| coverage | left half | 0.11 (τy 0.20) | 2.3 | 5.0 | 0.93 | 1.0 | 1.0 |
| roll | 0 / ±90° / ±180° | 0.030–0.034 | 0.66–0.68 | 1.3–1.6 | 0.92–0.93 | 0.41–0.45 | 0.27–0.31 |

Depth range and roll do nothing for the tilt; board tilt does nothing for it
either (it is needed for the focal length, as in any planar calibration:
σ(fx) 2.0 px near-frontal against 0.51 at ±45°). What the tilt needs is
coverage of the image periphery, where the distortion is: half the image
area costs 2.4×, one side 3.7–7×.

**F4 — image size at a fixed field of view**: σ(τ) ∝ 1/f (0.054° at
720×540, 0.030° at 1280×960, 0.017° at 1920×1080) at a fixed pixel noise,
while σ(c) stays 1.2–1.6 px: a tilt is a principal-point offset over `f`.

**F6 — free sets** (σ(τx) in degrees; inflation = σ over the σ it would
have if the other camera coordinates were known)

| cell | default | + `k3` | + `p1, p2` | + `s1..s4` | + `p`, `s` | `corr(τx, p1)` |
|---|---:|---:|---:|---:|---:|---:|
| baseline (50°, −0.05) | 0.030 (×3.0) | 0.030 | 0.43 (×43) | 0.20 (×20) | 29 (×2900) | −0.997 |
| 90°, −0.15 | 0.0062 (×1.3) | 0.0071 | 0.021 (×4.4) | 0.010 (×2.2) | 0.21 (×45) | −0.953 |
| 20°, −0.02 | 0.034 (×1.3) | 0.034 | 3.3 (×126) | 0.22 (×8.6) | 80 | −1.000 |

`p1, p2` themselves are well determined when the tilt is held (σ(p1) =
8e-5 at baseline) and lost when it is free (σ(p1) = 1.2e-3): the pair is the
problem, not either term.

## Stated σ against Monte-Carlo (table M)

100 noise draws per cell through `optimize_planar_intrinsics`, plain least
squares, started at the truth; ratio = scatter of the estimates / mean
stated σ over `fx, cx, cy, k1, τx, τy` (and `p1` when free). Factrs agrees
with tiny-solver to 0.016 in every ratio; "tight" runs to a 1e-14 relative
decrease (**measured**).

| cell | ratio, default stop | ratio, tight | notes |
|---|---|---|---|
| baseline | 0.89–1.15 | 0.89–1.15 | |
| weak distortion (−0.005) | **0.69–0.82** for fx, cx, cy, τx, τy (k1 1.02) | 1.03–1.15 | the default stops along the valley and keeps memory of its start |
| HFOV 20° | 0.84–1.03 | 0.85–1.03 | |
| tilt (20°, 0) | 0.84–1.02 | 0.85–1.02 | |
| 5 views | 0.93–1.07 | 0.93–1.07 | |
| board tilt ±3° | 0.87–1.06 | 0.87–1.06 | |
| left half only | 0.96–1.03 | 0.96–1.03 | |
| noise 0.5 px | 0.89–1.15 | 0.89–1.15 | |
| `p1, p2` free | **0.76–0.89** | 0.89–1.01 | σ(τx) 0.42°, 14× the default's |
| `p1, p2` free, 90°, −0.15 | 0.72–1.03 | 0.72–1.03 | stated σ of τx, `p1`, `cy` 25–40 % above the scatter (hypothesis: linearization over a 90° field) |
| true `p1 = 5e-4`, `p` held | 0.89–1.15 | 0.89–1.15 | **bias τx +0.19° (6σ), cy +4.5 px (3.4σ), fx +0.54 px**; predicted 0.195°, 4.7 px |
| true `p1 = 5e-4`, `p` free | 0.79–0.89 | 0.89–1.02 | no significant bias; σ(τx) 0.42° |

No result in any cell lacked a covariance. Where the tilt is determined, the
stated σ is honest on both backends; the one systematic failure is the
stopping rule in the long valley, a solver issue, not a covariance one.

The held-`p` cell is the price of the default fix list: a real decentring is
reported as tilt and principal point, with a σ that does not cover it. It is
not a reprojection problem (the decentring is absorbed to first order); it
matters to a user who reads the tilt or principal point as a physical
quantity.

## Parameterizations (F5 and table B)

Four coordinate choices for the same model, each solved by the library's
Levenberg–Marquardt loop as a user residual (`solve.rs`; the raw one as a
user residual has the library factor's outcomes and iteration counts, median
and range, in every cell and seed level, so the comparison is the
solver's):

- **P0** raw: `fx, fy, cx, cy` (the distortion centre), lens-frame poses (current);
- **P1** `c̃ = c − f·tan τ` per axis (the substitution tried on a real rig);
- **P2** the sensor-normal pinhole `K'` exactly, lens-frame poses;
- **P3** `K'` with poses in the sensor frame (`P' = R·P`): without
  distortion the tilt then changes nothing, its column is the distortion
  signal alone.

Tilt as the homography's entries, `(h20, h21) = (sin τy, −cos τy sin τx)`,
is a smooth bijection of the tilt alone near zero: it rescales the tilt's
row and column of the covariance by a near-identity Jacobian and changes no
correlation with the other coordinates. It relabels, so it is not in the
table (**derived**).

**F5 (Fisher, measured)**

| cell | | P0 | P1 | P2 | P3 |
|---|---|---:|---:|---:|---:|
| baseline | cond camera / full | 281 / 9.5e5 | 246 / 7.5e5 | 243 / 7.5e5 | 243 / 2.8e5 |
| | max \|ρ\| of τx (with) | 0.91 (cy) | 0.65 | 0.66 | 0.66 |
| weak distortion −0.005 | cond camera / full | 21 000 / 1.1e8 | 935 / 1.1e8 | 232 / 1.2e8 | 232 / 2.9e5 |
| | max \|ρ\| of τx | 1.000 (cy) | 0.88 (fy) | 0.90 | 0.90 |
| tilt 20° | cond camera / full | 591 / 1.0e6 | 333 / 8.0e5 | 198 / 8.1e5 | 198 / 3.0e5 |
| HFOV 20° | cond camera / full | 545 / 2.0e6 | 568 / 1.1e7 | 543 / 1.1e7 | 543 / 1.3e7 |
| | max \|ρ\| of τx | 0.80 (cy) | 0.98 | 0.98 | 0.98 |
| real-rig-like (720×540, 35°, −0.07, −5°, 22 views) | max \|ρ\| of τx | 0.52 (cy) | 0.83 | 0.83 | 0.83 |

σ(τx) is identical in all four (it must be: the marginal σ of a kept
coordinate is invariant under a change of the others). What moves is which
coordinate carries the correlation, and that depends on the regime: with
weak distortion the projective principal point `c'` is the determined one
and P2/P3 are better conditioned (21 000 → 232); on a narrow field or a
strongly distorting lens the distortion centre `c` is, and P0 is better
(the real-rig-like cell reproduces the real rig's 0.005–0.115 against
−0.90 after the substitution). P3 also removes the tilt's coupling to
every pose (full condition number 1.1e8 → 2.9e5 in the weak cell).

**Convergence by parameterization** (direct joint solve from the seeded
route's linear init; reached / 16 and median iterations; table B's starts)

| cell | seed | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|
| baseline | coarse | 16, 8 | 16, 9 | 16, 9 | 16, 8 |
| weak −0.005 | good | 15, 13 (1 stopped in the valley) | 16, 17 | 16, 14 | 16, 7 |
| weak −0.005 | coarse | 15, 20 (1 valley) | 16, 24 | 16, 20 | 16, 14 |
| weak −0.005 | poor | 16, 30 | 16, 35 | 16, 28 | 16, 30 (8–60) |
| (10°, 10°), weak −0.01 | coarse | 16, 16 | 16, 19 | 16, 13 | 16, 8 |
| HFOV 20° | coarse | 16, 9 | 16, 22 | 16, 21 | 16, 16 |
| HFOV 20° | poor | 16, 9 | 16, 35 | 16, 35 | 16, 23 |
| 5 views | poor | 16, 10 | 16, 12 | 16, 12 | 15, 13 (1 wrong-sign tilt) |
| real-rig-like | poor | 16, 8 | 16, 14 | 16, 15 | 16, 12 |
| 5 views | bad | 16, 12 | 16, 18 | 16, 18 | 13, 16 (2 wrong sign, 1 local minimum) |
| HFOV 20° | bad | 16, 10 | 16, 43 | 16, 43 | 16, 29 |

No parameterization reaches a minimum another misses, except P3's four
wrong-basin results on five views (one of 336 starts up to poor seeds,
three of 112 from bad seeds) and P0's two valley stops (the stopping rule,
below). P3 halves the iterations in the weak-distortion valley and doubles
them on a narrow field or a well-determined lens, with long tails (up to 60)
from poor seeds. P1 is never the best. **Decision: keep the raw
parameterization.** A substitution relabels the minimum; the information,
and therefore σ(tilt), is the same, and no coordinate choice is
well-conditioned in every regime.

## Solver (table S)

Direct joint solves from the coarse seed, default stopping rule against a
run to a 1e-14 relative decrease (**measured**):

| cell | iterations default / tight | move to the tight solution, median / max | default reached |
|---|---|---|---:|
| baseline | 8 / 10 | 0.0006σ / 0.002σ | 16/16 |
| weak distortion −0.005 | 20 / 28 | 0.07σ / **0.66σ** | 15/16 |
| (10°, 10°), weak −0.01 | 16 / 21 | 0.013σ / 0.035σ | 16/16 |
| HFOV 20°, tilt 20°, 5 views, real-rig-like | 7–9 / 9–11 | ≤ 0.001σ / 0.006σ | 16/16 each |

The stopping rule is fine wherever the tilt is well determined and stops
short along the valley where it is not (with table M's under-dispersion).
Jacobi scaling, frozen at the first iterate: the tilt columns' norms at the
init differ from those at the truth by at most 1.8× (`cx`, `cy` 1.00, `fx`
1.2×) over every start and cell, so re-scaling per iteration has nothing
to correct here (**measured**; not tried as a solver change).

## Convergence (table B)

Seeds perturbed from the truth (each error uniform in [0.5, 1]× the level's
bound, random sign, both tilt axes, both principal-point axes, `fx = fy`
scaled together): good 5 %, 2°, 20 px; coarse 15 %, 5°, 50 px; poor 30 %,
10°, 100 px; bad 50 %, 15°, 200 px. 16 starts per cell and level, a fresh
noise draw each, σ = 0.2 px. *Reached*: every one of `fx, fy, cx, cy, k1,
τx, τy` within 0.5σ of the reference minimum (a tight solve from the truth
on the same data). *Flagged*: the result itself shows the failure (no
covariance, or RMS > 1.1× the reference's). Cells: baseline; weak
distortion −0.005; HFOV 20°; tilt (20°, 0); tilt (10°, 10°) with −0.01;
5 views; real-rig-like (**measured**, both backends).

| solve | good / coarse / poor seeds | bad seeds | failure modes, bad seeds | flagged |
|---|---|---|---|---|
| seeded route (`step_init_with_seed` + `step_optimize`), tiny-solver and Factrs alike | 336 / 336 | 31 / 112 | on a bound 75, local minimum 4, valley 2 | 76 of 81 |
| direct joint solve from the route's linear init, unbounded (P0) | 334 / 336 (2 valley stops, weak cell) | **112 / 112** | — | 0 of 2 |
| P1, P2 from the same init | 336 / 336 | 112 / 112 | — | — |
| P3 from the same init | 335 / 336 (1 wrong sign) | 109 / 112 | wrong sign 2, local minimum 1 | 3 of 3 |

(Bad-seed counts per cell: route 3–5 of 16 everywhere; the bound hit is
`fx` or `cy`.) Two findings:

- **The route's bounds set its basin.** With up to 30 %, 10°, 100 px of
  seed error every route solve reaches the minimum; past that the route
  fails in two thirds of the starts, almost always on its joint-solve
  bounds (`fx` within [0.75, 1.5]× the seed, principal point ±0.15·f),
  while an unbounded joint solve from the same init reaches the minimum
  every time, in 9–46 iterations. The failures are mostly honest (a bound
  leaves no covariance); 5 of 81 are not (a local minimum with a covariance
  and a near-reference RMS). This matches the real-data basin (focal ×0.5
  and ×2 fail) and is the evidence for CAL-TILT-BOUNDS.
- **Frozen scaling and the clamp are not the limit; the bounds are.** No
  unbounded solve ended in a wrong basin from any seed in the sweep.

**The cold route** (no seed; `step_init` + `step_optimize`, documented as
experimental), 8 draws per cell and backend: reaches the minimum on the
baseline and 5-view cells (8/8), 4/8 on the weak lens (3 valley stops and
1 local minimum, none flagged), and 0/8 on HFOV 20°, tilt 20°, the
(10°, 10°) weak cell and the real-rig-like cell. There it settles at a
tilt of about 4–6.5° in a wrong principal-point basin (`cy` 150–500 px off)
and ends on its `fx` or `cy` bound or its iteration budget (flagged), or,
in 4 of 16 (10°, 10°) solves, in a local minimum with a covariance. The
sweep confirms the seeded route as the supported one; a 20° tilt is
outside the cold sweep's ±10° seeds and ±0.30 rad window by construction.

## Tangential terms and the default fix list

- Freeing `p1, p2` with the tilt multiplies σ(tilt) by 3.4 (90° field,
  corner distortion −0.15) to about 100 (20°, −0.02) over F1, with
  `|corr(τ, p)|` 0.90–1.000; the principal point follows (σ(cy) 1.3 →
  10 px at baseline). Thin-prism terms do the same (×2.2–20), and freeing
  both is unidentified (`corr(p1, s3) = −1.000`, σ(τx) up to 80°).
- They are separable only on a wide field with strong distortion: at 90°
  and −0.15 the tilt keeps σ = 0.021° with `p` free, and the Monte-Carlo
  is honest there (stated σ 25–40 % conservative). That is a fisheye-like
  regime, not a Scheimpflug lens.
- The price of holding them, measured: a real decentring is absorbed as
  `Δτx = −p1/k1`, `Δcy ≈ fy·Δτx` (0.19° and 4.5 px for `p1 = 5e-4`,
  `k1 = −0.15`; the expansion predicts 0.195° and 4.7 px), 6σ and 3.4σ,
  invisible in the covariance. Freeing `p1, p2` removes the bias at 14×
  the σ. Neither choice measures a physical tilt on a decentred lens; the
  held one keeps a usable σ for the camera as a whole.

**Policy check: keep** the default (only `k1, k2` free under a tilt, `p1,
p2`, `k3`, `s1..s4` held). `k3` free costs nothing on σ(tilt) (F6) but is
held for the usual overfitting reason; the evidence does not argue either
way for it.

## Minimum view design for a tilted camera

From F1, F3 and table M, for σ(tilt) ≲ 0.1° at 0.2 px noise:

- lens distortion of at least about 2 % at the image corner on a 35–70°
  field (F1); a lens with less, or a field under about 20°, leaves the tilt
  to the principal point's own σ (hold the tilt at a known mount angle
  there, or accept `corr(c, τ) ≈ 1`);
- corners detected out to the image corners: central-only coverage costs
  2.4×, one-sided 3.7–7× (F3);
- board tilt of ±20–45° for the focal length (σ(fx) 2.0 → 0.51 px), not
  for the tilt; depth range and roll do not matter;
- σ(tilt) falls as 1/√(views) (5 → 40 views: 0.065° → 0.022°).

## Recommendations

| # | recommendation | evidence | status |
|---|---|---|---|
| 1 | Keep the raw parameterization; no substitution | F5 and the parameterization table: σ(tilt) invariant, the better-conditioned coordinates switch with the regime, P1 never best, P3 helps only in the weak-distortion valley and adds wrong-sign failures | decided (no change) |
| 2 | Keep the default fix list | F1′, F6, table M policy cells | decided (no change) |
| 3 | A stopping rule in σ units: stop when the remaining Gauss-Newton step is under 0.01σ in the Fisher metric | table M weak cell (0.69–0.82 → 1.03–1.15), table S (0.66σ short) | done: the default rule (`lm-stopping.md`); default and tight now agree in every cell of tables M and S |
| 4 | Run the seeded route's joint stage unbounded, with the bounds only as a runaway fallback | table B bad seeds: route 31/112, unbounded 112/112 | backlog CAL-TILT-BOUNDS |
| 5 | Name the cause when the quality report flags a tilt pair, with the tilt's σ inflation `1/√(1 − R²)` | F1 (corr 0.98–1.000 where σ > 0.1°), F6 | done: the quality report's `ParameterConditioning` check (`crates/vision-calibration/tests/quality_tilt_conditioning.rs`) |
| 6 | Document the observability, the view design and the decentring bias for users | the sections above | the book's Scheimpflug chapter |
| 7 | Frozen Jacobi scaling and per-iteration rescaling | table S: column norms move ≤ 1.8× | no change |

Not recommended: exporting the sensor-normal pinhole `K'` as the
"well-determined" intrinsics. It is the better-determined set only with
weak distortion; on a narrow field the distortion centre is (F5).

What to do with a tilt the data do not determine — the decision statistic,
the cost of fixing it, priors and the frame of each answer — is
[tilt-determination.md](tilt-determination.md).

## Hypotheses and open items

- **Real-data transfer** (hypothesis): the mechanism (distortion centre vs
  projective principal point) predicts the real rig's low `corr(cy, tilt_x)`
  (35° field, `k1 ≈ −0.43`) and the earlier 0.995 (`k1 = −0.04`, 8 views ≤
  0.45 rad); the synthetic real-rig-like cell reproduces the first. The
  bounds finding (recommendation 4) needs the real-data basin as its gate.
- The 90° `p`-free cell's stated σ is 25–40 % above the scatter; the
  linearization over a wide field is the guess, not measured.
- Not in the sweep: `k2`-dominated lenses (the `k2·τ·r⁴` remainder),
  rational and division models under a tilt, robust losses, the rig and
  hand-eye solves (the rig's shared poses pin the tilt better,
  `scheimpflug-intrinsics.md`), and lenses with true decentring and a true
  tilt together beyond the one `p1` cell.
- A per-iteration-rescaled or projected-bound LM was not implemented; the
  measurements above say neither is where the problem is.

# Is the camera-fixed residual field a camera property, and is it compact?

The Brown-Conrady + tilt model leaves a smooth field on the two private
six-camera Scheimpflug captures, and a B-spline removes much of it
([camera-residual-field.md](camera-residual-field.md),
[ADR 0030](../adrs/0030-camera-residual-field.md)). This note asks whether
that field is a property of the camera at all, whether a few parameters
describe it, and which physical terms could. Dev-facing; aggregate numbers
only.

**Data.** `scheimpflug_rig_ringgrid_cam0..5` (ring-grid markers, photometric
centres, per-marker fitted blur) and `scheimpflug_rig_ref_cam0..5`
(PuzzleBoard corners). Both patterns are shown on **the same e-ink
display**, so the two captures differ in pattern and detector, not in
target. They are **the same six cameras**: each unit's intrinsics agree
between the captures (`fx` within 0.3–13 px, `cx` within 2–8 px; the unit
with `cx ≈ 400` and the one with `cy ≈ 278` are the same index in both).

## Conclusions

1. **A large part of the "camera-fixed" field is coupled to defocus and
   depth, not fixed in the image** (measured). On the ring-grid capture a
   field whose amplitude follows the *geometric* blur (blur² modelled from
   image position and inverse depth, 84–91 % of the blur variance) predicts
   held-out views better than the image-fixed field (0.0986 against 0.1041
   px RMS); with a depth-coupled term it reaches 0.0834 without any image
   field, and 0.0731 with one — the calibration-free smooth-fit floor
   (0.073). The detector's per-marker blur noise is not what carries it
   (the measured-minus-geometric blur gains 12 %, the geometric part 52 %).
   On the corner capture the same unit's defocus model, taken from the
   ring-grid capture, gains 40 % on its own (the image field 35 %). Fitted
   next to the image field, these terms change it by 37–44 % of its energy
   each; together the image field is no longer separately determined.
2. **The display is not the source of the image field** (measured). A
   board-frame term shared by the six cameras — the display's smooth
   in-plane mapping (polynomial to degree 6) or a free offset per board
   point (rasterisation) — moves at most 16 % of the image field. The
   per-point offsets are a real, separate, additive term on the corner
   capture (−29 % alone, −34 % on top of the field), none on the ring-grid
   capture. Pixel phase, the perspective bias of a circle's centre and the
   board's in-plane orientation explain nothing.
3. **No physical lens term from the decentred-surface expansion carries
   it** (derived and measured). The terms a system of tilted or decentred
   rotationally symmetric surfaces adds beyond the library's model (skew,
   the fifth-order `m ±2` and `m ±1` pairs) gain ≤ 8 %; two terms the
   expansion forbids at low order (`w̄²`, `w̄³`) carry half to two thirds of
   the field's gain, are identifiable and consistent across all twelve camera-captures
   — and are mirror-symmetric about the tilt plane, as the defocus pattern
   is. Fitted next to the defocus and depth terms they add 0.5 % on the
   corner capture and 11 % on the ring-grid capture, and `Im w̄²` changes
   sign: they are largely a projection of the coupling.
4. **A compact empirical basis exists per rig** (measured): two principal
   fields with per-camera coefficients give 84 % of the field's held-out
   gain on the ring-grid capture (76 % on the corner capture), and two
   fields learned on five cameras give 76 % on the sixth (ring-grid; 55 %
   on corners). Their coefficients scatter 1.4–3.5 times their σ between
   folds.

**Verdict: (c), with a residue of (b).** Most of the field is a coupling
between defocus/depth and the measured position, so its image-fixed
average is a property of the camera *and the capture's depth distribution*
(and possibly of the detector), not of the camera alone; a camera slot
fitted on one capture would carry a capture-specific error into the next.
What stays fixed in the image after the coupling is modelled is small and
compact but not the same on the two captures (a residual `Re w̄³` on the
ring-grid capture, nothing on the corner capture). What decides between an
optical cause (the centroid of a defocused bundle moving with vignetting)
and a detector cause (a blur-dependent centre bias) is a capture, not this
data (below).

## Method

Bench example `scheimpflug_model_study`, `--parts field` (`linfield.rs`,
`field_study.rs`), `fieldsweep` (`field_synth.rs`):

```text
cargo run --release -p vision-calibration-bench --features tier-b \
    --example scheimpflug_model_study -- \
    --registry crates/vision-calibration-bench/registry/private.json \
    --out <dir> --parts field <ids of one or both captures>
```

- **Linearized about the route.** The route's camera and poses are the
  linearization point; every candidate term is a set of pixel-displacement
  columns (central differences of the model's projection, or the term
  itself), poses eliminated per view. One Gauss-Newton step solves a
  0.1 px term to second order. It reproduces the nonlinear held-out
  numbers of the earlier notes: route 0.1426 / 0.3150 (0.143 / 0.315),
  degree 2–4 polynomial camera held 0.1164 / 0.2784 (0.116 / 0.278),
  B-spline 10×8 orthogonal 0.0997 / 0.2478 (0.100 / 0.248).
- **Held out** exactly as before: 4 folds by view; any basis derived from
  data (selected terms, principal fields) is derived from the training
  folds only, or from other cameras.
- **Always at the projection.** Every image term is evaluated at the
  model's projection; per-point covariates (blur, depth, scale, board
  position) are geometry or detector outputs, never the residual.
- **`φ`**: complex monomials `z^p z̄^q` (degree ≤ 6, about the image
  centre) with the typical camera's eight parameter directions and a common
  rotation removed, orthonormal over the covered image (49 fields). A
  coefficient is the px RMS of its field there.
- **Null and recovery** (`--parts synth,field synth_plain_cam1..6`,
  `synth_zb3_cam1..6`): plain-model scenes with 0.07 px noise and a
  defocus-like blur covariate that does not affect the detections.

## Q1 — is it a camera property?

Held-out pooled RMS / mean px; *Δ* is the change of the held-out variance;
*moves* is how much of the image field's energy changes when the term is
fitted next to it (over each camera's coverage).

| term | ring-grid alone (Δ vs route) | ring-grid with field (Δ vs field) | moves | corners alone | corners with field | moves |
|---|---|---|---:|---|---|---:|
| route | 0.1426 | — | | 0.3150 | — | |
| image field `φ` ≤ 6 | 0.1041 (−47 %) | — | | 0.2543 (−35 %) | — | |
| measured blur × deg 0–3 | 0.1016 (−49 %) | 0.0857 (−32 %) | 35 % | no blur | | |
| geometric blur × deg 0–3 | 0.0986 (−52 %) | 0.0846 (−34 %) | 44 % | 0.2430 (−40 %)¹ | 0.2137 (−29 %) | (collinear) |
| blur residual × deg 0–3 | 0.1337 (−12 %) | 0.0933 (−20 %) | 24 % | | | |
| depth `(z̄/z − 1)` × deg 1–4 | 0.1223 (−26 %) | 0.0856 (−32 %) | 37 % | 0.2859 (−18 %) | 0.2095 (−32 %) | (collinear) |
| — radial part only | 0.1393 (−5 %) | 0.1007 (−7 %) | 1.5 % | 0.3108 (−3 %) | 0.2435 (−8 %) | 10 % |
| — `|m| ≥ 2` only | 0.1301 (−17 %) | 0.0916 (−23 %) | 12 % | 0.3048 (−6 %) | 0.2320 (−17 %) | 26 % |
| geometric blur + depth | 0.0834 (−66 %) | 0.0731 (−51 %) | (collinear) | 0.2262 (−48 %)¹ | 0.2113 (−31 %) | (collinear) |
| board in-plane polynomial ≤ 6, shared | 0.1418 (−1 %) | 0.1048 (+1 %) | 6 % | 0.3012 (−9 %) | 0.2460 (−6 %) | 16 % |
| board per-point offsets, shared | 0.1420 (−1 %) | 0.1039 (−0 %) | 1 % | 0.2660 (−29 %) | 0.2059 (−34 %) | 14 % |
| board orientation `cos/sin kθ`, k ≤ 6 | 0.1726 (+47 %) | 0.1164 (+25 %) | 0 % | 0.4078 (+68 %) | 0.2793 (+21 %) | 1 % |
| pixel phase | 0.1430 (+1 %) | 0.1044 (+1 %) | 0 % | 0.3152 (+0 %) | 0.2544 (+0 %) | 0 % |
| circle-centre perspective bias | 0.1426 (−0 %) | 0.1043 (+0 %) | 0 % | 0.3146 (−0 %) | 0.2545 (+0 %) | 0 % |
| local scale × deg 0–2 | 0.1371 (−8 %) | 0.0983 (−11 %) | 7 % | 0.3084 (−4 %) | 0.2321 (−17 %) | 20 % |

¹ The same unit's geometric blur model fitted on the ring-grid capture,
applied to the corner capture's geometry (same cameras, same focus).
*Collinear*: the field changes by more than its own energy; with the
term present the image field is not separately determined, and a "share"
means nothing. On plain synthetic data no row of this table lowers the
held-out variance (+0.0 to +1.2 %, the orientation term +4.9 %), none goes
below the noise, and the board-frame terms change it by +0.0 to +2.0 %.

**What each alternative's lever is in this capture.**

- *Defocus against depth*: within a 60-px cell the depth spans a factor
  1.8–2.1, and the blur (0.7–3.4 px) is V-shaped in depth about the tilted
  focus surface, so signed depth and unsigned defocus are separable; they
  add (66 % together against 52 % and 26 % alone on the ring-grid capture).
  The depth part that matters is not radial (`|m| ≥ 2`): not the
  entrance-pupil walk of a symmetric lens, which is radial.
- *Board frame against image frame*: a board point moves over the image by
  50 px (corners) to 100 px (ring-grid) RMS between views, but within one
  camera its board position predicts its image `u` to 81–97 % and `v` to
  19–62 %, so per camera the two frames are only partly separable. The six
  cameras see the one display from six directions (rolls about 60° apart);
  a shared board-frame term must explain all six at once, which is what
  separates it here.
- *Orientation*: the board's roll varies by ≤ 15° within a camera, so a
  per-camera orientation term is not identifiable (the held-out rows get
  worse); across cameras the rolls differ by about 60° while the field's
  strongest coefficients keep their sign and phase in the image (Q3), which
  a lattice-tied effect would rotate.
- *Sharp against blurred markers* (ring-grid, halves split by blur relative
  to its typical value at the position, camera free): the two halves'
  fields differ by 0.15–1.5 px (null scenes: 0.03–0.05 px) and correlate
  −0.01 to +0.72 (parity halves +0.81 to +0.96). The ratio statistic of
  the earlier note is not usable: on null scenes it is larger than on the
  data, because the halves differ in geometry as well as in noise.

**Same units across the two captures.** Field correlation over the cells
both cover, camera fields with both captures' camera directions removed:
same unit +0.76 ± 0.07, different units +0.61 ± 0.14. Coefficients per
unit: `Im w̄²` correlates +0.84 between the captures, `Re w̄³` +0.41. Since
both captures share the display, this separates the unit from the
detector, not from the display; the board-frame rows above separate the
display.

## Q2 — a compact description

Each camera's field in `φ` (full data). Field RMS over the camera's own
coverage: ring-grid 0.11–0.38 px, corners 0.20–0.34 px; noise in it 2–16 %.
Strongest `φ` of the rig mean: ring-grid `Im z̄²` (m −3) 35 %, `Im zz̄`
(m −1) 21 %, `Re z²z̄` 13 %, `Re z³` 10 %; corners `Re z̄³` (m −4) 25 %,
`Im zz̄` 18 %, `Im z̄²` 13 %, `Re z³` 8 %. Pair correlations over common
cells: ring-grid +0.73 ± 0.12, corners +0.53 ± 0.19 (null scenes
−0.01 ± 0.18, a truly shared synthetic term +0.97); the same camera's
training folds against each other +0.95 to +0.99. Principal fields over the
cells every camera covers: ring-grid 82 / 11 / 4 %, corners 64 / 18 / 7 %.
Mirror symmetry about the vertical through the principal point (the
plane of the tilt), share of each field's energy that is symmetric:
ring-grid 0.75–0.96, corners 0.61–0.97, null scenes 0.44–0.63 (0.5 for an
unstructured field). A board-frame effect seen at six rolls has no reason
to keep this symmetry; anything the tilt geometry makes, including the
defocus pattern, does.

Held-out, bases from training folds (Δ variance vs route; the field `φ` ≤ 6
per camera: ring-grid −46.7 %, corners −34.8 %; B-spline 10×8 −51.2 %,
−38.1 %):

| basis | k | ring-grid per camera | ring-grid shared | corners per camera | corners shared |
|---|---:|---:|---:|---:|---:|
| strongest `φ` of the training rig mean | 1 | −12.3 % | −10.6 % | −10.9 % | −9.4 % |
| | 2 | −26.9 % | −23.8 % | −15.3 % | −12.7 % |
| | 4 | −33.7 % | −29.8 % | −18.5 % | −15.7 % |
| | 6 | −35.6 % | −31.5 % | −21.4 % | −18.0 % |
| | 10 | −34.7 % | −32.9 % | −24.4 % | −19.7 % |
| principal fields of the training fields | 1 | −27.8 % | −23.2 % | −23.1 % | −20.8 % |
| | 2 | −39.4 % | −35.1 % | −26.3 % | −21.1 % |
| | 4 | −43.7 % | −36.6 % | −32.6 % | −21.4 % |
| other five cameras' principal fields | 1 | −22.3 % | | −19.1 % | |
| | 2 | −35.3 % | | −19.1 % | |
| | 4 | −38.4 % | | −19.8 % | |
| other five cameras' mean field, held | — | | −5.6 % | | −15.5 % |

(Six cameras give at most six principal fields, so `k = 6, 10` exist only
for the `φ` selection.) Per-camera coefficients of a fixed 2–4 field basis:
the camera-to-camera spread of a leading coefficient is 27–66 % of its mean, the fold jackknife
scatter 1.4–3.5 times the stated σ (the route itself shows 1.2–2.3 on real
views), correlation with any camera parameter ≤ 0.57.

## Q3 — physical terms

**Derivation.** A system of rotationally symmetric surfaces, surface `j`
decentred by a small complex `σ_j` (a tilted surface is a decentred centre
of curvature), produces image displacements `w^p w̄^q σ^a σ̄^b`, `w` the
undistorted normalized point about the axis, that turn with the image:
`p − q + a − b = 1`. With `m = p − q − 1` the rotation order, `m = b − a`.
Expanding the third-order distortion `|w − s|²(w − s)` and the fifth-order
`|w − s|⁴(w − s)` (summing over surfaces):

| order in `σ` | `m` | terms | in the library's model? |
|---|---|---|---|
| 0 | 0 | `w²w̄`, `w³w̄²` (real) | `k1`, `k2` |
| 1 | ±1 | constant; `Āw² + 2Aww̄`; `−(3Aw²w̄² + 2Āw³w̄)` | `cx, cy`; tangential / thin prism / tilt keystone; `s2, s4` partly, `w³w̄` not |
| 2 | ±2, 0 | `B w̄` (anamorphism); `B̄w³ + 3Bww̄²` | `fx ≠ fy` only, not its 45° part (skew); not |
| 3 | ±3 | `w̄²`, `ww̄³` | not |
| 4 | −4 | `w̄³` needs seventh-order distortion | not |

So the terms a decentred or tilted lens group adds at plausible order are
the 45° anamorphism (skew), the linked fifth-order pair `B̄w³ + 3Bww̄²`
(`NAT B5`) and the linked degree-4 pair (`NAT A5`); `w̄²` and `w̄³` need the
decentrations to third and fourth order. Lens tilt as distinct from sensor
tilt is the same model (the camera frame follows the lens axis); the
entrance-pupil walk of a symmetric lens is the radial depth term of Q1.
The linked forms are pinned by a unit test against the expansion
(`model.rs`).

**Measured**, each term alone with the camera free, held-out (Δ variance
vs route; share of the image field's gain):

| term | cols | ring-grid | corners | coefficient across cameras (ring-grid / corners) | jackknife / σ | largest corr |
|---|---:|---|---|---|---|---|
| skew `Im w̄` | 1 | −1.5 % (3 %) | −0.6 % (2 %) | +1.9e-4 ± 4.1e-4 / +3.4e-4 ± 3.1e-4 | 1.1–1.4 | `cx` −0.72 |
| `NAT B5` | 2 | −1.6 % (3 %) | −1.7 % (5 %) | not consistent | 0.8–1.5 | `cy` +0.67 |
| `w³ + ww̄²` free | 4 | −3.1 % (7 %) | −3.0 % (9 %) | not consistent | 0.7–1.8 | `tilt_x` +0.68 |
| `NAT A5` | 2 | −8.1 % (17 %) | −7.4 % (21 %) | `Im` −2.5e-2 / −3.1e-2 | 0.6–1.2 | `cy` +0.88 |
| radial twist `Im w²w̄` | 1 | −0.5 % | −0.6 % | — | 0.8–0.9 | |
| `w̄²` (m −3) | 2 | −20.8 % (45 %) | −10.1 % (29 %) | `Im` +3.4e-3 ± 1.6e-3 / +3.7e-3 ± 2.1e-3 | 1.4–2.0 | `cy` −0.82 |
| `w̄³` (m −4) | 2 | −22.8 % (49 %) | −11.8 % (34 %) | `Re` −8.7e-3 ± 2.1e-3 / −9.2e-3 ± 2.7e-3 | 1.0–1.7 | `cy` +0.71 |
| `w̄² + w̄³` | 4 | −30.9 % (66 %) | −17.4 % (50 %) | `Im w̄²` +2.3e-3, `Re w̄³` −6.2e-3 / +2.8e-3, −7.5e-3 | 0.8–2.2 | `cy` −0.72 |
| all lens terms of degree ≤ 4 outside the model | 18 | −40.6 % (87 %) | −26.9 % (77 %) | | 1.0–2.7 | `cx` −0.92 |

**Next to the coupling.** `w̄² + w̄³` fitted with the geometric-blur and
depth terms of Q1: corners 0.2261 → 0.2256 (−0.5 %), coefficients no longer
consistent across cameras (`Im w̄²` −3.6e-3 ± 8.3e-3, `Re w̄³`
−1.2e-2 ± 2.2e-2); ring-grid 0.0833 → 0.0787 (−10.7 %), `Re w̄³`
−3.9e-3 ± 1.4e-3 (alone −6.2e-3), `Im w̄²` −2.2e-3 ± 2.0e-3 (alone
+2.3e-3). The residue that survives on one capture and not on the other is
not a camera term.

**Synthetic.** Identifiability over sensor tilt 0–20°, focal length 600 /
1150 / 2300 px (720×540) and the board over the whole image or its lower
two thirds (20 noise-free views, linearized at the truth; largest
correlation with a camera parameter, σ inflation from freeing the camera):

| term | max \|corr\| over the grid | σ inflation | verdict |
|---|---|---|---|
| skew | 0.13–0.96, grows with tilt (`cx`) | 1.0–6.0 | not identifiable under a tilt |
| `w̄²` | 0.18–0.70 | 1.1–1.8 | identifiable |
| `w̄³` | 0.17–0.59 | 1.1–1.3 | identifiable |
| `w̄² + w̄³` | 0.23–0.60 | 1.0–1.7 | identifiable |
| `NAT B5` | 0.14–0.62 | 1.4–2.1 | identifiable, weakly |
| `NAT A5` | 0.60–0.91 (`tilt`) | 1.7–4.4 | not identifiable |

Recovery (six scenes, `w̄³` truth `(2.5, −1.5)·10⁻³`, 0.07 px noise): 
`(2.48 ± 0.06, −1.53 ± 0.08)·10⁻³`, held-out at the noise (0.0988), `w̄²`
stays at zero. Null (plain scenes): every lens term within 1σ of zero,
held-out unchanged (≥ 0.0988), jackknife / σ 0.7–1.4.

**Acceptance** (held-out gain on both captures, stable over folds, similar
across cameras, synthetic recovery, null, identifiable over the sweep):
`w̄²` and `w̄³` pass every numerical criterion; no term passes the physical
one — they are not produced at plausible order by any decentred or tilted
rotationally symmetric optics, and the NAT terms that are produced gain
nothing. Their image-axis phases (`Im w̄²`, `Re w̄³`, and `Im zz̄` among
the field's strongest) are what the mirror symmetry of a camera tilted
about `x` allows, which the defocus pattern of a Scheimpflug camera shares.

## What is undetermined, and what would settle it

- **Optics or detector for the defocus coupling.** The same poses captured
  at two apertures: a chief-ray (distortion) term does not change with the
  aperture, a defocus-coupled term scales with the blur. Then the same
  rendered markers blurred with the measured, not Gaussian, defocus PSF
  through the detector, for the detector's share.
- **Signed defocus.** Tested in [tilt-from-focus.md](tilt-from-focus.md)
  with the fitted plane of sharp focus: at a fixed image position signed
  defocus is affine in inverse depth, so next to an image field it is the
  depth term (held-out 0.0868 against 0.0867); the unsigned blur is the
  separate term. Views through focus at one image position cannot
  separate a vignetting centroid shift from a depth (pupil) term; the same
  poses at two focus settings or two apertures can.
- **The display.** The same rig against a printed board removes the
  display's per-point offsets and any aliasing (sharp markers: ~0.05 px
  extra scatter, observed by the maintainer); the board-frame rows say the
  image field does not come from it, but only for the display's static
  part.
- **A different lens type** to test whether `w̄²`, `w̄³` follow the lens
  design or this tilt geometry.

Until then, the B-spline slot of ADR 0030 would model a capture-dependent
average, not a camera property; see the ADR.

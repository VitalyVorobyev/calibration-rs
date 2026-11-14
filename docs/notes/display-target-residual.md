# A target shown on a display: what limits the residual

The private entries `scheimpflug_rig_ref_cam0..5` calibrate a six-camera
Scheimpflug rig against a PuzzleBoard shown on an e-ink display rather than
printed. A display target has a pixel grid, a cover layer above the ink and
possibly a non-flat stack. This note measures which of those, and which
camera term, remove the residual, on held-out captures. Dev-facing;
aggregate numbers only. The camera-side background is
[scheimpflug-residual-model.md](scheimpflug-residual-model.md).

## Data

Six cameras of one lens type in a ring, 720×540 tiles, `f ≈ 1150 px`,
`tilt_x ≈ −0.09 rad`, `k1 ≈ −0.41`, about 6.1 px/mm on the board. 20
synchronized captures, 15–20 kept per camera, 18 575 corners on 623 board
points (5 mm cells, 160 × 125 mm). The ray meets the board at 11° / 23° /
38° (10th percentile / median / 90th percentile) on every camera, and the
in-plane view direction is spread around the ring: each camera's median
azimuth differs from its neighbour's by roughly 60°, so a board-fixed
displacement reaches each camera along a different image direction.

The calibration-free yardstick (`calib-bench diagnose floor`, a quartic
board-to-image fit per view): smooth-fit **0.213 px RMS**, calibrated 0.309,
gap 0.223 px; 20 µm of per-corner offset repeats across views and
cameras (38 % of the smooth-fit variance).

## Method

The bench example `scheimpflug_model_study` with `--parts target`
(`target.rs`, `target_study.rs`):

```text
cargo run --release -p vision-calibration-bench --features tier-b \
    --example scheimpflug_model_study -- \
    --registry crates/vision-calibration-bench/registry/private.json \
    --out <dir> --parts target [--models <name>,...] \
    scheimpflug_rig_ref_cam0 ... scheimpflug_rig_ref_cam5
```

**One joint fit over the six cameras.** Each camera keeps its intrinsics,
`k1 k2`, tilt and its own pose per view (no rig constraint); the target terms
are shared. A shared target term therefore has to explain six cameras that
see the board from six directions, which is what tells a target property from
a camera property. The baseline configuration reproduces the seeded route
exactly (it stops after one iteration at the same cost).

**Held-out captures.** Four folds by capture (the image file), so a held-out
capture is held out of every camera at once. Each fold refits the route per
camera and the joint model on the training captures, then fits only the
held-out views' poses with cameras and target held. Every target term,
including the per-point offsets, is estimated on training captures only.

**The terms.**

- *In-plane offsets* (free target points): each board point moved by
  `(dx, dy)` in the board plane, per point or separable (`dx` per column,
  `dy` per row). Gauge: a planar similarity, removed by holding two points
  (or two columns and two rows). A point needs 8 training observations to
  be free.
- *H1, shape*: `w(x, y)`, degree 2..=2 or 2..=4 in normalized board
  coordinates, shared or per camera.
- *H2, cover layer*: the ink seen through a plane-parallel layer of
  thickness `t`, `n = 1.5`. With `d = P − C` in the board frame, the ray
  appears at the board plane at
  `P + t·d_∥·(1/|d_z| − 1/sqrt(n²|d|² − |d_∥|²))`, outward along the
  in-plane view direction. Its paraxial part `t(1 − 1/n)` is a shift of the
  plane and goes into the pose; the rest grows as `tan³θ`.
- *H4-like, generic view-angle shift*: `a₃·tan²θ·d_∥/|d_z|`, the cubic term
  of any shift that depends on the view angle alone. Refraction predicts
  `a₃ = t(n² − 1)/(2n³) ≈ 0.185 t > 0`.
- *H3, image field*: a polynomial displacement in normalized pixel
  coordinates per camera (degrees 2..=d, 24 terms per camera at d = 4), with
  the camera parameters free or held at the route's values, or one field
  shared by the six cameras.

## Results

Held-out captures, all 18 575 corners; RMS / mean / median in px, and the
mean per camera.

| model | train RMS / mean | held-out RMS / mean / median | cam0 | cam1 | cam2 | cam3 | cam4 | cam5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline (route) | 0.314 / 0.267 | 0.315 / 0.268 / 0.238 | 0.311 | 0.230 | 0.236 | 0.261 | 0.303 | 0.274 |
| + offsets per point | 0.260 / 0.221 | 0.269 / 0.228 / 0.202 | 0.239 | 0.206 | 0.205 | 0.230 | 0.254 | 0.237 |
| + offsets separable | 0.281 / 0.239 | 0.284 / 0.241 / 0.214 | 0.254 | 0.210 | 0.223 | 0.238 | 0.270 | 0.254 |
| + shape, degree 2 (H1) | 0.313 / 0.266 | 0.314 / 0.267 / 0.237 | 0.309 | 0.227 | 0.237 | 0.259 | 0.301 | 0.273 |
| + shape, degree 4 (H1) | 0.312 / 0.265 | 0.313 / 0.267 / 0.237 | 0.308 | 0.228 | 0.235 | 0.261 | 0.300 | 0.272 |
| + shape 4 per camera | 0.288 / 0.244 | 0.294 / 0.249 / 0.220 | 0.295 | 0.222 | 0.219 | 0.238 | 0.270 | 0.251 |
| + cover (H2) | 0.314 / 0.267 | 0.315 / 0.268 / 0.238 | 0.311 | 0.230 | 0.236 | 0.261 | 0.303 | 0.273 |
| + angle shift `a₃` | 0.313 / 0.267 | 0.315 / 0.268 / 0.238 | 0.311 | 0.230 | 0.236 | 0.260 | 0.303 | 0.273 |
| + shape 4 + cover | 0.311 / 0.265 | 0.313 / 0.266 / 0.237 | 0.308 | 0.227 | 0.235 | 0.261 | 0.300 | 0.272 |
| + field 4 (H3) | 0.180 / 0.148 | 0.186 / 0.152 / 0.128 | 0.169 | 0.137 | 0.137 | 0.141 | 0.168 | 0.162 |
| + field 4, camera held | 0.274 / 0.233 | 0.280 / 0.237 / 0.208 | 0.279 | 0.208 | 0.205 | 0.232 | 0.259 | 0.244 |
| + field 4 shared by the cameras | 0.237 / 0.197 | 0.242 / 0.200 / 0.172 | 0.205 | 0.180 | 0.166 | 0.169 | 0.203 | 0.273 |
| + offsets + field 3 | 0.152 / 0.125 | 0.161 / 0.132 / 0.112 | 0.138 | 0.118 | 0.124 | 0.125 | 0.148 | 0.142 |
| **+ offsets + field 4** | 0.144 / 0.117 | **0.153 / 0.124 / 0.103** | 0.128 | 0.116 | 0.119 | 0.123 | 0.134 | 0.127 |
| + offsets + field 6 | 0.133 / 0.107 | 0.145 / 0.116 / 0.095 | 0.119 | 0.108 | 0.112 | 0.115 | 0.123 | 0.117 |
| + offsets separable + field 4 | 0.155 / 0.126 | 0.159 / 0.130 / 0.108 | 0.136 | 0.118 | 0.125 | 0.122 | 0.140 | 0.136 |
| + offsets + field 4, camera held | 0.227 / 0.191 | 0.240 / 0.201 / 0.174 | 0.217 | 0.179 | 0.184 | 0.206 | 0.218 | 0.206 |
| + offsets + field 4 + cover | 0.144 / 0.117 | 0.153 / 0.124 / 0.104 | 0.127 | 0.117 | 0.120 | 0.123 | 0.134 | 0.127 |
| + offsets + field 4 + shape 4 | 0.142 / 0.116 | 0.151 / 0.123 / 0.102 | 0.128 | 0.114 | 0.119 | 0.122 | 0.132 | 0.124 |
| + offsets + field 4 + shape 4 + cover | 0.141 / 0.115 | 0.150 / 0.122 / 0.101 | 0.126 | 0.114 | 0.119 | 0.120 | 0.130 | 0.123 |

Fitted values, full data, with the four training folds in brackets:

| term | value | across folds / cameras |
|---|---|---|
| offsets per point, RMS after the similarity gauge | 33.9 µm (35.5 with the field) | folds 30.5–32.6 µm |
| of which: smooth (quartic in board position) | 18 µm, at most 48 µm | |
| of which: separable rest (`dx(col)`, `dy(row)`) | 18 µm | |
| of which: non-separable rest | 22 µm | |
| shape, degree 2, peak-to-valley after a plane | 170 µm | folds 163–181 µm |
| shape, degree 4 | 224 µm (346 with offsets and field) | folds 208–264 µm |
| shape 4 per camera | 3.0–7.6 mm | folds 2.4–12 mm: absorbs camera error |
| cover `t`, alone | 1.08 ± 0.21 mm | folds 0.65–1.28; per camera −1.1 to +2.5 |
| cover `t`, with shape 4 | −1.88 ± 0.25 mm | folds −1.6 to −2.3 |
| cover `t`, with offsets and field 4 | −1.27 ± 0.17 mm | per camera −2.6 to +0.5 |
| angle shift `a₃`, alone | +0.179 ± 0.022 mm | folds 0.176–0.186 |
| angle shift `a₃`, with offsets and field 4 | −0.080 ± 0.018 mm | |

## Reading

**The target terms that matter are in-plane.** Per-point offsets fitted on
training captures lower every camera's held-out error (0.315 → 0.269 px
alone, 0.186 → 0.153 on top of the field), and they take the repeatable
per-point part of the held-out residual from 24.7 µm to zero. They are a
property of the board: in the board frame, the per-point means of the six
cameras' baseline residuals correlate +0.21 on held-out captures (+0.24 in
`diagnose floor`), positive on 14 of 15 pairs, and with the offsets fitted
that drops to −0.06. (Measured in image pixels the same correlation is
−0.05: cameras that see the board turned see one offset along different
image directions. `diagnose floor` took it in pixels until this study,
which made the shared target term look absent.) The offsets are half a
smooth in-plane deformation (18 µm RMS, up to 48 µm: the display's
mapping of the pattern) and half the pixel-grid rounding (separable 18 µm,
non-separable 22 µm). The separable form, 50 parameters instead of 1200,
recovers about two thirds of the gain.

**H1, the board's shape: present, irrelevant.** A shared shape converges to
the same 170–260 µm peak-to-valley in every fold, but it lowers the
held-out residual by 0.002 px. A per-camera shape lowers it more (0.294)
only by growing to millimetres and disagreeing between cameras: it is
soaking up camera error, as on the ring-grid rig.

**H2, the cover layer: not detectable.** Alone, a shared cover fits
`t = 1.08 ± 0.21 mm` and the generic angle shift fits
`a₃ = 0.179 ± 0.022 mm`, mutually consistent with refraction through about
1 mm (`0.185 t`) and stable over the folds. But neither changes the
held-out residual by more than 0.0003 px, and both change sign once the
camera model can move (`t = −1.27 mm`, `a₃ = −0.080 mm` with the field):
what they fit is camera error, not a layer. Whatever cover there is, at
view angles up to 38° its effect beyond what the pose absorbs is below
what this data can see.

**H4, a view-angle-dependent detector bias: not seen** in the form any
view-angle-only shift takes after the pose (the `a₃` term above). Pixel
locking (residual against the sub-pixel phase of the detection) is
0.002 px in `u` and 0.005 px in `v`.

**H3, the camera model, is the dominant term**, as on the ring-grid rig. A
degree-4 image field per camera takes the held-out RMS from 0.315 to 0.186
px. It is a property of each unit: one field shared by the six cameras gets
only to 0.242. Like the ring-grid result, it needs the camera parameters
free to work: with the route's camera held, the same field reaches only
0.280, and with them free it drives `k1` from −0.41 to about −6 and
`tilt_x` from −0.09 to between −0.08 and −0.23 rad, with the field
cancelling most of it. The held-out numbers are real, the parameter values
are not physical, and in places the distortion folds over (the
board-to-image Jacobian turns singular). A field is the measure of how much
a better camera model could gain, not that model.

## The floor

With offsets and a degree-4 field, every camera's held-out mean is
0.114–0.134 px (pooled RMS 0.153, mean 0.124, median 0.103); a degree-6
field takes it to 0.108–0.123 at the price of `fy` (σ 36 px). What is left:

- **A per-view smooth part, ≈ 0.09 px**: a quartic in board position per
  held-out view still explains `sqrt(0.153² − 0.124²)`. It does not repeat
  over views, and it is not the board flexing per capture: mapped into the
  board frame (with the camera-held field model, whose Jacobian stays
  regular), the same capture's per-view fields correlate between cameras
  no more than different captures' do (−0.06 against −0.06; baseline
  +0.03 against +0.03). So it belongs to each camera's view. Its size
  correlates with the view's depth at degree 4 (+0.49 baseline, +0.22
  with offsets and field) but not consistently at degree 2: a hint of the
  depth-dependent error the ring-grid study found, not a finding.
- **≈ 0.12 px of per-detection scatter** that no smooth per-view map
  removes. It is not white: neighbouring corners still correlate +0.15 in
  `u` and +0.19 in `v` after the per-view quartic, and the held-out
  residual is larger in `v` than in `u` (0.118 against 0.098 px RMS). A
  200–300 ppi panel puts a display pixel at 0.5–0.75 camera pixel at this
  scale, so each view samples the display's pixels at a different phase;
  an aliasing pattern that changes from view to view and is correlated
  over neighbouring corners fits that, but this data cannot prove it.

So the reachable floor with this target and detector is a held-out mean of
about 0.12 px (median 0.10); 0.1 px mean would need the depth-dependent
camera term and a cleaner target signal.

## What would settle the rest

- Repeated captures at one pose (target and camera still): sensor noise
  changes between them, a display aliasing pattern does not.
- The same poses with the pattern scaled so a cell is an integer number of
  display pixels: the rounding offsets vanish, and what remains of the
  per-point offsets is the display's in-plane mapping.
- The same rig against a printed board, or the display behind a slight
  defocus: separates the detector-on-display scatter from the lens.
- Views at one image position over a range of depths: separates the
  depth-dependent camera error.
- Views beyond 45°: the only way a cover layer becomes visible at all.

## Recommendation

1. **Per-point target offsets** are the target term the evidence supports:
   opt-in, estimated jointly with the calibration (free target points with
   the similarity gauge held) or applied from a file measured once per
   target and display setting. Backlog `CAL-TARGET-OFFSETS`.
2. **No shape and no cover layer.** Both are consistent across folds when
   alone and change nothing; with the camera free they become camera-error
   absorbers. Not worth a parameter, a config field or a covariance block.
3. **The camera field** (`CAM-RESIDUAL-FIELD`) gains most, and this
   dataset confirms its open design problem: the gain depends on the field
   trading with `k1` and the tilt, so the basis must span what the
   parametric model cannot rather than what it can.

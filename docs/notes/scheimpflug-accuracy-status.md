# Scheimpflug rig accuracy: where the studies stand

A closing record of the accuracy studies on the two private six-camera
Scheimpflug captures (`scheimpflug_rig_ringgrid_cam0..5`, a coded ring grid;
`scheimpflug_rig_ref_cam0..5`, a PuzzleBoard). Both targets are shown on the
same e-ink display and both captures use the same six camera units. Each
line points to the note that holds the evidence. *Measured* means a number a
harness produced; *hypothesis* means not tested.

## What is delivered

| | before | after | where |
|---|---|---|---|
| Ring-grid per-camera intrinsics, pooled RMS px | 0.637 | 0.142 | rig example |
| Ring-grid rig stage, pooled RMS px | 0.804 | 0.258 | rig example |
| Ring-grid hand-eye stage, pooled RMS px | 0.769 | 0.230 | rig example |
| Leave-one-camera-out reprojection, RMS px | 0.877 | 0.268 | rig example |
| Triangulated nearest-neighbour distance error, RMS µm | 100 | 24 | rig example |

The rig example is `scheimpflug_rig_ringgrid_rig` of the private-examples
crate, run once per centre method. The gain is the detector's
([ringgrid-bias](ringgrid-bias.md)): an opt-in photometric centre method
(`center_method = "photometric"`), which removes a blur- and code-dependent
bias of edge-based ring centres. It covers every target kind of the
detector. The cross-camera rows do not rely on a camera fitting its own
observations, and neither run shows a scale bias.

## What was measured and holds

- **The remaining ring-grid residual is not the lens model.** Free `k3`,
  tangential, rational and thin-prism terms, a separate distortion centre,
  distortion in sensor coordinates, a sensor-surface height and the lens-side
  terms of a decentred group each gain a few percent or destabilise the
  tilt ([scheimpflug-residual-model](scheimpflug-residual-model.md),
  [parametric-residual-term](parametric-residual-term.md)).
- **Most of it is a displacement coupled to defocus blur and depth**:
  held-out 0.143 → 0.083 px RMS when modelled, 0.073 (the calibration-free
  floor) with an image-fixed field on top
  ([parametric-residual-term](parametric-residual-term.md)). Its cause is
  open (below).
- **A displayed target adds a fixed offset per feature** from its pixel
  grid: 34 µm RMS on the PuzzleBoard capture, shared by the cameras in the
  board frame, worth 0.315 → 0.269 px held-out
  ([display-target-residual](display-target-residual.md)). Board flatness and
  a cover layer on the display are not detectable.
- **A tilted camera without distortion is a rotated plain pinhole**, also in
  a rig and in hand-eye; the tilt is observable only through distortion or a
  prior. No reparameterisation changes its uncertainty
  ([tilt-identifiability](tilt-identifiability.md)).
- **Fixing an undetermined tilt at zero is sound only with the tangential
  terms freed**: median cost 0.29× the noise floor against 2.0× with them
  held (synthetic; [tilt-determination](tilt-determination.md)).
- **Blur determines the plane of sharp focus** (0.91–0.96 of blur variance)
  and with it the tilt's direction; its size needs pixel pitch and pupil
  magnification ([tilt-from-focus](tilt-from-focus.md)).
- **An unmodelled defocus-coupled displacement biases the
  distortion-derived tilt** by many noise σ in synthetic data, and on the
  real data accounts for 13–51 % of the gap to the focus-derived tilt
  ([tilt-bias](tilt-bias.md)).
- **Photometric fit residuals on real images are half structured**: sensor
  noise is white, the rest is a slowly varying intensity mismatch that is
  not a common profile (3.9 % of variance). The centre covariance that
  accounts for correlated residuals is right when pooled (scatter/stated
  1.16) but 3× short on sharp markers and about right on blurred ones.

## What was tried and rejected

- **A free image-space residual field as the answer.** Its first numbers
  came from evaluating the field at the observed pixel, which shrinks noise
  and passes held-out checks; evaluated at the projection the gain is half
  as large and mostly the defocus coupling
  ([camera-residual-field](camera-residual-field.md)). ADR 0030 is on hold;
  the held-field slot implementation is kept under the tag
  `archive/cam-field-core`.
- **A blur gradient across each marker as the cause of the defocus
  coupling.** Real in synthetic scenes (bias ≈ gradient × blur width), but
  calibrating with a blur-gradient term in the detector model left the
  residual unchanged (0.137 px both ways).
- **A centroid shift proportional to signed defocus.** Not separable from a
  depth term with this capture, and the fitted pupil offset is implausible.
- **Display rasterisation as the source of the extra scatter on sharp
  markers.** At this rig's scale it adds at most 0.005 px in synthetic
  scenes.
- **A tilt / principal-point variable substitution.** Same minimum, same
  tilt uncertainty, slower.

## What is open, and what would settle it

| open question | settles it |
|---|---|
| Cause of the defocus- and depth-coupled displacement (about 0.06 px on the ring-grid rig) | the same poses at two apertures: a geometric term stays, a defocus term scales with blur |
| Whether a window or filter in front of the lenses is tilted to the lens axis (hypothesis: a tilted plate gives a depth-dependent, non-radial shift) | the hardware description; then a plate model with three or four parameters |
| The 0.26–0.41° disagreement between the distortion- and focus-derived tilt | pixel pitch, focal length, f-number and pupil magnification from the datasheets; a capture at two focus settings |
| About 0.05 px of scatter on sharp markers beyond the pixel-noise covariance | repeated frames at one fixed pose; a printed target |
| Why the rigid rig reaches 0.258 px where each camera alone reaches 0.142 px | not investigated |
| The stated hand-eye translation σ of 40–65 mm on a 110 mm translation, and no stated σ for rig extrinsics | not investigated |

## Later captures: untilted devices, optimized and precision targets

Three captures taken after the studies above, on plain-pinhole devices,
measured with the bench (`calib-bench run`, `diagnose floor`) and
leave-one-view-out fits. Mean reprojection error in pixels, default
distortion (`k1 k2 p1 p2`), photometric ring centres:

| capture | floor (RMS) | intrinsics | rig | hand-eye, rigid robot poses |
|---|---|---|---|---|
| three cameras, PuzzleBoard rendered on whole display pixels | 0.067 | 0.056 | 0.067 | 0.25 |
| three cameras, anti-aliased coded ring grid on the display | 0.030–0.037 | 0.030–0.036 | 0.049–0.054 | 0.34–0.38 |
| two cameras, precision plain ring plate (62 px/mm) | 0.015 | 0.034 | 0.060 | 3.9 |

- **Intrinsics and the rig stage are below 0.1 px on every capture**; the
  held-out error is within 10 % of the fitted one, and the calibrated residual sits at the
  calibration-free floor. On the precision plate that is under a micrometre
  on the target, so neither the pipeline nor the photometric ring centres
  carry a floor of their own. Ellipse centres do: 0.16–0.30 px on the same
  images.
- **A target's quality is a length, not a pixel count.** The floor is about
  24 µm on the aliased PuzzleBoard, 18 µm on the whole-pixel one and 8 µm on
  the anti-aliased ring grid; whether that is under 0.1 px depends on the
  camera's magnification. The whole-pixel rendering removes the repeatable
  per-corner offset (16 µm → 4 µm).
- **The tangential terms are real on these lenses** (`p1`, `p2` up to
  2.7e-3, repeating across captures); holding them costs 0.03–0.05 px at the
  rig stage.
- **Hand-eye against the robot poses as given is limited by the robot.** Two
  views commanded to the same pose differ by 0.11 mm on the precision plate;
  the per-pose discrepancy is about 0.1 mm and 0.1° on both devices and
  repeats between captures of one day. Rounding of the pose file explains at
  most 0.27 px of 4.4. In pixels the stage scales with magnification (0.3 px
  at 3.7 px/mm, 4 px at 62 px/mm), so its meaningful unit is millimetres at
  the tool.
- **Hand-eye with per-view robot corrections reproduces the rig number** and
  validates nothing by itself: with few views the corrections absorb the
  whole discrepancy.
- **A pose set of rotations about one point leaves the target's scale
  unobservable** (a free scale fits 1.11 at no cost). Captures need
  translations and more than one rotation centre.
- Open on these captures: robot repeatability against drift against
  kinematic error (repeated visits to one pose over a session would tell);
  a view-by-view disagreement between the two cameras of the plate capture
  that follows the tool orientation (rig flex or a view-angle centre bias);
  a shared out-of-plane shape of about 0.35 mm on the display.

For the tilted rigs above this narrows the cause: the same detector and
pipeline reach the floor on untilted optics, so what remains there belongs
to the tilted optics and their focus gradient.

## What is designed and not built

Backlog entries, each with its evidence in a note: CAL-TILT-DETERMINE and
CAM-PARAM-PRIOR (estimate the tilt only when the data determine it, else a
prior, else zero with the tangential terms freed), CAL-TILT-SYSTEMATIC,
CAL-TILT-FOCUS, CAL-TARGET-OFFSETS, SOLVER-VALLEY-STOP, CAL-TILT-BOUNDS,
CAL-TILT-QUALITY, CAM-FIELD-ORIGIN. The ring-grid centre covariance is not
passed into calibration yet.

## The study harness

`crates/vision-calibration-bench/examples/scheimpflug_model_study/` and
`tilt_identifiability/` are scaffolding. They stay while the open questions
above need them for a new capture and go when those close; the findings
that must not regress are pinned by library tests (`tilt_is_a_rotated_pinhole`
in core, the floor harness tests).

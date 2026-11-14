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
  the held-field slot implementation stays on its branch.
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

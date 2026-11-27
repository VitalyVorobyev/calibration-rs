# Inspecting Results

The Inspect screen judges one run of the open project: how the model fits the data, where it does not, how well each parameter is determined, and whether the calibration passes its [quality checks](quality.md). It works on the run chosen in the run list or the run selector, or the one a calibration just wrote. An export calibrated elsewhere (any `*Export.json` produced by a calibration session, with `diagnostics.image_manifest` populated, or a `vcal run -o` folder) joins the project as a run with **Import Export…**. It works for all seven export kinds, single-camera or rig, with or without a laser stage.

Inspect is a row of tabs; the tab is part of the address, so a link opens the same one. The camera of the per-camera tabs is the app-wide selection: choose it in the tab's camera list (a single-camera run has no list) or step with `,` and `.`. A tab a run cannot have, such as Epipolar for a single camera, is not offered.

| Tab | What it shows |
|---|---|
| **Scene** | the calibrated rig in 3D |
| **Frames** | one frame with its detected corners and residual arrows |
| **Parameters** | every parameter of the camera, value ± σ |
| **Distortion** | the displacement the lens and sensor model make, over the sensor |
| **Residuals** | where the data and the model disagree, per feature and as a map |
| **Projection σ** | how well each pixel's ray is determined |
| **Quality** | the checks of the run's quality report |
| **Epipolar**, **Depth** | the two-camera views, for rigs |

Every map follows the same conventions. A residual is `observed − projected`, in pixels, and a displacement is where a ray lands minus where it would land without the stage. A magnitude is drawn on a sequential colour map from zero to a round top (the 99.5th percentile of the values, rounded up to 1, 2 or 5 × 10ⁿ), a signed quantity on a diverging map centred on zero; the legend carries the unit, says when values exceed its top, and names a limit that is a place on the scale. Nodes without a value are blank, never zero. The pointer reading under the map, the provenance line and the sentence under it say the same in text.

## Parameters

Every parameter of the selected camera, whatever its model, grouped as intrinsics, projection, distortion and sensor. Each row gives the value with its uncertainty at the decimals of σ (`1412.37 ± 0.21`), its unit (a sensor tilt is shown in degrees), and σ as a share of the value for the intrinsics. A parameter without a σ says why: held fixed, not among the estimated parameters, or why the [covariance](uncertainty.md) could not be computed (for example the parameters the data does not determine, or one that ended on a bound, which is marked). **Copy as JSON** puts the camera's parameters on the clipboard as the wire gives them (a tilt in radians), with σ or the reason it is missing. For a rig, the camera-to-rig and other pose parameters the covariance is over are listed under **Pose** by their σ alone, since a pose's covariance is over a small change of it; rotations are in degrees and translations in millimetres. Below the groups: the degrees of freedom, the parameters held fixed, the residual noise estimated from the fit, and a correlation heatmap of |ρ| (hover a cell for the signed value). A result without a covariance says so instead.

## Distortion

The camera's [distortion](distortion.md) as a map over the sensor: for each node, the displacement of where its ray lands against the camera without that stage. **Layer** chooses the lens alone, the sensor tilt (offered when the sensor model can add one) or both together; **Component** chooses the magnitude `|d|` or the signed `dx` and `dy`. **Iso-lines** draw lines of equal value, labelled; **Arrows** draw the direction of the displacement; **Image** puts the frame of the selected view under the map (not for a tiled camera, whose frame is a region of a larger image). **Colours** and **Range** set the colour map and fix the top of the scale so two cameras or two runs can be compared on one scale. The map has no uncertainty of its own: the lens layer can be large while the projection stays precise, see Projection σ.

## Residuals

The residuals of the selected camera, in three parts.

- **The residual map** is the systematic part of the residuals, smoothed over the sensor: where the data and the model disagree in the same way across views. **Show** chooses the systematic error `|μ|`, the RMS, its components, the standard error, the significance χ and the support (effective samples and views); **Views** maps all views, the selected view or a chosen few. Arrows give the direction of the mean residual. Where a field is distinguishable from zero (χ at least 3.03, p < 0.01) it is drawn in full colour, elsewhere faded, and **Fade not significant** switches that off. With one view, or too few effective views, there is no significance test: the map includes the pose error of the views and nothing is faded. For a rig or hand-eye run the residuals include the pose chain's error. The sentence under the map states where the systematic error peaks and how much of the area that could be tested differs from zero. A significant field is not a reason to add a model term, and a flat map does not prove the model.
- **The per-feature scatter and table** show every feature's residual, `dy` against `dx` with equal axes (one series per view while there are six or fewer), and the same records as a table sorted worst first. Choosing a point or a row selects its view and feature; the **Frames** tab then shows that view and rings the feature. Records left out of the map as outliers are marked.
- **The statistics table** lists the mean, RMS and largest error per camera, per view or per camera and view; a row selects its camera and view.

A run without per-feature residuals still opens: the map says it cannot be made and the tables say there is nothing to list.

## Projection σ

The standard deviation, in pixels, of where the ray through each pixel lands, from the covariance of the camera's own parameters, on the same grid as the other maps ([Projection Uncertainty](uncertainty_projection.md)). A run without a covariance, or whose camera parameters were all held fixed, says why there is no map. Rig results hold each camera's intrinsics fixed in the final solve unless `rig.refine_intrinsics` is set, so their projection map is unavailable by default.

## Quality

The run's [quality report](quality.md): the overall verdict, the worst of the assessed checks, and each check with its verdict (pass, warning, fail, or not applicable with the reason), what it examines, the measured value and the limit it was held to, and one sentence on the outcome. Checks of other cameras are not listed; a check offers to open the tab that shows more of what it is about. An imported run whose quality could not be assessed has no report and says so.

## Frames

The frame viewer: step through `(view, camera)` frames with the steppers or the arrow keys. The source image renders with the detected corners, each with an arrow from the observed corner along `observed − projected` (magnified, the factor is in the legend), colour-coded by reprojection error. These are the [per-feature residuals](per_feature_residuals.md) of the export. **Compare** shows two frames side by side with a linked viewport; **Laser** switches a target frame to its paired laser frame with the point-to-plane residuals. A histogram of the image and the cursor position sit below.

## Scene

A Three.js scene of the calibrated rig: one frustum per camera (bounded by the back-projected image border, so lens distortion shows), the target board at the selected pose or at all poses at once, and each laser plane as a light-sheet fan rooted beside its camera when the export carries them. Click a frustum to select a camera and see its intrinsics and its pose relative to a chosen reference camera; click a board to select a pose. Rig exports render directly. A single-camera `laserline_device` export also renders: its camera sits at the rig origin, and its camera-frame laser plane and per-view poses are drawn in that frame. The other single-camera exports carry no extrinsic to place a frustum and show an empty-state message.

## Epipolar

Click a point in one camera's image and the corresponding epipolar line is drawn in a second camera's image, computed from the calibrated relative pose, raw or undistorted. It is a quick visual sanity check that two cameras in a rig are calibrated consistently with each other (see [Epipolar Geometry](epipolar.md) for the maths). It needs a rig export with at least two cameras.

## Depth

Pick two cameras and a pose from a rig export and click compute. The app rectifies the pair ([Stereo from a Calibration](mvg_rectification.md)), runs block matching or semi-global matching ([Dense Stereo](mvg_dense.md)) and gives you rectified, disparity, overlay, depth and 3D point-cloud views. Toggle semi-global aggregation to fill in the low-texture regions that plain block matching leaves blank. It needs a stereo rig export (two cameras with extrinsics), the same gate as Epipolar.

## Walkthrough: Read a Run

1. Calibrate the project (or open one with runs, or import an export), and choose the run.
2. **Quality** says whether the run passes and which checks need a look.
3. **Residuals** shows where the model and the data disagree; choose a point of the scatter to see that corner in **Frames**.
4. **Parameters** and **Projection σ** say how well the result is determined; **Distortion** shows what the lens does to a pixel.

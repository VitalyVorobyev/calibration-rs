# Inspecting Results

The four result workspaces work on the currently loaded export. Load one with **Open Export…** in the top bar (any `*Export.json` produced by a calibration session, with `diagnostics.image_manifest` populated), or arrive with one from a run. The viewer infers the export kind from its JSON shape (planar, rig hand-eye, rig plus laserline, ...) and shows the matching header label. It works for all eight export kinds, single-camera or rig, with or without a laser stage.

## Diagnose

`⌘1`, the default workspace: the residual viewer.

- Step through `(pose, camera)` frames with the pose and camera steppers or the arrow keys. The source image renders with the detected corners colour-coded by reprojection error, a heatmap on which the brightest corners are the worst. These are the [per-feature residuals](per_feature_residuals.md) of the export.
- A histogram panel summarizes the distribution under the cursor's region of interest.
- **Compare mode** shows two frames side by side with a linked viewport, so pan and zoom stay in sync.
- The **laser frame** toggle switches a target frame to its paired laser frame and shows the point-to-plane residuals instead of the corner reprojection error.
- **Stats** opens a side panel with a sortable per-pose residual table (mean, median and maximum in pixels, and the feature count; click a row to jump the viewer to that pose) and, for multi-camera exports, a cameras by poses residual matrix coloured by mean error (click a cell to jump to that `(pose, camera)`).
- **Uncertainty** shows the result's [parameter uncertainty](uncertainty.md): a table of each estimated parameter's value and σ (angles in degrees, translations in millimetres; poses show σ only), the parameters held fixed, the residual noise, a correlation heatmap of |ρ| (hover a cell for the signed value) and the [projection σ map](uncertainty_projection.md) of the camera on screen, computed at its frame's size. A result without a covariance says why, for example which parameters the data does not determine. Rig results hold each camera's intrinsics fixed in the final solve unless `rig.refine_intrinsics` is set, so their projection map is unavailable by default.

## 3D Viewer

`⌘2`: a Three.js scene of the calibrated rig. It draws one frustum per camera (bounded by the back-projected image border, so lens distortion shows), the target board at the selected pose or at all poses at once, and each laser plane as a light-sheet fan rooted beside its camera when the export carries them.

Click a frustum to select a camera and see its intrinsics and its pose relative to a chosen reference camera; click a board to select a pose. Rig exports render directly. A single-camera `laserline_device` export also renders: its camera sits at the rig origin, and its camera-frame laser plane and per-view poses are drawn in that frame. The other single-camera exports (planar, Scheimpflug, single-camera hand-eye) carry no extrinsic to place a frustum and show an empty-state message instead.

## Epipolar

`⌘3`: click a point in one camera's image and the corresponding epipolar line is drawn in a second camera's image, computed from the calibrated relative pose, raw or undistorted. It is a quick visual sanity check that two cameras in a rig are calibrated consistently with each other (see [Epipolar Geometry](epipolar.md) for the maths). It needs a rig export with at least two cameras.

## Depth

`⌘4`: pick two cameras and a pose from a loaded rig export and click compute. The app rectifies the pair ([Stereo Rectification](mvg_rectification.md)), runs block matching or semi-global matching ([Dense Stereo](mvg_dense.md)) and gives you rectified, disparity, overlay, depth and 3D point-cloud views. Toggle semi-global aggregation to fill in the low-texture regions that plain block matching leaves blank. It needs a stereo rig export (two cameras with extrinsics), the same gate as Epipolar.

## Walkthrough: Load an Export into Diagnose

1. Click **Open Export…** in the top bar.
2. Pick any `*Export.json` produced by a calibration session, from a Run or written by `session.export()` in your own code with `diagnostics.image_manifest` populated.
3. Diagnose infers the export kind from the JSON and shows the right header label. Step through frames with the arrow keys or the steppers.

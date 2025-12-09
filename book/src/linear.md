# Linear Calibration Building Blocks

Closed-form pieces that seed non-linear refinement.

- Homography estimation (DLT, normalization, degeneracies, conditioning tips).
- Planar pose from homography and Zhang-style planar intrinsics initialisation.
- Epipolar geometry: eight-point / five-point roadmap, normalization, rank enforcement.
- PnP solvers for sparse point correspondences and robustness strategies.
- Multi-camera rig extrinsics: pairwise averaging, reference frame selection, averaging strategies.
- Hand–eye calibration variants (AX=XB, separable rotation/translation, noise handling).
- Validation: synthetic test harnesses and residual inspection.

> TODO: include derivations and worked numeric examples.

# Pipelines

End-to-end flows that combine linear seeds with non-linear refinement.

- Planar intrinsics pipeline: input JSON schema, normalization, initial guess, LM refinement, report contents.
- Laserline device pipeline: joint intrinsics + laser plane calibration from planar target + laser pixels.
- Handling multiple views, outlier filtering, and robust kernels.
- Guidelines for detector integration (chessboard, AprilTag, dot grid).
- Extending to multi-camera rigs and stereo calibration (planned).
- Exporting results to JSON, and wiring outputs into downstream applications.

> TODO: add walkthrough with provided sample data and visualization scripts.

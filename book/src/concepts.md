# Core Concepts

Shared mental model for the project.

- Coordinate frames and notation conventions (`C`, `T`, `R`, homogeneous transforms).
- Camera models implemented in `calib-core` (pinhole, radial-tangential distortion, linescan roadmap).
- Error metrics and residuals; reprojection cost vs. algebraic errors.
- Robust estimation: RANSAC traits, consensus scoring, and kernel selection.
- Data serialization with `serde` for reproducible experiments.

> TODO: add figures for frame relationships and distortion mapping.

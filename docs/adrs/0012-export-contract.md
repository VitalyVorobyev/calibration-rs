# ADR 0012: The Export Is the Result's Wire Contract

## Context

A calibration's `*Export` JSON is what the app, Python, notebooks and the
regression suite consume. If it carried only parameters and aggregate
statistics, every consumer would have to re-implement calibration math
(re-projection, laser residuals, frame chains) to diagnose a result, and
each would identify the export's problem type by probing fields.

## Decision

Every pipeline `*Export` is self-describing and carries its diagnostics,
grouped apart from the result: the result fields (camera parameters,
poses, uncertainty, reprojection error) are the stable contract, and
`diagnostics` (`ExportDiagnostics`) holds the evidence for viewers.

- **`kind`** — a required discriminator, the shared `ExportKind` enum
  (snake_case), serialized as the first field. Consumers branch on it;
  deserialization without it fails.
- **`diagnostics.per_feature_residuals`** — per-(view, camera, feature) records:
  target reprojection residuals, laser residuals in metres and in the
  solver's pixel line distance, the projected laser line per (view, camera)
  as a polyline (distortion bends it), and optional per-camera histograms
  over the fixed bucket edges `[1, 2, 5, 10]` px.
  - Records are flat, pose-major vectors (view, then camera, then feature),
    each carrying its explicit indices. The order is part of the contract.
  - A value that does not exist (a point behind the camera, a ray missing
    the plane) is `None`, never a sentinel. An empty vector means the
    problem type has no observations of that kind.
  - Diagnostics are computed by the same kernels as the cost, so they agree
    with what the solver minimized.
- **`diagnostics.image_manifest`** (optional) — the frames behind each (view, camera)
  slot, with an optional ROI for tiled multi-camera images and a
  `FrameKind` (`target` | `laser`). It is written by the runner, never read
  by the pipeline, and indexed like the residuals.
- **`uncertainty`** — the parameter covariance (ADR 0026).
- Solver statistics (`SolveReport`) stay on the session output, not the
  export.

`ProblemType::export` receives the input, since the observed pixels are
needed for the records. Residual helpers live where their math lives:
target residuals in core, laser residuals in optim; the record types are
in core.

### Evolution

- Export envelopes and record types are `#[non_exhaustive]`.
- Adding a field with `serde(default)` keeps `schema_version`; renaming,
  removing or reordering bumps `ProblemType::schema_version()` (ADR 0007).
- Optional fields are skipped when `None`, so exports without them stay
  byte-stable.

## Consequences

- Viewers are thin renderers of static JSON, not parallel implementations
  of the geometry.
- Export size grows with the dataset (a 20-view, 6-camera rig is on the
  order of 15k records); the extra work is negligible next to the solve.
- A `ProblemType` outside the workspace must accept the input in `export`.

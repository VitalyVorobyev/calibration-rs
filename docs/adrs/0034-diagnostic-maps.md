# ADR 0034: Diagnostic Maps Share One Sign, One Grid, and Are Derived

## Context

Users and the app want to see two things over a camera's image: what the
lens and the sensor model do to each ray (a distortion map), and where the
model leaves a systematic misfit in the data (a residual map). Before this,
the HTML report binned `|observed − projected|` at the *observed* pixel into
16 columns, the app drew per-feature arrows `projected − observed`, and the
solver minimized `observed − projected`: three conventions, and a binning
that correlates position with the noise it averages. The forces: a missing
distortion term should look the same in both maps; the maps must not mislead
where the data do not reach; and they must cost no wire format.

## Decision

- **One sign: `actual − reference`, in pixels.** A residual is
  `observed − projected` (`TargetFeatureResidual::residual_px` is its single
  definition, the solver's sign). A distortion layer is the full model's
  pixel minus the simpler model's pixel at the same ray. A term the model
  lacks then shows in the residual map with the sign and units it has in a
  distortion layer. The app's arrows follow this sign.
- **Residuals sit at their projection**, never their observation (the
  evidence rule of ADR 0030).
- **Distortion layers telescope in pipeline order**, reference = the same
  camera with the stage removed, same projection, same `K`:
  `distortion = K∘D∘P − K∘P`, `sensor = K∘S∘D∘P − K∘D∘P`,
  `total = distortion + sensor` (exact). For a fisheye the reference is the
  equidistant `f·θ` image, defined beyond 90°; a pinhole reference is not
  offered.
- **A node has a value only on the lens's own sheet**: the inverse sensor
  homography is finite, the backprojected ray round-trips to the node
  within `1e-6` px, the ray is in the projection's domain, and
  `det ∂D/∂n > 1e-6` on 16 points of the segment from the axis to the ray.
  The determinant at the ray alone is not enough: a Newton inverse past a
  radial fold lands on the branch through the axis or on a rising outer
  branch, both orientation-preserving and both round-tripping exactly.
- **No σ on the distortion map.** The `K`/distortion split is a trade-off
  direction (`fx` against `k1`); the gauge-free precision is the projection
  covariance, evaluated at the same nodes.
- **Residual field estimator: Gaussian local-linear**, bandwidth by K-fold
  cross-validation over *views* (ladder 3–22 % of the diagonal, one-SE rule
  toward the smoother map), standard error clustered by view (CR1 with the
  effective number of views), significance `χ = √(μ̂ᵀV̂⁻¹μ̂)`; support by
  effective samples (≥ 8) and the local design's conditioning; outliers
  beyond 8 robust σ left out and listed. Defaults are options.
- **One grid**: `ImageGrid`, the centres of `stride × stride` pixel blocks,
  row-major, which is the app's dense-plane layout. Planes are row-major
  `Vec<f64>` with `NaN` for no value.
- **Derived on demand, not exported**: no export field, no JSON schema. The
  maps are deterministic functions of an export (and an image size).
- **The report reuses `residual_map`** instead of binning.

## Consequences

- Additive API: `core::{ImageGrid, DistortionMap, DisplacementPlane,
  distortion_map}`, `pipeline::analysis::{residual_map, ResidualMap,
  ResidualMapOptions, Bandwidth, BandwidthSelection}`,
  `TargetFeatureResidual::residual_px`, under the facade's `analysis`.
- Measured (tests in `maps/tests.rs`, `analysis/residual_map/tests.rs`,
  `tests/residual_map_end_to_end.rs`): local-linear corner error 0.0065–
  0.0088 px vs 0.069–0.079 px for the kernel average on a 0.3 px cubic
  field at `h` = 100 px; clustered SE / Monte-Carlo scatter 0.81–1.07 over
  36 captures (6–100 views); CV picks one of the two smoothest bandwidths
  under no field in every run.
- Known limits, measured: `χ > 3.03` under no field occurs at ~3 % of nodes
  with ~20 effective views and 4–19 % with ~6 (a Hotelling-type tail: V̂ is
  itself estimated from the views), against the nominal 1 %; and the SE
  holds no fit-parameter noise, so on a calibration with a wide kernel
  88 % (not 95 %) of nodes lie within 3 SE of the noise-free refit's map.
  A small-sample (F) reference for `significance` and a parameter-noise term
  in the SE are the candidate fixes; until then the docs say so.
- A significant field is not a model-term decision (ADR 0032 decides those
  on held-out views), and chain-level residuals (hand-eye, rig) include pose
  chain error; free-pose residuals for those exports are a follow-up.

## Alternatives rejected

- **`projected − observed` for the residual (the app's former arrows).**
  The residual and distortion maps would show the same missing term with
  opposite signs.
- **Binned or Nadaraya–Watson means.** Biased `O(h·|∇μ|)` at the coverage
  edge, where distortion misfit is largest (10× the local-linear error in
  the test above).
- **Leave-one-feature-out bandwidth selection.** It fits each view's
  pose-correlated pattern as field and picks too small an `h`.
- **Feature-level (iid) standard errors.** Residuals of a view share its
  pose error; the iid SE understates the scatter (2.6× at the image centre
  in the Monte-Carlo test).
- **Exporting the maps.** A dense float grid is a poor JSON payload, and
  the maps are recomputable exactly.
- **A pinhole reference for fisheye distortion.** Undefined at 90° and
  dominated by the projection choice, not the lens.

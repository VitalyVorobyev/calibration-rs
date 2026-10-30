# Detector feature covariance on real data — pre-registered study

Does weighting a calibration by the corner detector's per-corner position
covariance make it better, or its stated uncertainty more honest, **on real
images**? This note fixes the arms, metrics, datasets and decision rule before
the first run. Its result section is appended after the run and does not
change anything above it.

The weighting is in: observations may carry a 2×2 covariance, the
reprojection residuals are whitened by it, and the export reports σ̂₀
(`uncertainty.md` §Per-point covariance). The chessboard and ChArUco detectors
report the covariance on request (`detector.chess_corners.covariance`). The
Monte-Carlo study in `uncertainty.md` shows the weighting is right *when the
stated covariances are true*. It cannot show that the detector's covariances
are true for real images, and synthetic renders are too ideal to show it,
so this study uses real images only.

Prior evidence, from the corner-localizer study in calib-targets-rs:
thresholding corners by their reported covariance did not improve held-out
geometry. Weighting is a different question; it is open.

## Arms

Every comparison is paired: the same images, the same detector settings and
the same corners. Only the weighting differs.

| Arm | Observations carry |
|---|---|
| `uniform` | no covariance; the default |
| `known1` | the detector covariance with `image_noise = { known = 1.0 }` |
| `estimated` | the detector covariance with `image_noise = "estimate"` (per-image noise) |
| `uniform_noise` | `uniform`'s corners plus isotropic Gaussian noise, σ = 0.05 px per axis, fixed seed (power check) |

- **`known1`.** The calibration's variance factor σ̂₀ absorbs any common scale.
  This arm therefore tests the covariance's per-corner structure from the
  localizer and its systematic-error model, with every image weighted alike.
  It runs on every dataset, JPEG included.
- **`estimated`.** Adds the differences in noise between images. It runs only
  where the estimate is available for every image. Where it declines, the
  run's typed error is recorded and the arm is absent for that dataset.
- **Corner sets.** The covariance does not move corners: the detector returns
  the same positions with and without it, which a check asserts before any
  metric. All arms therefore use the same corner set.
- **Localizer.** The default center-of-mass localizer only. A localizer change
  is a separate study.

## Datasets

Each dataset is calibrated with its acceptance-registry configuration: model,
fix lists and route as `calib-bench accept` runs it.

| Kind | Dataset | Role |
|---|---|---|
| public chessboard | `stereo_left`, `stereo_right` | E1–E3 |
| public chessboard | `kuka_1` | E1–E3, E4 (hand-eye) |
| public chessboard | `ds8` cameras 0–2 | E1–E3, E4 (rig hand-eye) |
| public chessboard | `stereo_rig` | E4 (rig) |
| private ChArUco | `stereo_charuco` cameras 1, 2 | E1–E3, E4 (rig) |
| private ChArUco | `scheimpflug_rig` tiles 0–5 | E1–E3 |

- **Excluded.** PuzzleBoard and ring-grid sets, which have no covariance.
- **Decisive datasets.** A dataset is decisive only if it passes the power
  check: `uniform_noise` raises the E1 held-out RMS by an amount whose 95 %
  interval excludes 0. A dataset that fails is reported, not decisive.
- **Private numbers.** They appear only as aggregates (medians and intervals),
  never per image.

## Metrics

All statistics are paired by view. Every interval is a 95 % bootstrap over
views (10 000 resamples, fixed seed). Datasets are pooled by the median of
their per-dataset effects.

- **E1 Accuracy.** Both measures are in raw pixels, never whitened, so an arm
  cannot grade itself.
  - *Held-out reprojection RMS.* Folds are made by view: view `i` goes to fold
    `i mod 5`. Calibrate on four folds, then fit each held-out view's pose
    with the camera fixed, and take the RMS of its residuals.
  - *Homography RMS after undistortion.* Undistort the held-out view's corners
    with the fold's camera, fit a board-to-image homography (normalized DLT
    plus Gauss–Newton), and take the RMS.
  - Effect: arm − `uniform`, per view.
- **E2 Efficiency.** Does the weighting make the estimate less variable?
  - *Split-half projection difference.* Split the views into two halves, A
    and B, calibrate each, and compare the cameras where they matter. Cast
    rays through a 16 × 12 pixel grid with camera A, project them with
    camera B, and take the RMS pixel distance.
  - Use 20 seeded random half-splits plus the even/odd split; the effect is
    the median over splits, arm − `uniform`.
  - Comparing rays (points at infinity) removes the frames' relative pose.
    Only intrinsics are compared.
- **E3 Honesty.** Does the stated uncertainty match the real scatter?
  - For the same splits and grid, with `d` the projection difference and
    `Σ_A`, `Σ_B` each half's 2 × 2 projection covariance at that pixel
    (`CalibratedCameras::projection_uncertainty`):
    `z² = dᵀ (Σ_A + Σ_B)⁻¹ d`.
  - An honest uncertainty gives a mean `z²/2` of 1, with 68.3 % of grid points
    below `z² = 2.30` and 95.4 % below `6.18` (χ² with 2 degrees of freedom).
  - Reported per arm: the median over splits of the mean `z²/2`; both coverage
    figures; and σ̂₀ of the full calibration (σ̂ for `uniform`).
- **E4 Downstream.** These are reported and are not part of the decision.
  - *Rigs* (`stereo_rig`, `stereo_charuco`, `ds8`): the split-half difference
    of each camera-to-rig transform, in rotation (degrees) and translation
    (millimetres).
  - *Hand-eye* (`kuka_1`, `ds8`): the split-half difference of the hand-eye
    transform, and its stated σ.
- **E5 Cost.** Detection wall time with the covariance on and off, as the
  median over images of a single-threaded release build.

## Decision rule

The default stays `uniform` unless one of the following holds.

- **Flip the default to `known1`** (or to `estimated` where every image's
  noise is estimable) only if all of these hold:
  1. *Not worse anywhere.* E1, on both measures, is not worse on any decisive
     dataset with an interval excluding 0, and E2 is not worse on any decisive
     dataset with an interval excluding 0.
  2. *Better somewhere that matters.* Either the pooled E1 effect is negative
     with an interval excluding 0 on both measures, or E3 is materially more
     honest. "Materially" means the pooled `|mean z²/2 − 1|` is smaller than
     `uniform`'s by at least 0.2, and both coverage figures are within ±5
     points of nominal.
  3. *Cheap enough.* E5 adds at most 15 % to the detection time.
- **Keep it opt-in, documented as a reporting and weighting option,** if
  E1, E2 and E3 are all neutral (no interval excluding 0, and no material E3
  difference).
- **Recommend against weighting by it** (in the rustdoc and the book) if E1 or
  E2 is worse on any decisive dataset with an interval excluding 0. The
  covariance stays available as a report.

A change to this rule after the first run is a new pre-registration, written
before the run it governs.

## Running

`calib-bench feature-cov` (vision-calibration-bench) runs every arm through
the dataset runner. It writes to `target/feature-cov/`, which is git-ignored.
The public part is reproducible from a clean checkout after
`calib-bench suite fetch`; the private part needs the local registry.

## Result

(Appended after the run.)

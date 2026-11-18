# Detector feature covariance on real data — pre-registered study

Does weighting a calibration by the corner detector's per-corner position
covariance make it better, or its stated uncertainty more honest, **on real
images**? **Settled (2026-10-31): no.** On real chessboard and ChArUco images
the weighting does not improve held-out accuracy, on some cameras it makes it
worse, and the library recommends against it; the detector still reports the
covariance. The arms, metrics, datasets and decision rule below were fixed
before the first run (one amendment, written before the full run, is folded
into the method and listed under "Pre-registration"); the result section was
appended after the run.

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
geometry. Weighting is a different question; it was open.

## Arms

Every comparison is paired: the same images, the same detector settings and
the same corners. Only the weighting differs.

| Arm | Observations carry |
|---|---|
| `uniform` | no covariance; the default |
| `known1` | the detector covariance with `image_noise = { stated = 1.0 }` |
| `estimated` | the detector covariance with `image_noise = "estimate"` (per-image noise) |
| `uniform_noise` | `uniform`'s corners plus isotropic Gaussian noise, σ = 0.05 px per axis, fixed seed, on training and held-out pixels alike (power check) |

- **`known1`.** The calibration's variance factor σ̂₀ absorbs any common scale.
  This arm therefore tests the covariance's per-corner structure from the
  localizer and its systematic-error model, with every image weighted alike.
  It runs on every dataset, JPEG included.
- **`estimated`.** Adds the differences in noise between images. It runs only
  where the estimate is available for every image; where it declines, the
  arm is absent for that camera with its typed error recorded.
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
- **Unit.** The camera. A rig camera enters E1–E3 as a planar calibration
  with the rig route's camera configuration; E4 covers the joint solves.
- **Decisive cameras.** A camera is decisive only if it passes the power
  check: `uniform_noise` raises the E1 held-out RMS by an amount whose 95 %
  interval excludes 0. A camera that fails is reported, not decisive.
- **Private numbers.** They appear only as aggregates (medians and intervals),
  never per image.

## Method

Effects are arm − `uniform`, lower is better. E1 intervals are a 95 %
bootstrap over views, E2 and E3 intervals one over splits (10 000 resamples,
fixed seed). Cameras are pooled by the median of their per-camera effects.

- **E1 Accuracy**, in raw pixels, never whitened, so an arm cannot grade
  itself; paired by view.
  - *Held-out reprojection RMS.* View `i` goes to fold `i mod 5`. Calibrate
    on four folds, fit each held-out view's pose with the camera fixed, and
    take the RMS of its residuals.
  - *Homography RMS after undistortion.* Undistort the held-out view's
    corners with the fold's camera, fit a board-to-image homography
    (normalized DLT plus Gauss–Newton), and take the RMS.
- **E2 Efficiency.** Split the views into halves A and B (the even/odd split
  plus 20 seeded random half-splits) and calibrate each. Cast rays through a
  16 × 12 pixel grid with camera A, project them with camera B, and take the
  RMS pixel distance; the effect is the median over splits. Comparing rays
  removes the halves' relative pose, so only intrinsics are compared. The
  decisive grid is the nodes inside the convex hull of all the camera's
  detected corners; the full grid, which extrapolates, is reported only.
- **E3 Honesty.** For the same splits and decisive grid, with `d` the
  projection difference and `Σ_A`, `Σ_B` each half's 2 × 2 projection
  covariance at the node (`CalibratedCameras::projection_covariance`):
  `z² = dᵀ (Σ_A + Σ_B)⁻¹ d`, χ² with 2 degrees of freedom when the stated
  uncertainty is honest. Reported: the pooled ratio
  `r = Σ z² / (2 · number of (split, node) pairs)`, 1 when honest; the
  pooled coverage below 2.30 and below 6.18 (nominal 68.3 % and 95.4 %); and
  σ̂₀ of the full calibration (σ̂ for `uniform`).
  *Gate:* on synthetic scenes with exactly true noise, `r` within 0.85–1.15
  and both coverages within ±5 points of nominal, before any real-data run.
- **E4 Downstream**, reported only, not part of the decision: the split-half
  difference of each camera-to-rig transform (`stereo_rig`,
  `stereo_charuco`, `ds8`) and of the hand-eye transform with its stated σ
  (`kuka_1`, `ds8`), in degrees and millimetres.
- **E5 Cost.** Detection wall time with the covariance on and off, the
  median over images of a single-threaded release build.

An effect counts as **better** or **worse** only if its interval excludes 0
*and* its size reaches the practical margin; anything smaller is
**neutral**.

| Metric | Practical margin |
|---|---|
| E1, both measures | `max(0.005 px, 1 % of uniform's held-out RMS)` |
| E2 | 2 % of uniform's split RMS |

## Decision rule

The default stays `uniform` unless one of the following holds.

- **Flip the default to `known1`** (or to `estimated` where every image's
  noise is estimable) only if all of these hold:
  1. *Not worse anywhere.* No E1 measure and no E2 effect is **worse** on a
     decisive camera.
  2. *Better somewhere that matters.* The pooled E1 effect is **better** on
     both measures, or E3 is materially more honest: the arm's pooled
     `|r − 1|` is at most 0.8 × uniform's, and each of its two pooled
     coverages is at least 5 points closer to nominal than uniform's.
  3. *Cheap enough.* E5 adds at most 15 % to the detection time.
  4. *Available.* The arm is present on every decisive camera: a default the
     detector cannot honour everywhere is not a default.
- **Keep it opt-in, documented as a reporting and weighting option,** if
  every effect is neutral.
- **Recommend against weighting by it** (in the rustdoc and the book) if any
  weighting arm is **worse** on a decisive camera. The covariance stays
  available as a report.

## Pre-registration

The arms, datasets, metrics and rule were written before the first run.
Amendment 1 was written after smoke runs (2 splits, 500 resamples, 3 timing
images per camera, whose outcome counted for nothing) and before the full
run. Each change fixed a measurement that did not measure what the rule
assumed; none was motivated by which arm the smoke runs favoured.

1. **E3 statistic.** The original, the median over splits of each split's
   mean `z²/2`, read 0.64 on synthetic truth: a split's mean is dominated by
   a few parameter directions and skewed. The pooled ratio `r` replaced it,
   with the synthetic truth gate above.
2. **E2/E3 grid.** The full grid reaches image corners no training view
   covers, which dominated the split RMS (1–11 px); the decisive grid became
   the corners' hull.
3. **Practical margins.** E1 intervals excluded 0 on effects of about
   1e-4 px; the margins above were added.
4. **"Materially more honest."** Real-data `r` was 2–17 for every arm, so the
   original 0.2 absolute gap was noise and coverage within ±5 points of
   nominal out of reach; the relative criterion above replaced them.
5. **Availability.** On two Scheimpflug tiles some corners carried no
   covariance: the ChArUco wrapper did not pass the `chess_corners` override
   to the detector's corner re-detection. Fixed before the full run; the
   availability condition (4) was kept.
6. **Pooling.** The camera became the pooling unit, and rig cameras enter
   E1–E3 as planar calibrations.

## Tests that pin it

The real-data outcome needs the images and is not a test; what it rests on
is pinned:

- `crates/vision-calibration-optim/tests/feature_covariance.rs` — the
  whitened solve's covariance agrees with the scatter of Monte-Carlo solves
  when every point states its true covariance, single camera and rig
  (`--ignored`, run by CI's feature-covariance gate in release), and a
  whitened solve is consistent in the default run. The study's premise:
  weighting is right when the stated covariances are true.
- `crates/vision-calibration-bench/tests/split_half_honesty.rs` — the E3
  synthetic truth gate (`synthetic_truth_gate`: 24 scenes, with and without
  a stated covariance) and its companion with an error each view's corners
  share (`synthetic_truth_gate_with_view_errors`, the view scatter at work):
  the split-half `r` and both coverages within the gate's bounds (today
  `r` 0.886 and 0.937, against 0.908 at the study's run).

The study's harness (`calib-bench feature-cov`, with E1–E5, the decision
rule and the report) was removed once the study was settled; it is in the
history up to commit `160b9afc`.

## Result

Run on 2026-10-31 at the protocol settings: the even/odd split plus 20 seeded
random splits, 10 000 bootstrap resamples, and E5 single-threaded in a release
build. Amendment 1 applied, and the ChArUco re-detection fix was in. The
public and private registries ran separately; the rule is applied over both.

**Before any outcome.**
- *Synthetic truth gate:* passed. Over 24 scenes and 34 335 (split, node)
  pairs, `r` = 0.908, with 71.3 % below 2.30 and 96.9 % below 6.18.
- *Corner sets:* identical across arms.
- *Arms:* every arm was present on every camera. No covariance was missing,
  and `estimated` declined nowhere.
- *Power check:* 9 of 14 cameras are decisive: all 6 public cameras,
  `stereo_charuco` cameras 1 and 2, and `scheimpflug_rig` tile 5. On tiles
  0–4, the 0.05 px power noise does not move a held-out RMS of about 1 px
  beyond its interval.

**Outcome: recommend against weighting by the detector covariance.**
- Public cameras alone: every effect of every arm is Neutral (`KeepOptIn`).
- Private cameras: `known1` is Worse on two decisive cameras and `estimated`
  on one (`RecommendAgainst`).
- The clause "any weighting arm Worse on a decisive camera" decides, so
  pooling does not enter. No pooled effect is Worse either.

### E1–E3 on the public cameras

`known1` against `uniform`. `estimated` agrees with `known1` to the digits
shown on every camera, and their σ̂₀ differ by at most 0.3 %.

| camera | uniform held-out RMS | E1 held-out | E2 split RMS (effect) | E3 r: uniform / known1 | ≤ 2.30 / ≤ 6.18 (uniform) | σ̂ / σ̂₀ |
|---|---|---|---|---|---|---|
| `stereo_left` | 0.270 px | −0.0000 N | 6.79 px (+0.02 N) | 6.21 / 6.34 | 20.6 % / 42.2 % | 0.177 px / 1.52 |
| `stereo_right` | 0.281 px | −0.0000 N | 11.23 px (−0.10 N) | 9.54 / 9.59 | 11.7 % / 26.6 % | 0.186 px / 1.59 |
| `kuka_1` | 0.153 px | +0.0000 N | 1.51 px (−0.01 N) | 6.72 / 6.71 | 23.3 % / 44.7 % | 0.108 px / 0.98 |
| `ds8` camera 0 | 0.188 px | +0.0000 N | 5.24 px (−0.00 N) | 8.88 / 8.91 | 10.2 % / 26.4 % | 0.132 px / 1.28 |
| `ds8` camera 1 | 0.189 px | −0.0000 N | 7.46 px (+0.02 N) | 26.0 / 26.3 | 7.2 % / 16.2 % | 0.130 px / 1.24 |
| `ds8` camera 2 | 0.256 px | +0.0000 N | 4.68 px (+0.04 N) | 4.93 / 4.91 | 34.3 % / 54.8 % | 0.186 px / 1.60 |

- **E1.** Both measures are Neutral on every public camera. The effects are
  at most 2 × 10⁻⁴ px, against a margin of 0.005 px.
- **Power check.** `uniform_noise` raised the held-out RMS by 0.009–0.015 px,
  well resolved on every camera.
- **Pooled over the 6 cameras.**
  - E1: −0.0000 px on both measures.
  - E2: +0.007 px [−0.025, +0.031] (margin 0.12).
  - E3 `r`: `uniform` 7.80 [6.17, 9.44], `known1` 7.81, `estimated` 7.81.
  - E3 coverage, `uniform` / `known1`: 16.1 % / 16.8 % below 2.30, and
    34.4 % / 34.3 % below 6.18.

### E1–E3 on the private cameras (aggregates)

- **`stereo_charuco` camera 1.**
  - E1: Neutral (≤ 3 × 10⁻⁴ px on a 0.45 px held-out RMS).
  - E2, `known1`: **Worse**, +1.67 px [+0.33, +3.13] on a 27.5 px split RMS
    (margin 0.55).
  - E2, `estimated`: Neutral, +0.63 px.
- **`stereo_charuco` camera 2.** Every effect is Neutral. Its split RMS is
  74 px for every arm: the telephoto camera's focal length is weakly
  determined by half the views.
- **`scheimpflug_rig` tile 5.**
  - E1 held-out: **Worse** for both arms, +0.023 px [+0.017, +0.029]
    (`known1`) and +0.024 px [+0.018, +0.030] (`estimated`), on a 1.37 px
    RMS (margin 0.014).
  - E1 homography: Neutral.
  - E2: Neutral.
- **Tiles 0–4 (not decisive, reported only).**
  - E1 held-out is worse by 1.6–2.9 % on tiles 1 and 4, and Neutral on the
    others.
  - E2 is Better on most of them, by 0.12–0.47 px of split RMS.
  - On these tiles the weighting trades a less variable estimate for a
    worse fit to held-out views.
- **E3 on the tiles: not measurable.** On every tile, the single-camera
  seeded solve ends with `tilt_x` at its ±0.10 rad bound around the seed,
  where the covariance is unavailable (`at_bound`). This holds for all arms
  alike.
- **E3 over the 3 decisive private cameras:** `r` is 9.7 [3.7, 20.1] for
  `uniform`, against 13.4 for `known1` and 13.2 for `estimated`.

### Honesty of the stated uncertainty (E3)

**No arm is honest on any real camera.** This is reported per Amendment 1 §4.
- The split-half projection differences are 2–5 times the stated σ:
  `r` = 4.9–26, against 1 on the synthetic truth.
- Only 7–34 % of nodes fall below the 68.3 % contour.
- The per-corner covariance does not change this: `known1` and `estimated`
  match `uniform` to within the intervals.
- So what dominates real calibrations is error the covariance does not
  model: the lens model, the target and detector bias. The random corner
  noise that the covariance describes is the smaller part.
- The hand-eye split differences are closer to their prediction (E4 below):
  `kuka_1` is within it, and `ds8` matches in rotation but is 2.4 times
  over in translation.

### E4 (reported only; medians over 21 splits)

| dataset | quantity | `uniform` | `known1` | `estimated` |
|---|---|---|---|---|
| `kuka_1` | hand-eye split difference | 0.074° / 2.88 mm | 0.077° / 3.00 mm | 0.078° / 3.00 mm |
| `kuka_1` | predicted for the halves | 0.085° / 2.57 mm | 0.085° / 2.57 mm | 0.085° / 2.57 mm |
| `ds8` | hand-eye split difference | 0.266° / 27.6 mm | 0.288° / 32.2 mm | 0.288° / 32.1 mm |
| `ds8` | predicted for the halves | 0.208° / 11.7 mm | 0.207° / 11.6 mm | 0.207° / 11.6 mm |
| `stereo_rig` | camera 1 to rig | 0.179° / 1.55 mm | 0.202° / 1.58 mm | 0.203° / 1.58 mm |
| `stereo_charuco` | camera 1 to rig | 0.148° / 0.70 mm | 0.170° / 0.64 mm | 0.142° / 0.57 mm |

Weighting does not make the downstream results more consistent. It makes
the public ones slightly less so.

### E5

| registry | images | `uniform` median | `known1` | `estimated` |
|---|---|---|---|---|
| public | 166 | 9.59 ms | +1.3 % | +1.6 % |
| private | 174 | 4.14 ms | +3.2 % | +4.9 % |

The private E5 was measured again alone, single-threaded: +3.4 % and +4.8 %.

### Decision applied

- **The default stays off.** The rustdoc of
  `detector.chess_corners.covariance` and the book recommend against
  weighting a calibration by it. On real chessboard and ChArUco images it
  does not improve held-out accuracy, and on some cameras it makes it worse.
  The detector still reports the covariance (`Feature::cov_px`).
- **ADR 0026** records the policy.
- **Follow-ups (backlog):**
  - the real-data honesty gap above;
  - the tiles' tilt at its bound;
  - the ring-grid covariance, which is parked.

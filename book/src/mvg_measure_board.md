# Measuring a Board with a Stereo Calibration

The question this chapter answers: when a calibrated system reports a length as `d ± σ`, is the σ honest? A board of known size makes it checkable. Its squares have a published size, so the distance between any two corners is known, and a measurement made on views the calibration never saw can be compared with it.

The experiment runs on real images: OpenCV's fisheye stereo test set, 34 synchronised views of a chessboard with 6 × 8 corners and 24.4 mm squares, 1280 × 800 pixels, two fisheye cameras (the sample `opencv_fisheye_stereo`, pinned at `opencv_extra` revision `ebf9366b3845`). Both examples download it from its publisher on the first run.

```bash
cargo run --release -p vision-calibration --features fetch --example measure_board
python crates/vision-calibration-py/examples/measure_board.py
```

Both take a sample id: `opencv_stereo` runs the same procedure on a pinhole pair ([below](#a-pinhole-pair)).

## Procedure

1. **Fetch and detect.** `dataset::samples::fetch` downloads the sample; the dataset runner detects the board in every image and returns 34 usable rig views, each holding both cameras' detections.
2. **Split.** The even views (17) calibrate; the odd views (17) are held out. The split is by view index, fixed in advance, and each view keeps both cameras' detections.
3. **Calibrate.** The rig-extrinsics workflow runs on the 17 views with the fisheye model (equidistant projection, Kannala-Brandt distortion). `export.calibrated_rig()` is the rig with its covariance.
4. **Triangulate each held-out view.** Every corner both cameras saw is a track of two observations. The pixel noise is what the calibration states for its own detector, `rig.reprojection_noise(camera)`, not a number chosen for the experiment. The 48 corners are triangulated in one call, so their joint covariance carries the calibration's uncertainty.
5. **Measure.** For every pair of corners on one board row or column, 1 to 7 squares apart, `metrology::distance` gives `d ± σ`; the truth is `k × 24.4 mm`. `fit_plane` over the corners gives the plane's `rms`, a planarity check.
6. **Compare.** Per length, the mean error, the RMS error, the mean stated σ and the z-score RMS: the RMS of `(d − truth) / σ`. A z RMS near 1 means the σ describes the error.

The same run is repeated with the calibration's covariance replaced by zero, which leaves the share of the pixel noise alone. The difference between the two is what the calibration's uncertainty adds.

```rust
let rig = export.calibrated_rig()?;
let noise = [rig.reprojection_noise(0)?, rig.reprojection_noise(1)?];
let tracks: Vec<Vec<RigObservation>> = /* one per corner seen by both cameras */;
let set = PointSet::from_triangulation(&triangulate(&rig, Captures::Rig, &tracks, &RunControl::default())?)?;
let length = distance(&set, a, b)?;           // Scalar { value, sigma }
let error = length.value - truth;             // against the board's known size
```

The examples keep the measurement in a few functions and print the table below; the Rust one is tested on a synthetic stereo board without the network.

## The Result

Run on 2026-11-03, library 0.14.0, the sample revision above. 17 calibration views; 14 of the 17 held-out views measured (in three, one camera did not see the whole board). The calibration has a mean reprojection error of 0.280 px and a stated pixel noise σ̂ of 0.217 px in both cameras. Over the 14 views the board's triangulated corners lie 0.43 mm RMS from their best-fit plane. All lengths in millimetres.

| squares | pairs | mean error | RMS error | mean σ | z RMS | mean σ, observations only | z RMS, observations only |
|---|---|---|---|---|---|---|---|
| 1 | 1146 | +0.012 | 0.216 | 0.333 | 0.64 | 0.329 | 0.65 |
| 2 | 950 | +0.005 | 0.281 | 0.343 | 0.74 | 0.332 | 0.78 |
| 3 | 754 | −0.034 | 0.370 | 0.362 | 0.80 | 0.341 | 0.87 |
| 4 | 558 | −0.025 | 0.454 | 0.388 | 0.85 | 0.357 | 0.95 |
| 5 | 362 | −0.033 | 0.556 | 0.419 | 0.98 | 0.378 | 1.11 |
| 6 | 167 | −0.080 | 0.633 | 0.457 | 0.98 | 0.402 | 1.13 |
| 7 | 83 | −0.163 | 0.765 | 0.507 | 0.98 | 0.434 | 1.17 |

A 7-square length is 170.8 mm: the RMS error of 0.77 mm is 0.45 %, and the stated σ 0.30 %.

## What the z-Scores Say

**The σ is not tight everywhere, and it is not too small anywhere.** The z RMS runs from 0.64 for a single square to 0.98 for five squares and more. No length has a z above 1. The error therefore stays inside the stated σ on average, and the σ is conservative for short lengths.

**With and without the calibration's share.** The calibration adds 1 % to the σ of a one-square length and 17 % to that of a seven-square length: the further apart the corners, the more of their distance error is the calibration's (scale and baseline), which both corners share. The observations-only σ is too small for the long lengths (z 1.11 to 1.17); with the calibration's share the z falls to 0.98. For short lengths it makes no practical difference. So the share belongs in the σ of a measurement, and it matters where the measurement is long.

**The trend with length.** The RMS error grows 3.5 times from one to seven squares while the stated σ grows 1.5 times. Two mechanisms are consistent with this data; the run does not isolate either:

- *Short lengths are over-stated.* The σ treats the 0.217 px level as independent noise on every corner. If neighbouring corners of a real detection share errors (the same blur, illumination or lens-model residual), the common part cancels in their distance, and the distance is better than independent noise says. The effect fades as corners move apart.
- *Long lengths pick up a per-view scale error.* A per-view breakdown of the same measurements (not printed by the examples) gives the mean relative error of each held-out view's seven-square lengths: within ±0.3 % in 13 of 14 views, and −1.5 % in one. The one-square lengths of a view have the same sign in 12 of 14 views and a similar size (−1.0 % in that outlying view): the error is a scale of the view's whole reconstruction, not noise on single corners. That view alone has a z of 2.4 at seven squares; 11 of the 14 views are below 0.9. Where the view's board sits in the image, or how well the fisheye model fits there, would be the first suspects.

**How reliable is the table.** The pairs of one view are strongly correlated, so the effective sample is the 14 views, not the 4 000 pairs. The z of a long length rests on a few views and moves with one of them: it would be lower without the −1.5 % view. The table says the σ is of the right order and, for long lengths, close; it does not support a statement more precise than that.

## The Calibration's View Scatter

The calibration reports `view_scatter` 12.2: the variance of its shared parameters across the views is 12 times what independent corner noise predicts, measured by leaving out one view at a time, and its covariance is already scaled by it (see [Judging a Result](quality.md)). That scaling is where the calibration's 17 % at seven squares comes from; with the covariance as the corner noise alone predicts, the share would be a small fraction of that and the σ would sit near the observations-only column, z 1.17 at seven squares.

The real-data picture is therefore this: the fisheye model fits these images to a few tenths of a pixel, not to the detector's accuracy, so a σ built from independent corner noise understates what the calibration does not know. The scatter-corrected covariance the library reports closes that gap for long lengths here, and leaves the short ones conservative.

## A Pinhole Pair

The same procedure runs on OpenCV's pinhole stereo sample (`opencv_stereo`, pinned at `opencv` revision `d195a5ad7eb5`): 13 synchronised views, 640 × 480 pixels, of a chessboard with 6 × 9 corners, seen turned by 90 degrees in most views. OpenCV does not publish the printed square size, so the sample declares squares of 1 and every length below is in squares. The cameras are calibrated with the default model, pinhole with Brown-Conrady distortion.

```bash
cargo run --release -p vision-calibration --features fetch --example measure_board -- opencv_stereo
python crates/vision-calibration-py/examples/measure_board.py opencv_stereo
```

Run on 2026-11-03. All 13 views are usable; in two of them one camera's detection misses part of the board and is dropped, so those views carry one camera. 7 views calibrate (5 with both cameras) and 6 are held out, all 6 measured with all 54 corners. The calibration has a mean reprojection error of 0.199 px, a stated pixel noise σ̂ of 0.146 px in both cameras and a view scatter of 4.17. The triangulated corners lie 0.013 squares RMS from their best-fit plane.

| squares | pairs | mean error | RMS error | mean σ | z RMS | mean σ, observations only | z RMS, observations only |
|---|---|---|---|---|---|---|---|
| 1 | 558 | −0.0009 | 0.0096 | 0.0103 | 0.85 | 0.0102 | 0.86 |
| 2 | 468 | −0.0021 | 0.0111 | 0.0107 | 1.00 | 0.0103 | 1.04 |
| 3 | 378 | −0.0043 | 0.0125 | 0.0113 | 1.15 | 0.0107 | 1.22 |
| 4 | 288 | −0.0067 | 0.0153 | 0.0122 | 1.30 | 0.0113 | 1.40 |
| 5 | 198 | −0.0101 | 0.0177 | 0.0133 | 1.41 | 0.0120 | 1.54 |
| 6 | 108 | −0.0138 | 0.0203 | 0.0147 | 1.51 | 0.0130 | 1.69 |
| 7 | 72 | −0.0123 | 0.0202 | 0.0160 | 1.40 | 0.0139 | 1.59 |

Here the σ is honest for short lengths and too small for long ones: z RMS 0.85 at one square, 1.4 to 1.5 from five squares on. The mean error is negative and grows with the length, from −0.0009 squares at one square to −0.0138 at six (−0.23 %): the held-out boards come out slightly small, a scale error of the reconstruction that every corner pair of a view shares, and noise on single corners does not produce it. The calibration's share adds 1 % to the σ of one square and 15 % to that of seven, and lowers the long lengths' z from about 1.6 to 1.4 without closing the gap. Two features of the run are consistent with a scale the calibration states too tightly: it rests on 7 views, two of them seen by one camera, and the rig calibrated from all 13 views has a baseline of 3.327 ± 0.005 squares against 3.338 from these 7, a difference of about two of the 13-view σ. Six held-out views are a small sample, so the table supports the order of the σ, not its second digit. Per view, the shortfall is one board's: five of the six views have a z RMS of at most 1.2 at every length, and one (view 7 of the sample) comes out 0.5 % small with a z RMS of 2.9 at six squares.

The regression suite repeats this measurement from the sample's committed corner observations on every change and holds its numbers, per view and per length, against a stored reference (see [Regression Suite](regression_suite.md#measuring-a-board)); the table above is its total.

## What to Take from It

- State the pixel noise with `rig.reprojection_noise`, and pass the whole set of points of a measurement through one `triangulate` call; the joint covariance is what makes a length's σ right.
- Check a calibration on held-out views of a known object before relying on its σ; the two examples are templates for it.
- Expect conservative σ for short lengths on real images and a few views that fall outside it: a per-view scale error is not a noise a σ on pixels can describe.

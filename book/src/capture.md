# Capturing Calibration Images

A calibration is only as good as the views it sees. The solver does not
tell a good capture from a poor one by the residuals, which can be small
either way; the views decide how well the focal length, the principal point
and the distortion are determined. This chapter measures what matters, so a
capture can be planned before the first image is taken.

The short version, for one camera and a planar board:

- **15 to 30 views**, each with the whole board in the image.
- **Tilt the board 15–30°** out of the fronto-parallel pose, and **turn the
  tilt direction** from view to view: about the board's horizontal axis,
  then its vertical axis, both ways.
- **Make the board large in the image**: a third to a half of the image
  width.
- **Move the board over the whole image**, into every corner, not only the
  centre.

A rig needs this for every camera, plus views that two cameras see at once.
A robot moves the camera instead of the board, so the same rules apply to
the gripper's poses, rotated about at least two axes
([Choose Your Workflow](choose_workflow.md)).

## What each rule buys

The numbers below come from the `capture_planning` example, which you can
run and change:

```text
cargo run --release -p vision-calibration --example capture_planning
```

It simulates one camera (1280 × 960 pixels, fx = 1000 px, barrel
distortion) and a chessboard of 30 mm squares with 0.2 px corner noise, and
calibrates each capture plan. The baseline is 16 views of a board with
12 × 9 interior corners, 0.65 m away, centred on each image quadrant in turn
and tilted by 20°. Each table changes one thing. A view whose board leaves
the image is dropped, as a chessboard detector would drop it ("used").

The columns are the stated standard deviations of the focal length, the
principal point and the first radial coefficient, and the projection σ
([Projection Uncertainty](uncertainty_projection.md)): how far, in pixels,
the calibration's uncertainty moves where a ray lands, at the image centre
and at its worst anywhere in the image. The quality check warns when the
worst exceeds 1 px ([Judging a Result](quality.md)).

### Number of views

| Views | σ(fx) px | σ(cx) px | σ(k1) | σ centre px | σ max px |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.31 | 1.44 | 0.0010 | 1.45 | 3.09 |
| 8 | 1.24 | 0.76 | 0.0007 | 0.80 | 1.49 |
| 16 | 0.86 | 0.50 | 0.0005 | 0.50 | 0.92 |
| 32 | 0.68 | 0.40 | 0.0004 | 0.40 | 0.73 |

The uncertainty falls roughly as one over the square root of the number of
views: doubling the views gains about 30 %. More views are cheap insurance,
but they do not make up for a poor tilt or a small board.

### Tilt

| Tilt | Used | σ(fx) px | σ(cx) px | σ max px |
|---|---:|---:|---:|---:|
| 0° | 16 | fails | | |
| 3° | 16 | 36.8 | 1.10 | 29.0 |
| 6° | 16 | 8.44 | 0.77 | 6.15 |
| 11° | 16 | 2.18 | 0.56 | 1.71 |
| 20° | 16 | 0.86 | 0.50 | 0.92 |
| 29° | 12 | 0.63 | 0.62 | 0.95 |
| 40° | 4 | 12.99 | 1.19 | 12.07 |

The focal length is measured by perspective: how the board's far edge
shrinks against its near edge. A board facing the camera squarely shows
none, and its distance and the focal length trade off exactly, so the
initialization fails (see [When the views cannot work](#when-the-views-cannot-work)).
A few degrees leave the focal length tens of pixels
uncertain. From about 15° on it is well determined, and more tilt helps
little. Past 30° a large board no longer fits in the image when it sits
near a corner, and views are lost: here only 4 of 16 survive at 40°.

### Tilt direction

| Eight views tilted | σ(fx) px | σ(cx) px | σ max px |
|---|---:|---:|---:|
| all the same way | fails | | |
| about one axis, both ways | 95.9 | 0.38 | 282 |
| about both axes, both ways | 1.24 | 0.76 | 1.49 |

Views tilted the same way are parallel planes and tell the camera nothing
more than one of them; the initialization fails. Tilting about the board's
horizontal axis, both ways, gives two board orientations: they determine
the vertical focal length and leave the horizontal one almost free. Turn
the board about both axes.

### Board size in the image

| Board | Width in the image | σ(fx) px | σ(cx) px | σ max px |
|---|---:|---:|---:|---:|
| 6 × 5 corners | 18 % | 5.64 | 3.19 | 6.39 |
| 9 × 7 corners | 29 % | 1.86 | 1.09 | 2.05 |
| 12 × 9 corners | 40 % | 0.86 | 0.50 | 0.92 |

A larger board in the image has more corners and, more importantly, more
perspective across it. Going from a fifth to two fifths of the image width
cuts every σ by a factor of six or seven. Bring the board closer, or use a
larger board, until it fills a third to a half of the width; the limit is
that it still fits whole in the image when tilted near a corner.

### Placement

| Board placed | σ(fx) px | σ(k1) | σ centre px | σ max px |
|---|---:|---:|---:|---:|
| centre only | 0.79 | 0.0037 | 0.78 | 14.4 |
| each quadrant | 0.86 | 0.0005 | 0.50 | 0.92 |

Distortion grows toward the image corners. Views that keep the board in the
centre determine the focal length as well as any, but leave the distortion
coefficients to be extrapolated to the corners, where the projection σ
reaches 14 px. Moving the board into each quadrant, its outer corners near
the image corners, brings the worst σ under a pixel.

## A rig needs its joint error too

The tables above judge each parameter by its own σ. A rig adds parameters
that trade off with the cameras' own: one camera's rotation in the rig and
the two principal points all move the image of a board that sits in the
middle of both images the same way. Views that leave them to trade off
still give every parameter a σ that holds, but the error along the
combination the views do pin down (here camera 1's yaw against the
difference of the principal points) comes from larger errors that cancel,
and through the second order those still reach it. Anything that uses the
whole covariance at once — a calibration prior in
[bundle adjustment](mvg_bundle_adjust.md), a test of two calibrations
against each other — then trusts that combination several times too much.

The result's uncertainty predicts this (`second_order`, see
[Reading the Covariance](uncertainty_reading.md)), and the quality report's
`covariance_linearity` check warns above 1.2. For a synthetic stereo pair
(eight views of an 8 × 6 board with 30 mm squares at 0.6–0.8 m, 0.3 px
noise), over 300 calibrations each:

| Views | Predicted worst scatter / σ | Measured | NEES per parameter (measured) |
|---|---:|---:|---:|
| centred, tilted up to 0.2 rad | 2.83 | 3.00 | 1.64 |
| centred, tilted up to 0.3 rad | 1.87 | 2.05 | 1.27 |
| centred, tilted up to 0.5 rad | 1.45 | 1.66 | 1.15 |
| off centre by 12 cm and 9 cm, tilted up to 0.2 rad | 1.55 | 1.56 | 1.10 |
| off centre, tilted up to 0.5 rad | 1.06 | — | 1.02 |

Tilt and placement each help, and only both together bring the
combination within its σ (the last row's measured worst is below what 300
calibrations resolve). The effect grows with the noise variance: the first
row at 0.1 px predicts 1.35, at 0.03 px 1.04. So for a rig, put the board
in the image corners as well as tilting it, in views both cameras see.

## Check while you capture

You do not have to trust the rules: calibrate after the first ten or so
views and read the result. The quality verdict flags poor coverage, a
large projection σ and, for a rig, a joint error beyond the covariance,
and the projection map shows where in the image the calibration is weak:

```rust
use vision_calibration::analysis::{QualityThresholds, assess};
use vision_calibration::common::CalibratedCameras;

let export = session.export()?;
let report = assess(&export, &[[1280, 960]], &QualityThresholds::default())?;
println!("{:?}", report.verdict);
let map = export.projection_uncertainty(0, [1280, 960])?;
println!("σ centre {:.2} px, worst {:.2} px", map.center, map.max);
```

```python
report = vc.assess(result, image_sizes=[(1280, 960)])
print(report.verdict)
p = vc.projection_uncertainty(result, (1280, 960), camera=0)
print(f"σ centre {p.center:.2f} px, worst {p.max:.2f} px")
```

For a single camera, `rank_candidates` then says which board pose to take
next: it predicts, without a re-solve, how much each candidate view would
reduce the worst projection σ ([Planning the Next View](uncertainty_next_view.md)).

## When the views cannot work

When the board planes are parallel in every view, fronto-parallel or tilted
the same way throughout, no calibration can recover the focal length, and
the initialization stops before any solve with an error that says so and
names the fix:

```text
the 16 views do not determine the camera intrinsics: ...; tilt the board
about both of its axes, by 15° to 30°, from view to view
```

The initialization measures how far the board orientations spread across
the views, as seen through a camera whose focal length is the image extent
the corners cover, and rejects a spread under 1°. A spread of a few degrees
still initializes, with the large σ the tables above show. With strong lens
distortion, parallel boards no longer look parallel, and the same error
comes from Zhang's method finding no camera that fits the views.

In Rust it is a typed error to match on; in Python it is a `ValueError`
with the same message:

```rust
use vision_calibration::{Error, init};

match run_calibration(&mut session) {
    Err(Error::Linear(init::Error::DegenerateViews { .. })) => {
        println!("recapture: tilt the board about both of its axes");
    }
    other => other?,
}
```

## The image itself

The geometry above assumes every corner is found to a fraction of a pixel.
That needs:

- **Focus and sharpness**: the board in focus over its whole tilted extent,
  with an exposure short enough that it does not blur when held by hand.
- **Even lighting** without glare on the board; a matte print.
- **A flat, rigid board**, its square size measured on the print (the
  calibration's scale comes from it), not taken from the file.
- **Fixed camera settings**: focus, zoom and resolution unchanged through
  the capture and afterwards. A refocus changes the intrinsics.

The detection noise scales every σ above in proportion: a detector that
finds corners to 0.1 px instead of 0.2 px halves them.

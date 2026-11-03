//! How the capture shapes a planar calibration's uncertainty.
//!
//! One camera (1280 × 960, fx = fy = 1000 px, barrel distortion) and a
//! chessboard of 30 mm squares. The baseline capture is 16 views of a board
//! with 12 × 9 interior corners, 0.65 m away (it spans 40 % of the image
//! width), centred on each image quadrant in turn and tilted by 0.35 rad
//! (20°) about +x, +y, −x or −y. Each table changes one thing: the number of
//! views, the tilt, the tilt direction, the board's size in the image, or
//! where in the image the board goes.
//!
//! Every plan is projected with 0.2 px corner noise and calibrated. A view
//! whose board leaves the image is dropped, as a chessboard detector would
//! drop it. The table reports the stated σ of fx, cx and k1, and the
//! projection σ at the image centre and its largest over the image. A plan
//! whose board planes are all parallel (fronto-parallel, or tilted the same
//! way in every view) cannot determine the intrinsics, and its
//! initialization fails with an error that says to tilt the board about
//! both of its axes.
//!
//! Run with: `cargo run --release -p vision-calibration --example capture_planning`

use anyhow::Result;
use vision_calibration::Error;
use vision_calibration::analysis::{CandidateGrid, candidate_poses};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    BrownConrady5, CameraParams, FxFyCxCy, Iso3, PinholeCamera, PlanarDataset, Pt3, View,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration::init;
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsProblem, run_calibration};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::planar;

const IMAGE: [u32; 2] = [1280, 960];
const FOCAL_PX: f64 = 1000.0;
const SQUARE_M: f64 = 0.03;
const DISTANCE_M: f64 = 0.65;
const TILT_RAD: f64 = 0.35;
const VIEWS: usize = 16;
/// Uniform noise of ±0.35 px per axis: σ ≈ 0.2 px.
const NOISE_MAX_PX: f64 = 0.35;

/// Candidate index offsets of the tilted variants of a region (see
/// `CandidateGrid`): +x, +y, −x, −y.
const TILTS: [usize; 4] = [1, 3, 2, 4];

fn main() -> Result<()> {
    let truth = make_pinhole_camera(
        FxFyCxCy {
            fx: FOCAL_PX,
            fy: FOCAL_PX,
            cx: 640.0,
            cy: 480.0,
        },
        BrownConrady5 {
            k1: -0.15,
            k2: 0.05,
            ..BrownConrady5::default()
        },
    );
    let camera = pinhole_camera_params(&truth);
    let board = planar::grid_points(12, 9, SQUARE_M);
    let study = Study {
        truth: &truth,
        camera: &camera,
    };

    println!("Number of views:");
    header();
    for n in [4, 6, 8, 12, 16, 24, 32] {
        let plan = study.quadrants(&board, n, TILT_RAD)?;
        study.row(&format!("{n} views"), &board, &plan)?;
    }

    println!("\nTilt ({VIEWS} views):");
    header();
    for tilt in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7] {
        let plan = study.quadrants(&board, VIEWS, tilt)?;
        let label = format!("{tilt:.2} rad ({:.0}°)", tilt.to_degrees());
        study.row(&label, &board, &plan)?;
    }

    println!("\nTilt direction (8 views):");
    header();
    let poses = study.grid(&board, DISTANCE_M, TILT_RAD, 2)?;
    let all_one_way: Vec<Iso3> = (0..8).map(|k| poses[(k % 4) * 5 + 1]).collect();
    study.row("all about +x", &board, &all_one_way)?;
    let one_axis: Vec<Iso3> = (0..8).map(|k| poses[(k % 4) * 5 + 1 + k / 4]).collect();
    study.row("about ±x", &board, &one_axis)?;
    let both_axes = study.quadrants(&board, 8, TILT_RAD)?;
    study.row("about ±x and ±y", &board, &both_axes)?;

    println!("\nBoard size in the image ({VIEWS} views):");
    header();
    for (nx, ny) in [(6, 5), (9, 7), (12, 9)] {
        let sized = planar::grid_points(nx, ny, SQUARE_M);
        let share = (nx - 1) as f64 * SQUARE_M * FOCAL_PX / DISTANCE_M / f64::from(IMAGE[0]);
        let label = format!("{nx}×{ny} ({:.0} % wide)", 100.0 * share);
        let plan = study.quadrants(&sized, VIEWS, TILT_RAD)?;
        study.row(&label, &sized, &plan)?;
    }

    println!("\nPlacement ({VIEWS} views):");
    header();
    let mut centre = Vec::new();
    for d in [0.6, 0.65, 0.7, 0.75] {
        let poses = study.grid(&board, d, TILT_RAD, 1)?;
        centre.extend(TILTS.iter().map(|&t| poses[t]));
    }
    study.row("centre only", &board, &centre)?;
    let quadrants = study.quadrants(&board, VIEWS, TILT_RAD)?;
    study.row("each quadrant", &board, &quadrants)?;
    Ok(())
}

struct Study<'a> {
    truth: &'a PinholeCamera,
    camera: &'a CameraParams,
}

impl Study<'_> {
    /// The candidate poses of a `regions × regions` grid at `distance_m`.
    fn grid(&self, board: &[Pt3], distance_m: f64, tilt: f64, regions: usize) -> Result<Vec<Iso3>> {
        let grid = CandidateGrid::new(distance_m, tilt, [regions, regions]);
        Ok(candidate_poses(self.camera, board, IMAGE, &grid)?)
    }

    /// `n` views centred on the four image quadrants in turn, the tilt
    /// direction turning from view to view; views beyond the first 16 are
    /// 0.1 m further away. Without tilt, successive rounds step 5 cm back
    /// instead.
    fn quadrants(&self, board: &[Pt3], n: usize, tilt: f64) -> Result<Vec<Iso3>> {
        let near = self.grid(board, DISTANCE_M, tilt, 2)?;
        let far = self.grid(board, DISTANCE_M + 0.1, tilt, 2)?;
        Ok((0..n)
            .map(|k| {
                let poses = if (k / 16) % 2 == 0 { &near } else { &far };
                let region = k % 4;
                if tilt == 0.0 {
                    let pose = poses[region];
                    let back = pose.translation.vector.normalize() * 0.05 * (k / 4) as f64;
                    Iso3::from_parts((pose.translation.vector + back).into(), pose.rotation)
                } else {
                    poses[region * 5 + TILTS[(k + k / 4) % 4]]
                }
            })
            .collect())
    }

    /// Calibrate `plan` and print one table row.
    fn row(&self, label: &str, board: &[Pt3], plan: &[Iso3]) -> Result<()> {
        let noise = UniformPixelNoise {
            seed: 7,
            max_abs_px: NOISE_MAX_PX,
        };
        let views = planar::project_views_noisy(self.truth, board, plan, &noise)?;
        let [width, height] = IMAGE.map(f64::from);
        let inside: Vec<_> = views
            .into_iter()
            .filter(|view| {
                view.points_2d
                    .iter()
                    .all(|p| (0.0..width).contains(&p.x) && (0.0..height).contains(&p.y))
            })
            .map(View::without_meta)
            .collect();
        let used = inside.len();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        let outcome = PlanarDataset::new(inside)
            .map_err(Error::from)
            .and_then(|dataset| session.set_input(dataset))
            .and_then(|()| run_calibration(&mut session))
            .and_then(|()| session.export());
        let export = match outcome {
            Ok(export) => export,
            // Parallel board planes: the views cannot determine the
            // intrinsics, and the error says how to fix the capture.
            Err(Error::Linear(init::Error::DegenerateViews { .. })) => {
                println!("  {label:<20} {used:>5}   fails: tilt the board about both of its axes");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let Some(covariance) = export.parameter_uncertainty().covariance() else {
            let uncertainty = export.parameter_uncertainty();
            println!("  {label:<20} {used:>5}   no covariance: {uncertainty:?}");
            return Ok(());
        };
        let sigma = |label: &str| covariance.std_dev(label).unwrap_or(f64::NAN);
        let projection = export.projection_uncertainty(0, IMAGE)?;
        println!(
            "  {label:<20} {used:>5} {:>9.3} {:>9.3} {:>9.5} {:>9.3} {:>9.3}",
            sigma("fx"),
            sigma("cx"),
            sigma("k1"),
            projection.center,
            projection.max
        );
        Ok(())
    }
}

fn header() {
    println!(
        "  {:<20} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "plan", "used", "σ(fx) px", "σ(cx) px", "σ(k1)", "σ centre", "σ max"
    );
}

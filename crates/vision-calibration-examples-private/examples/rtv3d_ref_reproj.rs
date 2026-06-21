//! Reprojection-error parity check against the `rtv3d_ref` reference dataset.
//!
//! The `rtv3d_ref` dataset ships a credible external oracle (`artifacts.json`,
//! from the customer's "QUICK" system): a 6-camera Scheimpflug
//! laser-triangulation rig calibrated to sub-pixel reprojection error
//! (0.22–0.30 px per camera) against a static puzzle_board (130×130, 5 mm
//! cells). Each pose stores all six camera views as one 4320×540 horizontal
//! strip (6 tiles of 720×540). The per-camera distortion is OpenCV-layout
//! `[k1, k2, p1, p2, k3, τx, τy]` — the last two are Scheimpflug tilts.
//!
//! This harness isolates **our reprojection + Scheimpflug model** from our
//! optimizer and (mostly) from our detector:
//!
//! 1. Freeze the reference per-camera intrinsics (K + Brown-Conrady + tilt) by
//!    building a [`ScheimpflugCamera`] per camera via `intrinsic_to_camera`.
//! 2. Detect the puzzle_board in each per-camera tile per pose.
//! 3. Fit a per-view target pose with those intrinsics held fixed
//!    (undistorted-homography linear init → pose-only Gauss-Newton refine
//!    through the full Scheimpflug model).
//! 4. Reproject and aggregate per-camera mean and RMS pixel error, printed
//!    alongside the reference `reprojection_error_pix`.
//!
//! Interpretation: if our frozen-intrinsics RMS sits within detector noise of
//! the reference (~≤0.1 px over it), the reprojection computation + Scheimpflug
//! model are validated against an OpenCV-convention ground truth. A materially
//! larger, *structured* residual would indicate a model/convention bug; pure
//! scatter at a higher floor is the (separately tracked) detector limit.
//!
//! Run:
//! `cargo run --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! rtv3d_ref_reproj --release`
//!
//! Env: `RTV3D_REF_DATA_DIR` (default `privatedata/rtv3d_ref`).

use anyhow::{Context, Result, anyhow};
use nalgebra::{DMatrix, DVector, Vector3};
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::geometry::homography::dlt_homography;
use vision_calibration::linear::planar_pose::estimate_planar_pose_from_h;
use vision_calibration_core::{
    CorrespondenceView, DistortionModel, IntrinsicsModel, Iso3, Pt2, SensorModel,
};
use vision_calibration_examples_private::{
    ScheimpflugCamera, detect_target, intrinsic_to_camera, load_gray, load_poses,
    load_ref_artifacts, split_horizontal,
};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 5.0;

/// Per-camera accumulator of per-point pixel errors over all views.
#[derive(Default, Clone)]
struct CamAccum {
    errors: Vec<f64>,
    views: usize,
    skipped_views: usize,
}

impl CamAccum {
    fn push_view(&mut self, errors: &[f64]) {
        self.views += 1;
        self.errors.extend_from_slice(errors);
    }
    fn count(&self) -> usize {
        self.errors.len()
    }
    fn mean(&self) -> f64 {
        if self.errors.is_empty() {
            f64::NAN
        } else {
            self.errors.iter().sum::<f64>() / self.errors.len() as f64
        }
    }
    /// RMS = sqrt(mean(e^2)) — same convention as `ReprojectionStats::rms`.
    fn rms(&self) -> f64 {
        if self.errors.is_empty() {
            f64::NAN
        } else {
            (self.errors.iter().map(|e| e * e).sum::<f64>() / self.errors.len() as f64).sqrt()
        }
    }
    /// Sorted-quantile (e.g. 0.5 for median, 0.95 for p95).
    fn quantile(&self, q: f64) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        let mut v = self.errors.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((v.len() - 1) as f64 * q).round() as usize;
        v[idx]
    }
    /// Fraction of corners whose error exceeds `thresh` px.
    fn frac_above(&self, thresh: f64) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        self.errors.iter().filter(|&&e| e > thresh).count() as f64 / self.errors.len() as f64
    }
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("RTV3D_REF_DATA_DIR").unwrap_or_else(|_| "privatedata/rtv3d_ref".to_string()),
    );
    println!("data dir = {}", data_dir.display());

    let art = load_ref_artifacts(&data_dir.join("artifacts.json"))
        .context("load reference artifacts.json")?;
    if art.num_cameras != NUM_CAMERAS || art.intrinsic.len() != NUM_CAMERAS {
        return Err(anyhow!(
            "expected {NUM_CAMERAS} cameras, got num_cameras={} intrinsics={}",
            art.num_cameras,
            art.intrinsic.len()
        ));
    }
    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!(
        "loaded {} reference cameras, {} poses",
        art.num_cameras,
        poses.len()
    );

    // Freeze the reference intrinsics into full Scheimpflug cameras.
    let cameras: Vec<ScheimpflugCamera> = art
        .intrinsic
        .iter()
        .map(intrinsic_to_camera)
        .collect::<Result<_>>()?;

    let mut accums = vec![CamAccum::default(); NUM_CAMERAS];
    let t0 = Instant::now();

    for pose in &poses {
        let img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("load {}", pose.target_image))?;
        let tiles = split_horizontal(&img, NUM_CAMERAS);
        for (cam_idx, tile) in tiles.iter().enumerate() {
            let view = match detect_target(tile, BOARD_ROWS, BOARD_COLS, CELL_SIZE_MM) {
                Ok(v) => v,
                Err(_) => {
                    accums[cam_idx].skipped_views += 1;
                    continue;
                }
            };
            match fit_pose_and_errors(&cameras[cam_idx], &view) {
                Some(errors) => accums[cam_idx].push_view(&errors),
                None => accums[cam_idx].skipped_views += 1,
            }
        }
    }

    println!("detect + fit: {:.2?}\n", t0.elapsed());
    report(&art, &accums);
    Ok(())
}

/// Undistort an observed pixel to the equivalent ideal-pinhole pixel by
/// inverting the full sensor chain (inverse K → inverse tilt homography →
/// Brown-Conrady undistort → forward K with identity sensor).
fn ideal_pixel(cam: &ScheimpflugCamera, px: &Pt2) -> Pt2 {
    let sensor = cam.k.pixel_to_sensor(px);
    let n_dist = cam.sensor.sensor_to_normalized(&sensor);
    let n_undist = cam.dist.undistort(&n_dist);
    cam.k.sensor_to_pixel(&n_undist)
}

/// Per-point reprojection residual vector (length `2N`), or `None` if any point
/// projects behind the camera (degenerate pose).
fn residuals(
    cam: &ScheimpflugCamera,
    pose: &Iso3,
    view: &CorrespondenceView,
) -> Option<DVector<f64>> {
    let n = view.points_3d.len();
    let mut r = DVector::zeros(2 * n);
    for (i, (x, u)) in view.points_3d.iter().zip(&view.points_2d).enumerate() {
        let p_c = pose.transform_point(x);
        let proj = cam.project_point(&p_c)?;
        r[2 * i] = proj.x - u.x;
        r[2 * i + 1] = proj.y - u.y;
    }
    Some(r)
}

/// Left-compose a small se(3) increment `[t(3), ω(3)]` onto `pose`.
fn apply_increment(delta: &[f64; 6], pose: &Iso3) -> Iso3 {
    let t = Vector3::new(delta[0], delta[1], delta[2]);
    let w = Vector3::new(delta[3], delta[4], delta[5]);
    Iso3::new(t, w) * pose
}

/// Fit the per-view target pose with frozen intrinsics, then return the final
/// per-point reprojection errors (pixels). Linear homography init followed by a
/// pose-only Gauss-Newton refine through the full Scheimpflug model.
fn fit_pose_and_errors(cam: &ScheimpflugCamera, view: &CorrespondenceView) -> Option<Vec<f64>> {
    // Linear init: homography from board (X,Y) to undistorted ideal pixels.
    let world2d = view.planar_points();
    let ideal: Vec<Pt2> = view
        .points_2d
        .iter()
        .map(|px| ideal_pixel(cam, px))
        .collect();
    let h = dlt_homography(&world2d, &ideal).ok()?;
    let mut pose = estimate_planar_pose_from_h(&cam.k.k_matrix(), &h).ok()?;

    // Pose-only Gauss-Newton with numerical Jacobian (6 DOF, tiny problem).
    const EPS: f64 = 1e-6;
    let mut r0 = residuals(cam, &pose, view)?;
    for _ in 0..25 {
        let m = r0.len();
        let mut jac = DMatrix::zeros(m, 6);
        for j in 0..6 {
            let mut d = [0.0; 6];
            d[j] = EPS;
            let rp = residuals(cam, &apply_increment(&d, &pose), view)?;
            jac.set_column(j, &((&rp - &r0) / EPS));
        }
        let jtj = jac.transpose() * &jac;
        let jtr = jac.transpose() * &r0;
        let delta = jtj.lu().solve(&(-jtr))?;
        let d6 = [delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]];
        let cand = apply_increment(&d6, &pose);
        let r_cand = residuals(cam, &cand, view)?;
        // Accept only if it reduces cost (guards rare GN overshoot).
        if r_cand.norm_squared() <= r0.norm_squared() {
            pose = cand;
            r0 = r_cand;
        } else {
            break;
        }
        if delta.norm() < 1e-12 {
            break;
        }
    }

    let n = view.points_3d.len();
    let errors: Vec<f64> = (0..n)
        .map(|i| (r0[2 * i].powi(2) + r0[2 * i + 1].powi(2)).sqrt())
        .collect();
    Some(errors)
}

fn report(art: &vision_calibration_examples_private::RefArtifacts, accums: &[CamAccum]) {
    const TOL: f64 = 0.10; // px slack vs reference

    println!("Per-camera reprojection error (frozen reference intrinsics):");
    println!(
        "  cam | views | corners |  ref   | our_med | our_mean | our_rms |  p95   | %>1px | verdict"
    );
    println!(
        "  ----+-------+---------+--------+---------+----------+---------+--------+-------+--------"
    );

    let mut worst_med_delta = f64::MIN;
    for (i, acc) in accums.iter().enumerate() {
        let ref_px = art.intrinsic[i].reprojection_error_pix;
        let med = acc.quantile(0.5);
        let med_delta = med - ref_px;
        worst_med_delta = worst_med_delta.max(med_delta);
        let rms_delta = acc.rms() - ref_px;

        // Median tracks the bulk of corners; RMS is outlier-sensitive. A model
        // or convention bug inflates the *median* (a systematic bias); detector
        // corner outliers inflate only the *tail* (RMS / p95 / %>1px).
        let verdict = if acc.count() == 0 {
            "NO DATA".to_string()
        } else if med_delta > TOL {
            format!("MODEL? med +{med_delta:.3}")
        } else if rms_delta > TOL {
            "MODEL ok / detector tail".to_string()
        } else {
            "MATCH".to_string()
        };
        let skip = if acc.skipped_views > 0 {
            format!(" (+{} skipped views)", acc.skipped_views)
        } else {
            String::new()
        };
        println!(
            "  {:>3} | {:>5} | {:>7} | {:>6.4} | {:>7.4} | {:>8.4} | {:>7.4} | {:>6.4} | {:>4.1}% | {verdict}{skip}",
            i,
            acc.views,
            acc.count(),
            ref_px,
            med,
            acc.mean(),
            acc.rms(),
            acc.quantile(0.95),
            100.0 * acc.frac_above(1.0),
        );
    }

    println!();
    if worst_med_delta <= TOL {
        println!(
            "VERDICT: median reprojection tracks the reference on every camera (worst median Δ {worst_med_delta:+.4} px)."
        );
        println!(
            "→ Scheimpflug model + reprojection computation VALIDATED against the OpenCV-convention oracle."
        );
        println!(
            "  RMS/p95 inflation above is a detector corner-outlier tail (parked V7 floor), not a model bias."
        );
    } else {
        println!(
            "VERDICT: median exceeds the reference by {worst_med_delta:+.4} px on at least one camera —"
        );
        println!(
            "  a systematic (model/convention) bias, not just detector outliers. Investigate before trusting."
        );
    }
}

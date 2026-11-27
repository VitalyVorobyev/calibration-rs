//! The projection difference between two calibrations of different camera
//! models from the same synthetic planar observations, on both solver
//! backends.
//!
//! - A Brown-Conrady lens with 0.1 px of noise, calibrated as Brown-Conrady
//!   and as a rational model: both fit, and their projections differ by
//!   less than either differs from the truth, far below the noise.
//! - A lens with thin-prism terms, noise-free, calibrated as thin prism
//!   (exact) and as Brown-Conrady (which lacks the terms): the difference
//!   over the observed region is the Brown-Conrady camera's misfit at fixed
//!   rays. It is several times the fit's reprojection residual (3.4–3.8×
//!   here), because each view's pose absorbs part of the missing field in
//!   the residuals and no pose enters the projection difference.

#![allow(missing_docs)]

use nalgebra::{Matrix2, Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraParams, CameraProject, Compensation, CorrespondenceView,
    DistortionKind, DistortionParams, FitRegion, FxFyCxCy, IdentitySensor, ImageGrid, Iso3,
    Pinhole, PlanarDataset, ProjectionDifference, ProjectionDifferenceOptions, ProjectionParams,
    Pt2, Pt3, SensorParams, ThinPrism, Uncertainty, View, pinhole_camera_params,
    projection_difference,
};
use vision_calibration_optim::{SolverBackend, projection_covariance_at};
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init,
    step_optimize,
};
use vision_calibration_pipeline::session::CalibrationSession;

const SIZE: [u32; 2] = [1280, 960];
const VIEWS: usize = 30;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn intrinsics() -> FxFyCxCy<f64> {
    FxFyCxCy {
        fx: 900.0,
        fy: 900.0,
        cx: 640.0,
        cy: 480.0,
    }
}

fn brown() -> BrownConrady5<f64> {
    BrownConrady5 {
        k1: -0.15,
        k2: 0.05,
        k3: 0.0,
        p1: 0.0004,
        p2: -0.0003,
        iters: 8,
    }
}

/// Brown-Conrady plus the thin prism's `r⁴` terms, about 2 px at the
/// corners.
fn prism() -> ThinPrism<f64> {
    let b = brown();
    ThinPrism {
        k1: b.k1,
        k2: b.k2,
        k3: b.k3,
        p1: b.p1,
        p2: b.p2,
        s1: 0.0,
        s2: 0.004,
        s3: 0.0,
        s4: -0.003,
        iters: 8,
    }
}

/// A 14 × 11 board of 25 mm cells seen from `VIEWS` seeded poses whose
/// centres land across the image, 0.5–0.8 m away, tilted up to 0.4 rad.
fn poses() -> Vec<Iso3> {
    let mut rng = Rng(17);
    let k = intrinsics();
    let centre = (13.0 * 0.025 / 2.0, 10.0 * 0.025 / 2.0);
    (0..VIEWS)
        .map(|_| {
            let r = Rotation3::from_euler_angles(
                rng.range(-0.4, 0.4),
                rng.range(-0.4, 0.4),
                rng.range(-0.6, 0.6),
            );
            let z = rng.range(0.5, 0.8);
            let (u, v) = (rng.range(150.0, 1130.0), rng.range(120.0, 840.0));
            let target = Vector3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z);
            let t = target - r * Vector3::new(centre.0, centre.1, 0.0);
            Iso3::from_parts(Translation3::from(t), r.into())
        })
        .collect()
}

/// The board's points that `camera` projects into the image, with seeded
/// Gaussian noise of `noise` px per axis.
fn dataset<C: CameraProject>(camera: &C, poses: &[Iso3], noise: f64) -> PlanarDataset {
    let board: Vec<Pt3> = (0..11)
        .flat_map(|j| (0..14).map(move |i| Pt3::new(i as f64 * 0.025, j as f64 * 0.025, 0.0)))
        .collect();
    let mut rng = Rng(23);
    let views = poses
        .iter()
        .map(|pose| {
            let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = board
                .iter()
                .filter_map(|pw| {
                    let q = camera.project_camera_point(&pose.transform_point(pw).coords)?;
                    let inside = q.x >= 0.0
                        && q.y >= 0.0
                        && q.x <= f64::from(SIZE[0] - 1)
                        && q.y <= f64::from(SIZE[1] - 1);
                    let n = [rng.normal() * noise, rng.normal() * noise];
                    inside.then(|| (*pw, Pt2::new(q.x + n[0], q.y + n[1])))
                })
                .unzip();
            assert!(p3.len() >= 30, "a view sees {} points", p3.len());
            View::without_meta(CorrespondenceView::new(p3, p2).expect("view"))
        })
        .collect();
    PlanarDataset::new(views).expect("dataset")
}

fn calibrate(
    dataset: &PlanarDataset,
    distortion: DistortionKind,
    backend: SolverBackend,
) -> PlanarIntrinsicsExport {
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model.distortion = distortion;
    config.solver.backend = backend;
    config.solver.max_iters = 200;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_config(config).expect("config");
    session.set_input(dataset.clone()).expect("input");
    step_init(&mut session, None).expect("init");
    step_optimize(&mut session, None).expect("optimize");
    session.export().expect("export")
}

/// The pixels the export's features project to: the region the data cover.
fn covered(export: &PlanarIntrinsicsExport) -> Vec<[f64; 2]> {
    export
        .diagnostics
        .per_feature_residuals
        .target
        .iter()
        .filter_map(|r| r.projected_px)
        .collect()
}

/// The RMS of `|observed − projected|` over the export's features, pixels.
fn reprojection_rms(export: &PlanarIntrinsicsExport) -> f64 {
    let errors: Vec<f64> = export
        .diagnostics
        .per_feature_residuals
        .target
        .iter()
        .filter_map(|r| r.residual_px())
        .map(|[x, y]| x * x + y * y)
        .collect();
    (errors.iter().sum::<f64>() / errors.len() as f64).sqrt()
}

/// The RMS over the fitted nodes of the export's major-axis projection σ.
fn projection_sigma_rms(export: &PlanarIntrinsicsExport, diff: &ProjectionDifference) -> f64 {
    let Uncertainty::Available(covariance) = &export.uncertainty else {
        panic!("no covariance")
    };
    let nodes: Vec<[f64; 2]> = diff
        .grid
        .nodes()
        .enumerate()
        .filter(|(i, _)| diff.fitted[*i])
        .map(|(_, u)| u)
        .collect();
    let sigma = projection_covariance_at(&export.camera, covariance, 0, &nodes)
        .expect("projection covariance");
    let sum: f64 = sigma
        .iter()
        .map(|c| major_sigma(&c.expect("a ray")).powi(2))
        .sum();
    (sum / nodes.len() as f64).sqrt()
}

/// The major-axis σ of a 2×2 covariance.
fn major_sigma(c: &Matrix2<f64>) -> f64 {
    let half = 0.5 * (c[(0, 0)] + c[(1, 1)]);
    (half + (0.5 * (c[(0, 0)] - c[(1, 1)])).hypot(c[(0, 1)])).sqrt()
}

fn compare(a: &CameraParams, b: &CameraParams, region: &[[f64; 2]]) -> ProjectionDifference {
    compare_with(a, b, region, Compensation::Rotation)
}

fn compare_with(
    a: &CameraParams,
    b: &CameraParams,
    region: &[[f64; 2]],
    compensation: Compensation,
) -> ProjectionDifference {
    let grid = ImageGrid::with_columns(SIZE, 64).expect("grid");
    let options =
        ProjectionDifferenceOptions::new(compensation, FitRegion::Covered(region.to_vec()));
    projection_difference(a, b, &grid, &options).expect("projection difference")
}

/// The RMS of `|difference|` over the fitted nodes, each weighted by the
/// number of `pixels` in its block: the difference as the features sample it.
fn feature_weighted_rms(diff: &ProjectionDifference, pixels: &[[f64; 2]]) -> f64 {
    let stride = f64::from(diff.grid.stride());
    let (mut sum, mut weight) = (0.0, 0.0);
    for [x, y] in pixels {
        let column = ((x + 0.5) / stride).floor() as usize;
        let row = ((y + 0.5) / stride).floor() as usize;
        let i = row * diff.grid.columns() + column;
        if diff.fitted[i] {
            sum += diff.difference.dx[i].powi(2) + diff.difference.dy[i].powi(2);
            weight += 1.0;
        }
    }
    (sum / weight).sqrt()
}

fn backends() -> [SolverBackend; 2] {
    [SolverBackend::TinySolver, SolverBackend::Factrs]
}

/// Two adequate models of one noisy capture project alike: their difference
/// over the observed region is below either one's error against the truth,
/// far below the noise of a single feature.
#[test]
fn two_adequate_models_project_alike() {
    let truth = pinhole_camera_params(&Camera::new(Pinhole, brown(), IdentitySensor, intrinsics()));
    let data = dataset(
        &Camera::new(Pinhole, brown(), IdentitySensor, intrinsics()),
        &poses(),
        0.1,
    );
    for backend in backends() {
        let bc5 = calibrate(&data, DistortionKind::BrownConrady5, backend);
        let rational = calibrate(&data, DistortionKind::Rational8, backend);
        let region = covered(&bc5);
        let models = compare(&bc5.camera, &rational.camera, &region);
        let to_truth = compare(&truth, &bc5.camera, &region);
        let sigma = projection_sigma_rms(&bc5, &models);
        let raw = compare_with(&truth, &bc5.camera, &region, Compensation::None);
        eprintln!(
            "{backend:?}: BC5 vs truth without compensation rms {:.4} px",
            raw.rms_fit_px
        );
        let residual = reprojection_rms(&bc5);
        eprintln!(
            "{backend:?}: BC5 vs Rational8 rms {:.4} px (max {:.4}) over {} of {} nodes, \
             rotation {:.1e} rad; BC5 vs truth rms {:.4} px; BC5 projection σ rms {sigma:.4} px; \
             reprojection rms BC5 {residual:.4}, Rational8 {:.4} px",
            models.rms_fit_px,
            models.max_fit_px,
            models.fit_nodes,
            models.valid_nodes,
            models.rotation_angle(),
            to_truth.rms_fit_px,
            reprojection_rms(&rational),
        );
        // Fitted to one noise draw, the two models' errors are correlated:
        // they differ by less than either differs from the truth.
        assert!(models.rms_fit_px < to_truth.rms_fit_px);
        assert!(to_truth.rms_fit_px < 0.5 * residual);
        // The stated projection σ includes the frame-rotation direction
        // (the principal point against the poses) that the compensation
        // removes: it matches the uncompensated error (0.25 against 0.26 px
        // here), which is about 7× the compensated one.
        assert!(raw.rms_fit_px > 3.0 * to_truth.rms_fit_px);
        assert!((0.5..=2.0).contains(&(raw.rms_fit_px / sigma)));
    }
}

/// A model that lacks a term of the lens, against one that has it, on
/// noise-free data: the difference over the observed region is the lacking
/// model's misfit at fixed rays, larger than its reprojection residual,
/// which each view's pose shrinks (by 3.4× with the features' own weights,
/// 3.8× per covered block, on both backends).
#[test]
fn a_missing_term_shows_above_the_fit_residual() {
    let lens = Camera::new(Pinhole, prism(), IdentitySensor, intrinsics());
    let truth = CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::ThinPrism { params: prism() },
        SensorParams::Identity,
        intrinsics(),
    );
    let data = dataset(&lens, &poses(), 0.0);
    for backend in backends() {
        let bc5 = calibrate(&data, DistortionKind::BrownConrady5, backend);
        let full = calibrate(&data, DistortionKind::ThinPrism9, backend);
        let region = covered(&full);
        let models = compare(&full.camera, &bc5.camera, &region);
        let exact = compare(&truth, &full.camera, &region);
        let residual = reprojection_rms(&bc5);
        let ratio = models.rms_fit_px / residual;
        let weighted = feature_weighted_rms(&models, &region);
        eprintln!(
            "{backend:?}: feature-weighted ThinPrism9 vs BC5 rms {weighted:.4} px, ratio {:.2}",
            weighted / residual
        );
        eprintln!(
            "{backend:?}: ThinPrism9 vs BC5 rms {:.4} px (max {:.4}, max over the image {:.3}) \
             over {} nodes; BC5 reprojection rms {residual:.4} px, ratio {ratio:.2}; \
             ThinPrism9 vs truth rms {:.1e} px, ThinPrism9 reprojection rms {:.1e} px",
            models.rms_fit_px,
            models.max_fit_px,
            models.max_valid_px,
            models.fit_nodes,
            exact.rms_fit_px,
            reprojection_rms(&full),
        );
        assert!(exact.rms_fit_px < 1e-3, "{}", exact.rms_fit_px);
        assert!(ratio > 2.0 && weighted / residual > 2.0, "{ratio}");
        assert!(models.rms_fit_px > 1e3 * exact.rms_fit_px);
    }
}

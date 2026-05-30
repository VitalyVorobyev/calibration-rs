//! Hierarchical, multi-level reprojection-error analysis.
//!
//! A single reprojection number per calibration (e.g. "kuka_1 hand-eye =
//! 1.19 px") does not tell us *where* the error comes from: feature detection,
//! the camera model, the rig extrinsics, or the robot / hand-eye chain. This
//! module turns **one** calibration result into a hierarchical report: it
//! re-evaluates the *same* detected corners at several **constraint levels** and
//! reports per-corner residuals plus aggregates. The jump between adjacent
//! levels localizes the error.
//!
//! # Constraint levels
//!
//! - [`ReprojLevel::Intrinsic`] — the "floor". For each `(camera, view)` the
//!   board pose is **free**: solved by PnP from that one image with the *final
//!   calibrated* camera, refined to a per-view reprojection minimum, then
//!   reprojected. This is the best the camera + detector could do if every board
//!   pose were unconstrained.
//! - [`ReprojLevel::RigExtrinsic`] — the board pose is shared across cameras per
//!   view through the calibrated rig extrinsics (`cam_se3_rig * rig_se3_target`).
//!   Only meaningful for multi-camera rigs.
//! - [`ReprojLevel::HandEye`] — the board pose comes from the robot pose composed
//!   with the hand-eye chain (fully constrained). This is the headline number.
//! - [`ReprojLevel::Laser`] — reserved; unused by the builders here.
//!
//! Reading the deltas:
//!
//! - intrinsic ≈ hand-eye → the limit is the camera model / detection.
//! - intrinsic ≪ hand-eye → the robot / hand-eye chain adds the error.
//!
//! # Per-corner records
//!
//! Every level reuses the workspace per-corner record [`TargetFeatureResidual`],
//! so a viewer can drill into a single `(pose, camera, feature)` at any level
//! without a parallel struct. Each record already carries `projected_px`, so a
//! diagnose UI can draw overlay arrows directly from any [`LevelReport`].
//!
//! # Percentile convention
//!
//! [`LevelStats`] percentiles (`median`, `p95`) use **linear interpolation**
//! between order statistics on the ascending-sorted finite errors (the
//! numpy/pandas default): the `q`-quantile lies at fractional rank
//! `q * (n - 1)`. `rms` is `sqrt(mean(error^2))`. All aggregates ignore
//! non-finite / `None` errors.

use serde::{Deserialize, Serialize};

use nalgebra::{Matrix2x6, Matrix6, SVector, Vector2, Vector6};

use vision_calibration_core::{
    CameraProject, FxFyCxCySkew, Iso3, PinholeCamera, Pt2, Pt3, RigDataset, TargetFeatureResidual,
    View,
};
use vision_calibration_linear::homography::HomographySolver;
use vision_calibration_linear::planar_pose::estimate_planar_pose_from_h;
use vision_calibration_linear::pnp::PnpSolver;

use crate::Error;
use crate::planar_intrinsics::PlanarIntrinsicsExport;
use crate::rig_extrinsics::RigExtrinsicsExport;
use crate::single_cam_handeye::{SingleCamHandeyeExport, SingleCamHandeyeView};

// ─────────────────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────────────────

/// Constraint level at which a board pose is determined before reprojection.
///
/// Ordered from least to most constrained: an `Intrinsic`-level pose is free per
/// `(camera, view)`; a `HandEye`-level pose is fully determined by the robot pose
/// and the calibrated hand-eye chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReprojLevel {
    /// Board pose free per `(camera, view)` (PnP + per-view refinement).
    Intrinsic,
    /// Board pose shared across cameras per view via the rig extrinsics.
    RigExtrinsic,
    /// Board pose from the robot pose composed with the hand-eye chain.
    HandEye,
    /// Reserved for laser-plane residuals; unused by the builders here.
    Laser,
}

/// Aggregate statistics over the finite `error_px` values of a residual slice.
///
/// All fields are in pixels except `count`. When `count == 0` every numeric
/// field is `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelStats {
    /// Arithmetic mean of the finite errors.
    pub mean: f64,
    /// Median (50th percentile, linear interpolation).
    pub median: f64,
    /// Root-mean-square: `sqrt(mean(error^2))`.
    pub rms: f64,
    /// 95th percentile (linear interpolation).
    pub p95: f64,
    /// Maximum finite error.
    pub max: f64,
    /// Number of finite errors included.
    pub count: usize,
}

impl LevelStats {
    /// Compute statistics over the finite `error_px` values of a residual slice.
    ///
    /// Non-finite errors and `None` errors are ignored. An empty (or all-`None`)
    /// slice yields the all-zero `LevelStats` with `count == 0`.
    pub fn from_residuals(residuals: &[TargetFeatureResidual]) -> Self {
        let mut errs: Vec<f64> = residuals
            .iter()
            .filter_map(|r| r.error_px)
            .filter(|e| e.is_finite())
            .collect();
        Self::from_errors(&mut errs)
    }

    /// Compute statistics from a mutable error buffer (sorted in place).
    fn from_errors(errs: &mut [f64]) -> Self {
        let count = errs.len();
        if count == 0 {
            return Self {
                mean: 0.0,
                median: 0.0,
                rms: 0.0,
                p95: 0.0,
                max: 0.0,
                count: 0,
            };
        }
        let n = count as f64;
        let sum: f64 = errs.iter().sum();
        let sum_sq: f64 = errs.iter().map(|e| e * e).sum();
        errs.sort_by(|a, b| a.partial_cmp(b).expect("finite errors are comparable"));
        Self {
            mean: sum / n,
            median: percentile_linear(errs, 0.5),
            rms: (sum_sq / n).sqrt(),
            p95: percentile_linear(errs, 0.95),
            max: errs[count - 1],
            count,
        }
    }
}

/// Linear-interpolation percentile of an ascending-sorted, non-empty slice.
///
/// `q` is in `[0, 1]`; the result lies at fractional rank `q * (n - 1)`,
/// interpolating between adjacent order statistics (numpy/pandas default).
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = q * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// One constraint level of a [`ReprojReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelReport {
    /// Which constraint level produced these residuals.
    pub level: ReprojLevel,
    /// Aggregate over all finite residuals of this level.
    pub overall: LevelStats,
    /// Per-camera aggregates, indexed by camera index.
    pub per_camera: Vec<LevelStats>,
    /// Per-view aggregates, indexed by view / snap index, pooled over cameras
    /// and corners.
    pub per_view: Vec<LevelStats>,
    /// Per-corner residual records for this level.
    pub residuals: Vec<TargetFeatureResidual>,
}

impl LevelReport {
    /// Build a [`LevelReport`] from a residual vector, partitioning aggregates by
    /// the `camera` and `pose` indices carried on each record.
    ///
    /// `num_cameras` and `num_views` size the per-camera / per-view vectors so
    /// that empty slots (a camera or view with no finite residuals) still appear
    /// with `count == 0`.
    fn from_residuals(
        level: ReprojLevel,
        residuals: Vec<TargetFeatureResidual>,
        num_cameras: usize,
        num_views: usize,
    ) -> Self {
        let overall = LevelStats::from_residuals(&residuals);

        let mut per_camera_buf: Vec<Vec<f64>> = vec![Vec::new(); num_cameras];
        let mut per_view_buf: Vec<Vec<f64>> = vec![Vec::new(); num_views];
        for r in &residuals {
            let Some(e) = r.error_px.filter(|e| e.is_finite()) else {
                continue;
            };
            if let Some(slot) = per_camera_buf.get_mut(r.camera) {
                slot.push(e);
            }
            if let Some(slot) = per_view_buf.get_mut(r.pose) {
                slot.push(e);
            }
        }
        let per_camera = per_camera_buf
            .iter_mut()
            .map(|b| LevelStats::from_errors(b))
            .collect();
        let per_view = per_view_buf
            .iter_mut()
            .map(|b| LevelStats::from_errors(b))
            .collect();

        Self {
            level,
            overall,
            per_camera,
            per_view,
            residuals,
        }
    }
}

/// A hierarchical reprojection-error report: one [`LevelReport`] per constraint
/// level, plus a single headline number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReprojReport {
    /// The most-constrained level's `overall.mean` (pixels): `HandEye` if
    /// present, else `RigExtrinsic`, else `Intrinsic`. Equals the calibration's
    /// authoritative `mean_reproj_error`.
    pub headline_px: f64,
    /// Levels, ordered least-constrained first.
    pub levels: Vec<LevelReport>,
}

impl ReprojReport {
    /// Assemble a report from an ordered list of levels and set `headline_px`
    /// from the most-constrained available level.
    fn from_levels(levels: Vec<LevelReport>) -> Self {
        let headline_px = levels
            .iter()
            .max_by_key(|l| level_rank(l.level))
            .map(|l| l.overall.mean)
            .unwrap_or(0.0);
        Self {
            headline_px,
            levels,
        }
    }

    /// Borrow the [`LevelReport`] for a given level, if present.
    pub fn level(&self, level: ReprojLevel) -> Option<&LevelReport> {
        self.levels.iter().find(|l| l.level == level)
    }
}

/// Constraint precedence: higher means more constrained.
fn level_rank(level: ReprojLevel) -> u8 {
    match level {
        ReprojLevel::Intrinsic => 0,
        ReprojLevel::Laser => 1,
        ReprojLevel::RigExtrinsic => 2,
        ReprojLevel::HandEye => 3,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public builders
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`ReprojReport`] for a planar-intrinsics calibration.
///
/// Single level (`Intrinsic`): the export already optimized one free pose per
/// view, so the floor *is* the headline. Uses the export's own
/// `per_feature_residuals` when present (they match the calibrated solution),
/// else recomputes the floor from `views`.
pub fn planar_intrinsics_report<M>(
    export: &PlanarIntrinsicsExport,
    views: &[View<M>],
) -> Result<ReprojReport, Error> {
    let cam = &export.params.camera;
    let k = cam.k;
    let residuals = if !export.per_feature_residuals.target.is_empty() {
        export.per_feature_residuals.target.clone()
    } else {
        intrinsic_floor_single_cam(cam, &k, views, 0)
    };
    let num_views = views.len().max(max_pose(&residuals) + 1);
    let level = LevelReport::from_residuals(ReprojLevel::Intrinsic, residuals, 1, num_views);
    Ok(ReprojReport::from_levels(vec![level]))
}

/// Build a [`ReprojReport`] for a single-camera hand-eye calibration.
///
/// Two levels: the `Intrinsic` floor (free per-view pose via PnP) and the
/// `HandEye` level (the export's own residuals, board pose from the robot +
/// hand-eye chain). The gap between them is the error the robot / hand-eye chain
/// adds on top of the camera-only floor.
pub fn single_cam_handeye_report(
    export: &SingleCamHandeyeExport,
    views: &[SingleCamHandeyeView],
) -> Result<ReprojReport, Error> {
    let cam = &export.camera;
    let k = cam.k;

    let intrinsic = intrinsic_floor_handeye(cam, &k, views, 0);
    let intrinsic_level =
        LevelReport::from_residuals(ReprojLevel::Intrinsic, intrinsic, 1, views.len());

    let handeye_residuals = export.per_feature_residuals.target.clone();
    let handeye_views = views.len().max(max_pose(&handeye_residuals) + 1);
    let handeye_level =
        LevelReport::from_residuals(ReprojLevel::HandEye, handeye_residuals, 1, handeye_views);

    Ok(ReprojReport::from_levels(vec![
        intrinsic_level,
        handeye_level,
    ]))
}

/// Build a [`ReprojReport`] for a multi-camera rig-extrinsics calibration.
///
/// Two levels: the `Intrinsic` floor (free per-view pose **per camera**) and the
/// `RigExtrinsic` level (board pose shared across cameras via the rig
/// extrinsics). Uses the export's own residuals for the constrained level when
/// present, else recomputes from the dataset.
pub fn rig_extrinsics_report<M>(
    export: &RigExtrinsicsExport,
    dataset: &RigDataset<M>,
) -> Result<ReprojReport, Error> {
    let ncam = export.cameras.len();
    let mut intrinsic = Vec::new();
    for (cam_idx, cam) in export.cameras.iter().enumerate() {
        let k = cam.k;
        for (view_idx, view) in dataset.views.iter().enumerate() {
            let Some(obs) = view.obs.cameras.get(cam_idx).and_then(|c| c.as_ref()) else {
                continue;
            };
            let Some(pose) = intrinsic_floor_view(cam, &k, &obs.points_3d, &obs.points_2d) else {
                continue;
            };
            push_view_residuals(
                &mut intrinsic,
                cam,
                &pose,
                &obs.points_3d,
                &obs.points_2d,
                view_idx,
                cam_idx,
            );
        }
    }
    let num_views = dataset.views.len();
    let intrinsic_level =
        LevelReport::from_residuals(ReprojLevel::Intrinsic, intrinsic, ncam, num_views);

    let rig_residuals = if !export.per_feature_residuals.target.is_empty() {
        export.per_feature_residuals.target.clone()
    } else {
        rig_constrained_residuals(export, dataset)
    };
    let rig_level = LevelReport::from_residuals(
        ReprojLevel::RigExtrinsic,
        rig_residuals.clone(),
        ncam,
        num_views.max(max_pose(&rig_residuals) + 1),
    );
    Ok(ReprojReport::from_levels(vec![intrinsic_level, rig_level]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Intrinsic floor (the only genuinely new computation)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the per-view board pose `camera_se3_target` (`T_C_W`) for one
/// `(camera, view)`: an EPnP seed (algebraic) refined by a pose-only Gauss-Newton
/// minimization of reprojection error through the *full* calibrated camera model,
/// so the intrinsic level is a genuine per-view reprojection minimum rather than
/// the slightly looser DLT/EPnP solution.
///
/// Returns `None` when there are fewer than four corners or EPnP fails.
fn intrinsic_floor_view(
    camera: &PinholeCamera,
    k: &FxFyCxCySkew<f64>,
    obs_3d: &[Pt3],
    obs_2d: &[Pt2],
) -> Option<Iso3> {
    if obs_3d.len() < 4 || obs_3d.len() != obs_2d.len() {
        return None;
    }
    // Seed the pose. Calibration targets are planar (Z = 0 in the board frame),
    // and for coplanar points EPnP's control-point construction is degenerate —
    // so for a planar target we decompose a plane-induced homography instead
    // (`H = K[r1 r2 t]`). EPnP is kept as the fallback for a rare non-planar
    // target. Either seed is then polished by the pose-only refinement below.
    let max_z = obs_3d.iter().fold(0.0_f64, |m, p| m.max(p.z.abs()));
    let seed = if max_z < 1e-6 {
        let world2d: Vec<Pt2> = obs_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
        let h = HomographySolver::dlt(&world2d, obs_2d).ok()?;
        estimate_planar_pose_from_h(&k.k_matrix(), &h).ok()?
    } else {
        PnpSolver::epnp(obs_3d, obs_2d, k).ok()?
    };
    Some(refine_pose_only(camera, &seed, obs_3d, obs_2d))
}

/// Pose-only Gauss-Newton refinement of `camera_se3_target` (`T_C_W`).
///
/// Minimizes `Σ‖project(camera, pose·p3d) − p2d‖²` over the 6-DOF pose with a
/// numeric Jacobian on the left se(3) tangent (rotation first, then translation),
/// seeded from `seed`. A fixed small iteration count with a step-improvement
/// guard; returns the best pose seen. The EPnP seed is already close, so this
/// only polishes it to the reprojection minimum.
fn refine_pose_only(camera: &PinholeCamera, seed: &Iso3, obs_3d: &[Pt3], obs_2d: &[Pt2]) -> Iso3 {
    const MAX_ITERS: usize = 10;
    const EPS: f64 = 1e-6;

    let cost = |pose: &Iso3| -> f64 {
        let mut s = 0.0;
        for (p3d, p2d) in obs_3d.iter().zip(obs_2d.iter()) {
            let p_cam = pose * p3d;
            if let Some(proj) = camera.project_camera_point(&p_cam.coords) {
                s += (proj.coords - p2d.coords).norm_squared();
            }
        }
        s
    };

    let residual_at = |pose: &Iso3, idx: usize| -> Option<Vector2<f64>> {
        let p_cam = pose * obs_3d[idx];
        camera
            .project_camera_point(&p_cam.coords)
            .map(|proj| proj.coords - obs_2d[idx].coords)
    };

    let mut pose = *seed;
    let mut best_pose = pose;
    let mut best_cost = cost(&pose);

    for _ in 0..MAX_ITERS {
        // Normal equations H = JᵀJ, g = Jᵀr via numeric Jacobian on the left
        // tangent δ ∈ se(3): pose(δ) = exp(δ) * pose.
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();

        for i in 0..obs_3d.len() {
            let Some(r_i) = residual_at(&pose, i) else {
                continue;
            };
            let mut j_i = Matrix2x6::<f64>::zeros();
            let mut ok = true;
            for c in 0..6 {
                let mut delta = Vector6::zeros();
                delta[c] = EPS;
                let perturbed = apply_left_tangent(&pose, &delta);
                let Some(r_p) = residual_at(&perturbed, i) else {
                    ok = false;
                    break;
                };
                let col = (r_p - r_i) / EPS;
                j_i[(0, c)] = col[0];
                j_i[(1, c)] = col[1];
            }
            if !ok {
                continue;
            }
            h += j_i.transpose() * j_i;
            g += j_i.transpose() * r_i;
        }

        let h_damped = h + Matrix6::from_diagonal_element(1e-9);
        let Some(step) = h_damped.lu().solve(&(-g)) else {
            break;
        };
        let candidate = apply_left_tangent(&pose, &step);
        let candidate_cost = cost(&candidate);
        if candidate_cost < best_cost {
            best_cost = candidate_cost;
            best_pose = candidate;
            pose = candidate;
        } else {
            break;
        }
    }

    best_pose
}

/// Apply a left se(3) tangent increment to a pose: `exp(δ) * pose`.
///
/// `delta` is `[ωx, ωy, ωz, vx, vy, vz]` (rotation first).
fn apply_left_tangent(pose: &Iso3, delta: &SVector<f64, 6>) -> Iso3 {
    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    let omega = Vector3::new(delta[0], delta[1], delta[2]);
    let v = Vector3::new(delta[3], delta[4], delta[5]);
    let rot = UnitQuaternion::from_scaled_axis(omega);
    let trans = Translation3::from(v);
    Iso3::from_parts(trans, rot) * pose
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Largest `pose` index across a residual slice, or 0 if empty.
fn max_pose(residuals: &[TargetFeatureResidual]) -> usize {
    residuals.iter().map(|r| r.pose).max().unwrap_or(0)
}

/// Intrinsic floor for a generic single-camera view slice.
fn intrinsic_floor_single_cam<M>(
    camera: &PinholeCamera,
    k: &FxFyCxCySkew<f64>,
    views: &[View<M>],
    cam_idx: usize,
) -> Vec<TargetFeatureResidual> {
    let mut out = Vec::new();
    for (view_idx, view) in views.iter().enumerate() {
        let p3d = &view.obs.points_3d;
        let p2d = &view.obs.points_2d;
        let Some(pose) = intrinsic_floor_view(camera, k, p3d, p2d) else {
            continue;
        };
        push_view_residuals(&mut out, camera, &pose, p3d, p2d, view_idx, cam_idx);
    }
    out
}

/// Intrinsic floor over hand-eye views (single camera).
fn intrinsic_floor_handeye(
    camera: &PinholeCamera,
    k: &FxFyCxCySkew<f64>,
    views: &[SingleCamHandeyeView],
    cam_idx: usize,
) -> Vec<TargetFeatureResidual> {
    let mut out = Vec::new();
    for (view_idx, view) in views.iter().enumerate() {
        let p3d = &view.obs.points_3d;
        let p2d = &view.obs.points_2d;
        let Some(pose) = intrinsic_floor_view(camera, k, p3d, p2d) else {
            continue;
        };
        push_view_residuals(&mut out, camera, &pose, p3d, p2d, view_idx, cam_idx);
    }
    out
}

/// Reproject one view's corners under `camera_se3_target` and push one
/// [`TargetFeatureResidual`] per corner with the given `pose` / `camera` index.
///
/// `TargetFeatureResidual` is `#[non_exhaustive]` in the core crate, so it cannot
/// be built with a struct literal from here; we default-construct then assign the
/// public fields.
#[allow(clippy::field_reassign_with_default)]
fn push_view_residuals(
    out: &mut Vec<TargetFeatureResidual>,
    camera: &PinholeCamera,
    camera_se3_target: &Iso3,
    obs_3d: &[Pt3],
    obs_2d: &[Pt2],
    view_idx: usize,
    cam_idx: usize,
) {
    for (feature, (p3d, p2d)) in obs_3d.iter().zip(obs_2d.iter()).enumerate() {
        let p_cam = camera_se3_target * p3d;
        let (projected_px, error_px) = match camera.project_camera_point(&p_cam.coords) {
            Some(proj) => (
                Some([proj.x, proj.y]),
                Some((proj.coords - p2d.coords).norm()),
            ),
            None => (None, None),
        };
        let mut r = TargetFeatureResidual::default();
        r.pose = view_idx;
        r.camera = cam_idx;
        r.feature = feature;
        r.target_xyz_m = [p3d.x, p3d.y, p3d.z];
        r.observed_px = [p2d.x, p2d.y];
        r.projected_px = projected_px;
        r.error_px = error_px;
        out.push(r);
    }
}

/// Recompute the rig-extrinsic constrained residuals from the export's cameras
/// and the `cam_se3_rig * rig_se3_target` chain, for the rare case the export
/// does not carry its own `per_feature_residuals.target`.
///
/// Iterates view-major, camera-inner (skipping camera-`None` slots), matching the
/// workspace indexing convention.
fn rig_constrained_residuals<M>(
    export: &RigExtrinsicsExport,
    dataset: &RigDataset<M>,
) -> Vec<TargetFeatureResidual> {
    let mut out = Vec::new();
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let Some(rig_se3_target) = export.rig_se3_target.get(view_idx) else {
            continue;
        };
        for (cam_idx, camera) in export.cameras.iter().enumerate() {
            let Some(obs) = view.obs.cameras.get(cam_idx).and_then(|c| c.as_ref()) else {
                continue;
            };
            let cam_se3_target = export.cam_se3_rig[cam_idx] * rig_se3_target;
            push_view_residuals(
                &mut out,
                camera,
                &cam_se3_target,
                &obs.points_3d,
                &obs.points_2d,
                view_idx,
                cam_idx,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests;

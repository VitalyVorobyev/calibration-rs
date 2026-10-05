//! Solver-independent quality metrics.
//!
//! Everything here is computed from a solved model (the pipeline export) and
//! the scene's ground truth. Nothing reads solver internals, so the numbers
//! are comparable across solver versions and backends.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use vision_calibration::core::{BrownConrady5, FxFyCxCySkew, Iso3, Real, ScheimpflugParams};
use vision_calibration::optim::{LaserPlane, RobustLoss};
use vision_calibration_core::PerFeatureResiduals;
use vision_calibration_core::synthetic::poses::pose_error;

use super::scenes::OutlierRef;

/// One camera's intrinsics, distortion and (for tilted sensors) tilt.
#[derive(Debug, Clone, Copy)]
pub struct CameraParamsSet {
    /// Pinhole intrinsics.
    pub k: FxFyCxCySkew<Real>,
    /// Brown-Conrady distortion.
    pub dist: BrownConrady5<Real>,
    /// Scheimpflug tilt, `None` for a plain pinhole sensor.
    pub sensor: Option<ScheimpflugParams>,
}

/// The model parameters a scene is judged on: either the ground truth that
/// generated the data or the solver's estimate.
///
/// Absent pieces (`None` / empty) are skipped by [`GtErrors::between`].
#[derive(Debug, Clone, Default)]
pub struct ModelParams {
    /// Per-camera parameters (empty when the problem does not estimate them).
    pub cameras: Vec<CameraParamsSet>,
    /// Rig extrinsics `cam_se3_rig`, reference camera at index 0.
    pub cam_se3_rig: Vec<Iso3>,
    /// Hand-eye transform (`gripper_se3_camera` / `gripper_se3_rig`).
    pub handeye: Option<Iso3>,
    /// Laser planes in the camera frame, one per laser-carrying camera.
    pub planes_cam: Vec<LaserPlane>,
}

/// Ground-truth parameter errors. Every field is `None` when the problem does
/// not estimate the quantity.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GtErrors {
    /// Max over cameras and axes of `|f - f_gt| / f_gt`.
    pub focal_rel: Option<f64>,
    /// Max over cameras of the principal-point distance, pixels.
    pub principal_point_px: Option<f64>,
    /// Max over cameras of `|Δ|` for `k1, k2, k3, p1, p2`.
    pub distortion_abs: Option<f64>,
    /// Max over cameras of the Scheimpflug tilt error `hypot(Δtilt_x, Δtilt_y)`, degrees.
    pub sensor_tilt_deg: Option<f64>,
    /// Max rig extrinsic rotation error, degrees.
    pub rig_rot_deg: Option<f64>,
    /// Max rig extrinsic translation error, millimetres.
    pub rig_trans_mm: Option<f64>,
    /// Hand-eye rotation error, degrees.
    pub handeye_rot_deg: Option<f64>,
    /// Hand-eye translation error, millimetres.
    pub handeye_trans_mm: Option<f64>,
    /// Max laser-plane normal angle, degrees.
    pub laser_normal_deg: Option<f64>,
    /// Max laser-plane distance error, millimetres.
    pub laser_distance_mm: Option<f64>,
}

impl GtErrors {
    /// Compare an estimate against the ground truth. A field is `Some` only
    /// when both sides carry the quantity.
    pub fn between(est: &ModelParams, gt: &ModelParams) -> Self {
        let mut out = Self::default();

        let cams: Vec<_> = est.cameras.iter().zip(&gt.cameras).collect();
        out.focal_rel = max_of(cams.iter().map(|(e, g)| {
            ((e.k.fx - g.k.fx).abs() / g.k.fx).max((e.k.fy - g.k.fy).abs() / g.k.fy)
        }));
        out.principal_point_px = max_of(
            cams.iter()
                .map(|(e, g)| (e.k.cx - g.k.cx).hypot(e.k.cy - g.k.cy)),
        );
        out.distortion_abs = max_of(cams.iter().map(|(e, g)| {
            [
                e.dist.k1 - g.dist.k1,
                e.dist.k2 - g.dist.k2,
                e.dist.k3 - g.dist.k3,
                e.dist.p1 - g.dist.p1,
                e.dist.p2 - g.dist.p2,
            ]
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()))
        }));
        out.sensor_tilt_deg = max_of(cams.iter().filter_map(|(e, g)| {
            let (es, gs) = (e.sensor?, g.sensor?);
            Some(
                (es.tilt_x - gs.tilt_x)
                    .hypot(es.tilt_y - gs.tilt_y)
                    .to_degrees(),
            )
        }));

        // Camera 0 pins the rig frame (identity in both), so skip it.
        let rig: Vec<_> = est
            .cam_se3_rig
            .iter()
            .zip(&gt.cam_se3_rig)
            .skip(1)
            .map(|(e, g)| pose_error(e, g))
            .collect();
        out.rig_rot_deg = max_of(rig.iter().map(|e| e.rot_deg));
        out.rig_trans_mm = max_of(rig.iter().map(|e| e.trans * 1e3));

        if let (Some(e), Some(g)) = (&est.handeye, &gt.handeye) {
            let err = pose_error(e, g);
            out.handeye_rot_deg = Some(err.rot_deg);
            out.handeye_trans_mm = Some(err.trans * 1e3);
        }

        let planes: Vec<_> = est
            .planes_cam
            .iter()
            .zip(&gt.planes_cam)
            .map(|(e, g)| plane_error(e, g))
            .collect();
        out.laser_normal_deg = max_of(planes.iter().map(|p| p.0));
        out.laser_distance_mm = max_of(planes.iter().map(|p| p.1));
        out
    }

    /// Named, present values in a fixed order (for tables and comparisons).
    pub fn entries(&self) -> Vec<(&'static str, f64)> {
        [
            ("focal_rel", self.focal_rel),
            ("pp_px", self.principal_point_px),
            ("dist_abs", self.distortion_abs),
            ("tilt_deg", self.sensor_tilt_deg),
            ("rig_rot_deg", self.rig_rot_deg),
            ("rig_trans_mm", self.rig_trans_mm),
            ("he_rot_deg", self.handeye_rot_deg),
            ("he_trans_mm", self.handeye_trans_mm),
            ("laser_n_deg", self.laser_normal_deg),
            ("laser_d_mm", self.laser_distance_mm),
        ]
        .into_iter()
        .filter_map(|(name, v)| v.map(|v| (name, v)))
        .collect()
    }
}

fn max_of(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.fold(None, |m, v| Some(m.map_or(v, |m: f64| m.max(v))))
}

/// Angle (degrees) and distance error (millimetres) between two laser planes.
///
/// `(n, d)` and `(-n, -d)` name the same plane, so the estimate is sign-aligned
/// to the ground truth before comparing.
pub fn plane_error(est: &LaserPlane, gt: &LaserPlane) -> (f64, f64) {
    let (mut n, mut d) = (est.normal.into_inner(), est.distance);
    let n_gt = gt.normal.into_inner();
    if n.dot(&n_gt) < 0.0 {
        n = -n;
        d = -d;
    }
    // atan2 keeps full precision for nearly parallel normals, unlike acos.
    let angle = n.cross(&n_gt).norm().atan2(n.dot(&n_gt)).to_degrees();
    (angle, (d - gt.distance).abs() * 1e3)
}

/// Robust-loss value `ρ(s)` at squared residual `s`, in the Ceres convention
/// (the convention tiny-solver implements).
///
/// `None`: `s`. `Huber(k)`: `s` for `s ≤ k²`, else `2k√s − k²`.
/// `Cauchy(k)`: `k² ln(1 + s/k²)`. `Arctan(k)`: `k · atan2(s, k)`.
pub fn rho(loss: RobustLoss, s: f64) -> f64 {
    match loss {
        RobustLoss::None => s,
        RobustLoss::Huber { scale: k } => {
            if s <= k * k {
                s
            } else {
                2.0 * k * s.sqrt() - k * k
            }
        }
        RobustLoss::Cauchy { scale: k } => k * k * (1.0 + s / (k * k)).ln(),
        RobustLoss::Arctan { scale: k } => k * s.atan2(k),
    }
}

/// Quality of one solve, judged only on the exported residuals and the
/// ground truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// `½ Σ ρ(eᵢ²)` over every target feature, with the scene's loss.
    pub objective: f64,
    /// Target reprojection RMS excluding the injected outliers, pixels.
    pub inlier_rms_px: f64,
    /// Target reprojection RMS including the injected outliers, pixels.
    pub all_rms_px: f64,
    /// Laser residual RMS (pixel domain) when the problem has laser features.
    pub laser_rms_px: Option<f64>,
    /// Number of target features scored.
    pub num_features: usize,
    /// Number of injected outliers.
    pub num_outliers: usize,
    /// Ground-truth parameter errors.
    pub gt: GtErrors,
}

impl QualityMetrics {
    /// Score a solve.
    ///
    /// `residuals` is the export's per-feature residual table; `outliers` the
    /// injected outlier set; `loss` the scene's robust loss.
    pub fn evaluate(
        residuals: &PerFeatureResiduals,
        outliers: &[OutlierRef],
        loss: RobustLoss,
        est: &ModelParams,
        gt: &ModelParams,
    ) -> Self {
        let outlier_set: BTreeSet<(usize, usize, usize)> =
            outliers.iter().map(OutlierRef::key).collect();

        let mut objective = 0.0;
        let (mut sq_all, mut n_all) = (0.0, 0usize);
        let (mut sq_in, mut n_in) = (0.0, 0usize);
        for f in &residuals.target {
            let Some(e) = f.error_px else { continue };
            let s = e * e;
            objective += rho(loss, s);
            sq_all += s;
            n_all += 1;
            if !outlier_set.contains(&(f.pose, f.camera, f.feature)) {
                sq_in += s;
                n_in += 1;
            }
        }
        let laser: Vec<f64> = residuals
            .laser
            .iter()
            .filter_map(|l| l.residual_px)
            .collect();
        let laser_rms_px = (!laser.is_empty())
            .then(|| (laser.iter().map(|r| r * r).sum::<f64>() / laser.len() as f64).sqrt());

        Self {
            objective: 0.5 * objective,
            inlier_rms_px: rms(sq_in, n_in),
            all_rms_px: rms(sq_all, n_all),
            laser_rms_px,
            num_features: n_all,
            num_outliers: outliers.len(),
            gt: GtErrors::between(est, gt),
        }
    }
}

fn rms(sum_sq: f64, n: usize) -> f64 {
    if n == 0 {
        f64::NAN
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use vision_calibration_core::synthetic::poses::make_iso;
    use vision_calibration_core::{PerFeatureResiduals, TargetFeatureResidual};

    #[test]
    fn rho_matches_the_ceres_convention() {
        assert_eq!(rho(RobustLoss::None, 4.0), 4.0);
        let h = RobustLoss::Huber { scale: 2.0 };
        // Quadratic inside the knee, linear outside, continuous at s = k².
        assert_eq!(rho(h, 3.0), 3.0);
        assert!((rho(h, 4.0) - 4.0).abs() < 1e-12);
        let just_above = rho(h, 4.0 + 1e-9);
        assert!((just_above - 4.0).abs() < 1e-8);
        assert!((rho(h, 25.0) - (2.0 * 2.0 * 5.0 - 4.0)).abs() < 1e-12);
        let c = RobustLoss::Cauchy { scale: 1.0 };
        assert!((rho(c, 1.0) - 2.0_f64.ln()).abs() < 1e-12);
        assert!((rho(RobustLoss::Cauchy { scale: 2.0 }, 4.0) - 4.0 * 2.0_f64.ln()).abs() < 1e-12);
        let a = RobustLoss::Arctan { scale: 2.0 };
        assert!((rho(a, 2.0) - 2.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        // Robust losses never exceed the squared loss.
        for s in [0.1, 1.0, 10.0, 1000.0] {
            assert!(rho(h, s) <= s + 1e-12 && rho(c, s) <= s + 1e-12);
        }
    }

    #[test]
    fn plane_error_is_sign_aligned() {
        let gt = LaserPlane::new(Vector3::new(0.3, 0.0, 1.0), -0.5);
        let flipped = LaserPlane::new(-Vector3::new(0.3, 0.0, 1.0), 0.5);
        let (ang, dist) = plane_error(&flipped, &gt);
        assert!(ang < 1e-6 && dist < 1e-9, "{ang} {dist}");
        let shifted = LaserPlane::new(Vector3::new(0.3, 0.0, 1.0), -0.501);
        let (_, dist) = plane_error(&shifted, &gt);
        assert!((dist - 1.0).abs() < 1e-9);
        let tilted = LaserPlane::new(Vector3::new(0.3, 0.1, 1.0), -0.5);
        let (ang, _) = plane_error(&tilted, &gt);
        assert!(ang > 1.0 && ang < 10.0);
    }

    fn cam(fx: f64) -> CameraParamsSet {
        CameraParamsSet {
            k: FxFyCxCySkew {
                fx,
                fy: fx,
                cx: 640.0,
                cy: 480.0,
                skew: 0.0,
            },
            dist: BrownConrady5::default(),
            sensor: None,
        }
    }

    #[test]
    fn gt_errors_report_only_shared_quantities() {
        let gt = ModelParams {
            cameras: vec![cam(1000.0), cam(1000.0)],
            cam_se3_rig: vec![
                Iso3::identity(),
                make_iso((0.0, 0.1, 0.0), (-0.1, 0.0, 0.0)),
            ],
            handeye: Some(make_iso((0.1, 0.0, 0.0), (0.0, 0.0, 0.1))),
            planes_cam: vec![],
        };
        let mut est = gt.clone();
        est.cameras[1].k.fx = 1010.0;
        est.cam_se3_rig[1] = make_iso((0.0, 0.1, 0.0), (-0.1, 0.0, 0.002));
        let e = GtErrors::between(&est, &gt);
        assert!((e.focal_rel.unwrap() - 0.01).abs() < 1e-12);
        assert!((e.rig_trans_mm.unwrap() - 2.0).abs() < 1e-9);
        assert!(e.rig_rot_deg.unwrap() < 1e-9);
        assert!(e.handeye_rot_deg.unwrap() < 1e-9);
        assert!(e.laser_normal_deg.is_none() && e.sensor_tilt_deg.is_none());

        let none = GtErrors::between(&ModelParams::default(), &gt);
        assert_eq!(none, GtErrors::default());
    }

    #[test]
    fn evaluate_separates_inliers_from_outliers() {
        let mk = |pose, feature, e| {
            let mut r = TargetFeatureResidual::default();
            r.pose = pose;
            r.feature = feature;
            r.error_px = Some(e);
            r
        };
        let mut residuals = PerFeatureResiduals::default();
        residuals.target = vec![mk(0, 0, 1.0), mk(0, 1, 1.0), mk(1, 0, 11.0)];
        let outliers = [OutlierRef {
            pose: 1,
            camera: 0,
            feature: 0,
        }];
        let m = QualityMetrics::evaluate(
            &residuals,
            &outliers,
            RobustLoss::Huber { scale: 1.0 },
            &ModelParams::default(),
            &ModelParams::default(),
        );
        assert!((m.inlier_rms_px - 1.0).abs() < 1e-12);
        assert!((m.all_rms_px - (123.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        // ½(1 + 1 + (2·11 − 1)) = 11.5
        assert!((m.objective - 11.5).abs() < 1e-12);
        assert_eq!((m.num_features, m.num_outliers), (3, 1));
        assert!(m.laser_rms_px.is_none());
    }
}

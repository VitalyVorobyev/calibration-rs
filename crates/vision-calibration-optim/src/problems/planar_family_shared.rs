use anyhow::{Result, ensure};
use nalgebra::DVector;
use std::collections::{HashMap, HashSet};
use vision_calibration_core::{
    BrownConrady5, FxFyCxCySkew, Iso3, PlanarDataset, Real, ScheimpflugParams,
};

use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{DISTORTION_DIM, pack_distortion};
use crate::params::intrinsics::{INTRINSICS_DIM, pack_intrinsics};
use crate::params::pose_se3::iso3_to_se3_dvec;

#[derive(Debug, Clone, Copy)]
pub(crate) enum PlanarReprojectionFactorModel {
    PinholeDistortion,
    PinholeDistortionScheimpflug,
}

#[derive(Debug, Clone)]
pub(crate) struct PlanarReprojectionIrOptions {
    pub robust_loss: RobustLoss,
    pub fix_intrinsics_indices: Vec<usize>,
    pub fix_distortion_indices: Vec<usize>,
    pub fix_pose_indices: Vec<usize>,
    pub sensor: Option<PlanarSensorIrOptions>,
    pub factor_model: PlanarReprojectionFactorModel,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanarSensorIrOptions {
    pub params: ScheimpflugParams,
    pub fix_indices: [bool; 2],
}

impl PlanarSensorIrOptions {
    fn fixed_index_vec(self) -> Vec<usize> {
        let mut indices = Vec::new();
        if self.fix_indices[0] {
            indices.push(0);
        }
        if self.fix_indices[1] {
            indices.push(1);
        }
        indices
    }
}

pub(crate) fn build_planar_reprojection_ir(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    poses: &[Iso3],
    opts: &PlanarReprojectionIrOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        dataset.num_views() == poses.len(),
        "pose count ({}) must match number of views ({})",
        poses.len(),
        dataset.num_views()
    );
    for &idx in &opts.fix_pose_indices {
        ensure!(
            idx < dataset.num_views(),
            "fixed pose index {} out of range ({} views)",
            idx,
            dataset.num_views()
        );
    }

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();
    let fixed_pose_set: HashSet<usize> = opts.fix_pose_indices.iter().copied().collect();

    let cam_id = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_intrinsics_indices),
        None,
    );
    initial_map.insert("cam".to_string(), pack_intrinsics(intrinsics)?);

    let dist_id = ir.add_param_block(
        "dist",
        DISTORTION_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_distortion_indices),
        None,
    );
    initial_map.insert("dist".to_string(), pack_distortion(distortion));

    let sensor_id = if let Some(sensor) = opts.sensor {
        let id = ir.add_param_block(
            "sensor",
            2,
            ManifoldKind::Euclidean,
            FixedMask::fix_indices(&sensor.fixed_index_vec()),
            None,
        );
        initial_map.insert(
            "sensor".to_string(),
            DVector::from_vec(vec![sensor.params.tilt_x, sensor.params.tilt_y]),
        );
        Some(id)
    } else {
        None
    };

    let mut pose_ids = Vec::with_capacity(poses.len());
    for (view_idx, pose) in poses.iter().enumerate() {
        let fixed = if fixed_pose_set.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let pose_key = format!("pose/{view_idx}");
        let pose_id = ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(pose_key, iso3_to_se3_dvec(pose));
        pose_ids.push(pose_id);
    }

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_id = pose_ids[view_idx];
        for (point_idx, (pw, uv)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let (params, factor) = match opts.factor_model {
                PlanarReprojectionFactorModel::PinholeDistortion => (
                    vec![cam_id, dist_id, pose_id],
                    FactorKind::ReprojPointPinhole4Dist5 {
                        pw: [pw.x, pw.y, pw.z],
                        uv: [uv.x, uv.y],
                        w: view.obs.weight(point_idx),
                    },
                ),
                PlanarReprojectionFactorModel::PinholeDistortionScheimpflug => {
                    let sensor_id = sensor_id
                        .expect("internal invariant: sensor_id required for scheimpflug model");
                    (
                        vec![cam_id, dist_id, sensor_id, pose_id],
                        FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 {
                            pw: [pw.x, pw.y, pw.z],
                            uv: [uv.x, uv.y],
                            w: view.obs.weight(point_idx),
                        },
                    )
                }
            };
            ir.add_residual_block(ResidualBlock {
                params,
                loss: opts.robust_loss,
                factor,
                residual_dim: 2,
            });
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

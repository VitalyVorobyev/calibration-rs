//! A solve's covariance in the public terms of [`Uncertainty`]: parameter
//! labels instead of IR blocks and coordinates, shared parameters apart
//! from per-view ones.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, Matrix3};
use vision_calibration_core::{
    Iso3, ParameterCovariance, ResidualNoise, Uncertainty, UncertaintyUnavailable, ViewCovariance,
};

use crate::backend::BackendSolution;
use crate::backend::covariance::{Coordinate, ParamUncertainty, UnavailableReason};
use crate::ir::{CameraModelSpec, HandeyeMode, ManifoldKind, ParamBlock, ProblemIR};
use crate::params::camera::camera_block_labels;
use crate::params::pose_se3::se3_dvec_to_iso3;

/// How one IR block's coordinates are named.
pub(crate) struct BlockLabels {
    /// The view the block belongs to; `None` for a shared block.
    pub view: Option<usize>,
    /// The block's name in labels, e.g. `camera_se3_target`.
    pub name: String,
    /// One label per reported coordinate (a Euclidean block's components, an
    /// SE3 block's `[rx, ry, rz, tx, ty, tz]`, an S² block's `[nx, ny, nz]`).
    /// Per-view labels omit the view, which [`ViewCovariance::view`] carries.
    pub coordinates: Vec<String>,
    /// For an SE3 block reported as the inverse of the pose it stores: the
    /// stored pose at the solution.
    pub inverse_of: Option<Iso3>,
    /// A block the problem fixes by construction, such as the identity
    /// camera-to-rig pose of a single-camera problem; it is not listed among
    /// the fixed parameters.
    pub internal: bool,
}

impl BlockLabels {
    /// A shared block whose coordinates have the given labels.
    pub(crate) fn shared(name: &str, coordinates: &[&str]) -> Self {
        Self::shared_owned(
            name.to_string(),
            coordinates.iter().map(|c| c.to_string()).collect(),
        )
    }

    /// [`Self::shared`] from owned labels.
    pub(crate) fn shared_owned(name: String, coordinates: Vec<String>) -> Self {
        Self {
            view: None,
            name,
            coordinates,
            inverse_of: None,
            internal: false,
        }
    }

    /// A shared SE3 pose, labelled `<name>.rx … <name>.tz`.
    pub(crate) fn shared_pose(name: &str) -> Self {
        Self {
            view: None,
            name: name.to_string(),
            coordinates: pose_labels(name),
            inverse_of: None,
            internal: false,
        }
    }

    /// A per-view SE3 pose, labelled `<name>.rx … <name>.tz`.
    pub(crate) fn view_pose(view: usize, name: &str) -> Self {
        Self {
            view: Some(view),
            ..Self::shared_pose(name)
        }
    }

    /// Report this SE3 block as the inverse of the stored pose `stored`.
    fn inverted(self, stored: Iso3) -> Self {
        Self {
            inverse_of: Some(stored),
            ..self
        }
    }

    fn internal(self) -> Self {
        Self {
            internal: true,
            ..self
        }
    }
}

/// `<name>.rx, <name>.ry, <name>.rz, <name>.tx, <name>.ty, <name>.tz`.
pub(crate) fn pose_labels(name: &str) -> Vec<String> {
    ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .map(|c| format!("{name}.{c}"))
        .collect()
}

/// The labelled uncertainty of a solve of `ir`; `labels` names each IR block
/// (by id).
pub(crate) fn labelled(
    ir: &ProblemIR,
    solve: &ParamUncertainty,
    labels: &[BlockLabels],
) -> Uncertainty {
    debug_assert_eq!(labels.len(), ir.params.len(), "one label set per IR block");
    let label = |c: &Coordinate| coordinate_label(&labels[c.block], c.index);
    let estimate = match solve {
        ParamUncertainty::Available(estimate) => estimate,
        ParamUncertainty::Unavailable(reason) => {
            let reason = match reason {
                UnavailableReason::NotConverged => UncertaintyUnavailable::NotConverged,
                UnavailableReason::NoRedundancy => UncertaintyUnavailable::NoRedundancy,
                UnavailableReason::RankDeficient { unobservable } => {
                    UncertaintyUnavailable::RankDeficient {
                        unobservable: dedup(unobservable.iter().map(label)),
                    }
                }
                UnavailableReason::AtBound { coordinates } => UncertaintyUnavailable::AtBound {
                    labels: dedup(coordinates.iter().map(label)),
                },
            };
            return Uncertainty::Unavailable { reason };
        }
    };

    let matrix = report_inverses(&estimate.matrix, &estimate.coordinates, labels);

    // Matrix positions of the shared coordinates and of each view's.
    let mut shared = Vec::new();
    let mut views: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, c) in estimate.coordinates.iter().enumerate() {
        match labels[c.block].view {
            None => shared.push(i),
            Some(view) => views.entry(view).or_default().push(i),
        }
    }
    let block = |positions: &[usize]| -> Vec<Vec<f64>> {
        positions
            .iter()
            .map(|&a| positions.iter().map(|&b| matrix[(a, b)]).collect())
            .collect()
    };
    let local_label = |i: usize| {
        let c = estimate.coordinates[i];
        labels[c.block].coordinates[c.index].clone()
    };

    Uncertainty::Available(ParameterCovariance {
        labels: shared.iter().map(|&i| local_label(i)).collect(),
        covariance: block(&shared),
        views: views
            .into_iter()
            .map(|(view, positions)| ViewCovariance {
                view,
                labels: positions.iter().map(|&i| local_label(i)).collect(),
                covariance: block(&positions),
            })
            .collect(),
        fixed: fixed_labels(ir, labels),
        noise: estimate
            .families
            .iter()
            .map(|f| ResidualNoise {
                kind: f.family,
                sigma: f.sigma,
                count: f.num_residuals,
                camera: None,
            })
            .collect(),
        num_residuals: estimate.num_residuals,
        dof: estimate.dof,
    })
}

/// Labels in first-seen order, without repeats.
fn dedup(labels: impl Iterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for label in labels {
        if !out.contains(&label) {
            out.push(label);
        }
    }
    out
}

/// The covariance with every block flagged [`BlockLabels::inverse_of`]
/// mapped to the tangent of the inverse pose.
///
/// For `T' = T·Exp(δ)`, `T'⁻¹ = T⁻¹·Exp(−Ad_T δ)`, so the inverse's
/// right-perturbation coordinates are `−Ad_T δ` with, for `δ = [ω, v]`,
/// `Ad_T = [[R, 0], [[t]× R, R]]`.
fn report_inverses(
    matrix: &DMatrix<f64>,
    coordinates: &[Coordinate],
    labels: &[BlockLabels],
) -> DMatrix<f64> {
    let mut map: Option<DMatrix<f64>> = None;
    for (block, names) in labels.iter().enumerate() {
        let Some(stored) = names.inverse_of else {
            continue;
        };
        let positions: Vec<usize> = (0..coordinates.len())
            .filter(|&i| coordinates[i].block == block)
            .collect();
        if positions.is_empty() {
            continue; // fixed block
        }
        debug_assert_eq!(
            positions.len(),
            6,
            "an inverted SE3 block is free as a whole"
        );
        let r = stored.rotation.to_rotation_matrix().into_inner();
        let t = stored.translation.vector;
        let tr = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0) * r;
        let m = map.get_or_insert_with(|| DMatrix::identity(matrix.nrows(), matrix.ncols()));
        for a in 0..3 {
            for b in 0..3 {
                let at = |row: usize, col: usize| (positions[row], positions[col]);
                m[at(a, b)] = -r[(a, b)];
                m[at(a + 3, b)] = -tr[(a, b)];
                m[at(a + 3, b + 3)] = -r[(a, b)];
                m[at(a, b + 3)] = 0.0;
            }
        }
    }
    match map {
        Some(m) => &m * matrix * m.transpose(),
        None => matrix.clone(),
    }
}

/// A coordinate's label, with `view<i>.` in front of a per-view one. An
/// inverted block's coordinates mix under the inversion, so a statement
/// about one of them names the block.
fn coordinate_label(labels: &BlockLabels, index: usize) -> String {
    let local = if labels.inverse_of.is_some() {
        labels.name.clone()
    } else {
        labels.coordinates[index].clone()
    };
    match labels.view {
        None => local,
        Some(view) => format!("view{view}.{local}"),
    }
}

/// Labels of the fixed coordinates: each one of a shared Euclidean block,
/// a fully fixed pose or per-view block by its name alone
/// (`view3.camera_se3_target`).
fn fixed_labels(ir: &ProblemIR, labels: &[BlockLabels]) -> Vec<String> {
    let mut out = Vec::new();
    for (block, names) in ir.params.iter().zip(labels) {
        if names.internal {
            continue;
        }
        let all_fixed = block.fixed.is_all_fixed(block.dim);
        let euclidean = block.manifold == ManifoldKind::Euclidean;
        match (names.view, all_fixed) {
            (Some(view), true) => out.push(format!("view{view}.{}", names.name)),
            (None, true) if !euclidean => out.push(names.name.clone()),
            (_, true) => {
                out.extend((0..names.coordinates.len()).map(|i| coordinate_label(names, i)))
            }
            (_, false) if euclidean => out.extend(
                (0..block.dim)
                    .filter(|&i| block.fixed.contains(i))
                    .map(|i| coordinate_label(names, i)),
            ),
            (_, false) => {}
        }
    }
    out
}

/// Labels for the blocks of a single-camera planar IR (the camera blocks of
/// a camera of model `camera`, `pose/<view>`, and a laserline's `plane_n`,
/// `plane_d`).
pub(crate) fn planar_block_labels(block: &ParamBlock, camera: CameraModelSpec) -> BlockLabels {
    if let Some(labels) = camera_block_labels(&block.name, camera, str::to_string) {
        return labels;
    }
    match block.name.as_str() {
        "plane_n" => BlockLabels::shared(PLANE_NORMAL, &PLANE_NORMAL_LABELS),
        "plane_d" => BlockLabels::shared(PLANE_DISTANCE, &[PLANE_DISTANCE]),
        name => {
            let view = name
                .strip_prefix("pose/")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("unexpected planar IR block {name}"));
            BlockLabels::view_pose(view, "camera_se3_target")
        }
    }
}

/// One uncertainty for cameras solved apart, from each camera's own
/// (`parts[i]` for camera `i`): the covariance is block-diagonal, labels gain
/// a `cam<i>.` prefix and each noise level names its camera when there is
/// more than one camera. The parts' fixed per-view parameters are dropped,
/// since their view indices are local to each camera's solve; `rig_fixed`
/// names what they stand for in the rig.
pub(crate) fn per_camera(parts: &[Uncertainty], rig_fixed: Vec<String>) -> Uncertainty {
    let multi_camera = parts.len() > 1;
    let prefix = |i: usize, label: &str| {
        if multi_camera {
            format!("cam{i}.{label}")
        } else {
            label.to_string()
        }
    };
    let prefixed = |i: usize, labels: &[String]| -> Vec<String> {
        labels.iter().map(|l| prefix(i, l)).collect()
    };
    let mut available = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        match part {
            Uncertainty::Available(cov) => available.push(cov),
            Uncertainty::Unavailable { reason } => {
                let reason = match reason {
                    UncertaintyUnavailable::RankDeficient { unobservable } => {
                        UncertaintyUnavailable::RankDeficient {
                            unobservable: prefixed(i, unobservable),
                        }
                    }
                    UncertaintyUnavailable::AtBound { labels } => UncertaintyUnavailable::AtBound {
                        labels: prefixed(i, labels),
                    },
                    other => other.clone(),
                };
                return Uncertainty::Unavailable { reason };
            }
        }
    }

    let n: usize = available.iter().map(|c| c.labels.len()).sum();
    let mut covariance = vec![vec![0.0; n]; n];
    let mut labels = Vec::with_capacity(n);
    let mut noise = Vec::new();
    let mut fixed = Vec::new();
    for (i, cov) in available.iter().enumerate() {
        debug_assert!(
            cov.views.is_empty(),
            "per-camera solves hold the views fixed"
        );
        let offset = labels.len();
        for (a, row) in cov.covariance.iter().enumerate() {
            covariance[offset + a][offset..offset + row.len()].copy_from_slice(row);
        }
        labels.extend(prefixed(i, &cov.labels));
        fixed.extend(
            cov.fixed
                .iter()
                .filter(|label| !label.starts_with("view"))
                .map(|label| prefix(i, label)),
        );
        noise.extend(cov.noise.iter().map(|level| ResidualNoise {
            camera: multi_camera.then_some(i),
            ..level.clone()
        }));
    }
    fixed.extend(rig_fixed);
    Uncertainty::Available(ParameterCovariance {
        labels,
        covariance,
        views: Vec::new(),
        fixed,
        noise,
        num_residuals: available.iter().map(|c| c.num_residuals).sum(),
        dof: available.iter().map(|c| c.dof).sum(),
    })
}

/// Labels of the intrinsics block, in IR order.
pub(crate) const INTRINSICS_LABELS: [&str; 4] = ["fx", "fy", "cx", "cy"];
/// Labels of the Scheimpflug sensor block, in IR order.
pub(crate) const SENSOR_LABELS: [&str; 2] = ["tilt_x", "tilt_y"];

/// A laser plane's unit normal, reported by its three components (the
/// covariance of a unit vector has rank 2), and its signed distance.
const PLANE_NORMAL: &str = "plane.normal";
const PLANE_NORMAL_LABELS: [&str; 3] = ["plane.nx", "plane.ny", "plane.nz"];
const PLANE_DISTANCE: &str = "plane.d";

/// The shape of a rig or hand-eye IR, for naming its blocks.
pub(crate) struct RigLabelling<'a> {
    /// Each camera's model, which names its blocks' coordinates.
    pub cameras: &'a [CameraModelSpec],
    /// Whether labels carry a `cam<i>.` prefix. Single-camera problems have
    /// none: their camera is the rig, and its camera-to-rig pose is the
    /// identity.
    pub multi_camera: bool,
    /// The hand-eye mode, which names the hand-eye and target poses.
    pub handeye_mode: Option<HandeyeMode>,
}

impl RigLabelling<'_> {
    /// Labels for every block of `ir` after `solution`.
    pub(crate) fn labels(&self, ir: &ProblemIR, solution: &BackendSolution) -> Vec<BlockLabels> {
        ir.params
            .iter()
            .map(|block| self.block(block, solution))
            .collect()
    }

    /// Labels for the blocks of the rig and hand-eye IRs: the camera blocks
    /// (`cam/<i>`, `proj/<i>`, `dist/<i>`, `sensor/<i>`), `extr/<i>` (stored as `rig_se3_cam`,
    /// reported as `cam_se3_rig`), `handeye`, `target` (`target_ref` in the
    /// laserline IR), `target/<view>`, `rig_pose/<view>`,
    /// `robot_delta/<view>`, and each camera's laser plane `plane_n/<i>`,
    /// `plane_d/<i>` (in that camera's frame).
    fn block(&self, block: &ParamBlock, solution: &BackendSolution) -> BlockLabels {
        let (kind, index) = match block.name.split_once('/') {
            Some((kind, index)) => (
                kind,
                Some(
                    index
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("unexpected rig IR block {}", block.name)),
                ),
            ),
            None => (block.name.as_str(), None),
        };
        let camera = |name: &str, coordinates: &[&str]| {
            let i = index.expect("per-camera blocks carry the camera index");
            let prefix = |s: &str| {
                if self.multi_camera {
                    format!("cam{i}.{s}")
                } else {
                    s.to_string()
                }
            };
            BlockLabels {
                view: None,
                name: prefix(name),
                coordinates: coordinates.iter().map(|c| prefix(c)).collect(),
                inverse_of: None,
                internal: false,
            }
        };
        let rig = if self.multi_camera { "rig" } else { "camera" };
        let target = match self.handeye_mode {
            Some(HandeyeMode::EyeInHand) => "base_se3_target",
            Some(HandeyeMode::EyeToHand) => "gripper_se3_target",
            None => "rig_se3_target",
        };
        if let Some(i) = index
            && matches!(kind, "cam" | "proj" | "dist" | "sensor")
            && let Some(labels) = camera_block_labels(kind, self.cameras[i], |s| {
                if self.multi_camera {
                    format!("cam{i}.{s}")
                } else {
                    s.to_string()
                }
            })
        {
            return labels;
        }
        match (kind, index) {
            ("extr", Some(_)) => {
                let stored = se3_dvec_to_iso3(solution.params[&block.name].as_view())
                    .expect("an SE3 block holds a pose");
                let pose = camera("cam_se3_rig", &[]);
                let labels = BlockLabels::shared_pose(&pose.name).inverted(stored);
                if self.multi_camera {
                    labels
                } else {
                    labels.internal()
                }
            }
            ("handeye", None) => BlockLabels::shared_pose(&match self.handeye_mode {
                Some(HandeyeMode::EyeToHand) => format!("{rig}_se3_base"),
                _ => format!("gripper_se3_{rig}"),
            }),
            ("target" | "target_ref", None) => BlockLabels::shared_pose(target),
            ("plane_n", Some(_)) => camera(PLANE_NORMAL, &PLANE_NORMAL_LABELS),
            ("plane_d", Some(_)) => camera(PLANE_DISTANCE, &[PLANE_DISTANCE]),
            ("target" | "rig_pose", Some(view)) => BlockLabels::view_pose(view, target),
            ("robot_delta", Some(view)) => BlockLabels {
                view: Some(view),
                ..BlockLabels::shared_pose("robot_delta")
            },
            _ => panic!("unexpected rig IR block {}", block.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SolveReport;
    use crate::backend::covariance::{CovarianceEstimate, FamilyNoise};
    use crate::ir::FixedMask;
    use nalgebra::{DVector, Translation3, UnitQuaternion, Vector3};
    use vision_calibration_core::ResidualKind;

    fn ir() -> ProblemIR {
        let mut ir = ProblemIR::new();
        ir.add_param_block(
            "cam",
            4,
            ManifoldKind::Euclidean,
            FixedMask::all_free(),
            None,
        );
        ir.add_param_block(
            "dist",
            5,
            ManifoldKind::Euclidean,
            FixedMask::fix_indices(&[2]),
            None,
        );
        ir.add_param_block(
            "pose/0",
            7,
            ManifoldKind::SE3,
            FixedMask::all_fixed(7),
            None,
        );
        ir.add_param_block("pose/1", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        ir
    }

    fn labels(ir: &ProblemIR) -> Vec<BlockLabels> {
        ir.params
            .iter()
            .map(|b| {
                planar_block_labels(
                    b,
                    CameraModelSpec::pinhole(
                        crate::ir::DistortionKind::BrownConrady5,
                        crate::ir::SensorKind::Identity,
                    ),
                )
            })
            .collect()
    }

    fn estimate(coordinates: Vec<Coordinate>, matrix: DMatrix<f64>) -> ParamUncertainty {
        let n = coordinates.len();
        ParamUncertainty::Available(Box::new(CovarianceEstimate {
            coordinates,
            matrix,
            families: vec![FamilyNoise {
                family: ResidualKind::Reprojection,
                num_residuals: 100,
                sigma: 0.25,
            }],
            num_residuals: 100,
            dof: 100 - n,
        }))
    }

    #[test]
    fn splits_shared_and_per_view_coordinates() {
        let ir = ir();
        let mut coordinates: Vec<Coordinate> =
            (0..4).map(|index| Coordinate { block: 0, index }).collect();
        coordinates.extend([0, 1, 3, 4].map(|index| Coordinate { block: 1, index }));
        coordinates.extend((0..6).map(|index| Coordinate { block: 3, index }));
        let n = coordinates.len();
        let matrix = DMatrix::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let Uncertainty::Available(c) = labelled(&ir, &estimate(coordinates, matrix), &labels(&ir))
        else {
            panic!("available");
        };
        assert_eq!(c.labels, ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]);
        assert_eq!(c.std_dev("p2"), Some(8f64.sqrt()));
        assert_eq!(c.views.len(), 1);
        assert_eq!(c.views[0].view, 1);
        assert_eq!(c.views[0].labels[5], "camera_se3_target.tz");
        assert_eq!(c.views[0].covariance[5][5], 14.0);
        assert_eq!(c.fixed, ["k3", "view0.camera_se3_target"]);
        assert_eq!(c.noise[0].sigma, 0.25);
    }

    #[test]
    fn unavailable_reasons_carry_labels() {
        let ir = ir();
        let solve = ParamUncertainty::Unavailable(UnavailableReason::RankDeficient {
            unobservable: vec![
                Coordinate { block: 0, index: 2 },
                Coordinate { block: 3, index: 5 },
            ],
        });
        assert_eq!(
            labelled(&ir, &solve, &labels(&ir)),
            Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::RankDeficient {
                    unobservable: vec!["cx".into(), "view1.camera_se3_target.tz".into()],
                }
            }
        );
    }

    /// The covariance of an inverted pose is that of the inverse's
    /// right-perturbation coordinates: pushing samples `T·Exp(δ)` through
    /// the inversion and taking `log(T⁻¹⁻¹·(T·Exp(δ))⁻¹)` gives `−Ad_T δ`.
    #[test]
    fn inverted_pose_covariance_matches_the_mapped_perturbation() {
        let stored = Iso3::from_parts(
            Translation3::new(0.12, -0.03, 0.4),
            UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3),
        );
        let mut ir = ProblemIR::new();
        ir.add_param_block("extr/1", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        let coordinates: Vec<Coordinate> =
            (0..6).map(|index| Coordinate { block: 0, index }).collect();
        // One direction at a time: Σ = e_k e_kᵀ maps to column k of −Ad_T.
        for k in 0..6 {
            let matrix = DMatrix::from_fn(6, 6, |i, j| if i == k && j == k { 1.0 } else { 0.0 });
            let labels = vec![BlockLabels::shared_pose("cam1.cam_se3_rig").inverted(stored)];
            let Uncertainty::Available(c) =
                labelled(&ir, &estimate(coordinates.clone(), matrix), &labels)
            else {
                panic!("available");
            };
            // Finite-difference image of a small step along e_k.
            let h = 1e-6;
            let mut delta = [0.0; 6];
            delta[k] = h;
            let step = Iso3::from_parts(
                Translation3::from(Vector3::new(delta[3], delta[4], delta[5])),
                UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2])),
            );
            let moved = (stored * step).inverse();
            let rel = stored * moved; // (T⁻¹)⁻¹ · T'⁻¹
            let w = rel.rotation.scaled_axis() / h;
            let v = rel.translation.vector / h;
            let column = [w.x, w.y, w.z, v.x, v.y, v.z];
            for i in 0..6 {
                for j in 0..6 {
                    let expected = column[i] * column[j];
                    assert!(
                        (c.covariance[i][j] - expected).abs() < 1e-5,
                        "k={k} ({i},{j}): {} vs {expected}",
                        c.covariance[i][j]
                    );
                }
            }
            assert_eq!(c.labels[0], "cam1.cam_se3_rig.rx");
        }
    }

    #[test]
    fn fully_fixed_poses_are_named_and_internal_blocks_omitted() {
        let mut ir = ProblemIR::new();
        ir.add_param_block(
            "extr/0",
            7,
            ManifoldKind::SE3,
            FixedMask::all_fixed(7),
            None,
        );
        ir.add_param_block(
            "handeye",
            7,
            ManifoldKind::SE3,
            FixedMask::all_fixed(7),
            None,
        );
        let solve = estimate(Vec::new(), DMatrix::zeros(0, 0));
        let labels = vec![
            BlockLabels::shared_pose("cam_se3_rig").internal(),
            BlockLabels::shared_pose("gripper_se3_camera"),
        ];
        let Uncertainty::Available(c) = labelled(&ir, &solve, &labels) else {
            panic!("available");
        };
        assert_eq!(c.fixed, ["gripper_se3_camera"]);
    }

    #[test]
    fn laserline_rig_blocks_are_named_per_camera() {
        let mut ir = ProblemIR::new();
        let free = FixedMask::all_free;
        ir.add_param_block("extr/1", 7, ManifoldKind::SE3, free(), None);
        ir.add_param_block("handeye", 7, ManifoldKind::SE3, free(), None);
        ir.add_param_block("target_ref", 7, ManifoldKind::SE3, free(), None);
        ir.add_param_block("plane_n/1", 3, ManifoldKind::S2, free(), None);
        ir.add_param_block("plane_d/1", 1, ManifoldKind::Euclidean, free(), None);
        let solution = BackendSolution {
            params: [(
                "extr/1".to_string(),
                DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0]),
            )]
            .into(),
            solve_report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            uncertainty: ParamUncertainty::Unavailable(UnavailableReason::NotConverged),
        };
        let labels = RigLabelling {
            cameras: &[],
            multi_camera: true,
            handeye_mode: Some(HandeyeMode::EyeToHand),
        }
        .labels(&ir, &solution);
        let names: Vec<_> = labels.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "cam1.cam_se3_rig",
                "rig_se3_base",
                "gripper_se3_target",
                "cam1.plane.normal",
                "cam1.plane.d"
            ]
        );
        assert_eq!(
            labels[3].coordinates,
            ["cam1.plane.nx", "cam1.plane.ny", "cam1.plane.nz"]
        );
        assert_eq!(labels[4].coordinates, ["cam1.plane.d"]);
    }

    #[test]
    fn cameras_solved_apart_merge_block_diagonally() {
        let part = |sigma: f64| {
            Uncertainty::Available(ParameterCovariance {
                labels: vec!["plane.d".into()],
                covariance: vec![vec![sigma * sigma]],
                views: Vec::new(),
                fixed: vec!["fx".into(), "view0.camera_se3_target".into()],
                noise: vec![ResidualNoise {
                    kind: ResidualKind::LaserLine,
                    sigma,
                    count: 50,
                    camera: None,
                }],
                num_residuals: 50,
                dof: 49,
            })
        };
        let Uncertainty::Available(c) =
            per_camera(&[part(1.0), part(2.0)], vec!["view0.rig_se3_target".into()])
        else {
            panic!("available");
        };
        assert_eq!(c.labels, ["cam0.plane.d", "cam1.plane.d"]);
        assert_eq!(c.covariance, [[1.0, 0.0], [0.0, 4.0]]);
        assert_eq!(c.fixed, ["cam0.fx", "cam1.fx", "view0.rig_se3_target"]);
        assert_eq!(c.noise[1].camera, Some(1));
        assert_eq!((c.num_residuals, c.dof), (100, 98));

        let rank_deficient = Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::RankDeficient {
                unobservable: vec!["plane.nx".into()],
            },
        };
        assert_eq!(
            per_camera(&[part(1.0), rank_deficient], Vec::new()),
            Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::RankDeficient {
                    unobservable: vec!["cam1.plane.nx".into()]
                }
            }
        );
    }
}

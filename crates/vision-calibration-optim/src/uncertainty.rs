//! A solve's covariance in the public terms of [`Uncertainty`]: parameter
//! labels instead of IR blocks and coordinates, shared parameters apart
//! from per-view ones.

use std::collections::BTreeMap;

use vision_calibration_core::{
    ParameterCovariance, ResidualNoise, Uncertainty, UncertaintyUnavailable, ViewCovariance,
};

use crate::backend::covariance::{Coordinate, ParamUncertainty, UnavailableReason};
use crate::ir::{ManifoldKind, ParamBlock, ProblemIR};

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
}

impl BlockLabels {
    /// A shared block whose coordinates have the given labels.
    pub(crate) fn shared(name: &str, coordinates: &[&str]) -> Self {
        Self {
            view: None,
            name: name.to_string(),
            coordinates: coordinates.iter().map(|c| c.to_string()).collect(),
        }
    }

    /// A per-view SE3 pose, labelled `<name>.rx … <name>.tz`.
    pub(crate) fn view_pose(view: usize, name: &str) -> Self {
        Self {
            view: Some(view),
            name: name.to_string(),
            coordinates: pose_labels(name),
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
                        unobservable: unobservable.iter().map(label).collect(),
                    }
                }
                UnavailableReason::AtBound { coordinates } => UncertaintyUnavailable::AtBound {
                    labels: coordinates.iter().map(label).collect(),
                },
            };
            return Uncertainty::Unavailable { reason };
        }
    };

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
            .map(|&a| positions.iter().map(|&b| estimate.matrix[(a, b)]).collect())
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
            })
            .collect(),
        num_residuals: estimate.num_residuals,
        dof: estimate.dof,
    })
}

/// A coordinate's label, with `view<i>.` in front of a per-view one.
fn coordinate_label(labels: &BlockLabels, index: usize) -> String {
    match labels.view {
        None => labels.coordinates[index].clone(),
        Some(view) => format!("view{view}.{}", labels.coordinates[index]),
    }
}

/// Labels of the fixed coordinates: each one of a shared block, a fully
/// fixed per-view block by its name alone (`view3.camera_se3_target`).
fn fixed_labels(ir: &ProblemIR, labels: &[BlockLabels]) -> Vec<String> {
    let mut out = Vec::new();
    for (block, names) in ir.params.iter().zip(labels) {
        let all_fixed = block.fixed.is_all_fixed(block.dim);
        match (names.view, all_fixed) {
            (Some(view), true) => out.push(format!("view{view}.{}", names.name)),
            (_, true) => {
                out.extend((0..names.coordinates.len()).map(|i| coordinate_label(names, i)))
            }
            (_, false) if block.manifold == ManifoldKind::Euclidean => out.extend(
                (0..block.dim)
                    .filter(|&i| block.fixed.contains(i))
                    .map(|i| coordinate_label(names, i)),
            ),
            (_, false) => {}
        }
    }
    out
}

/// Labels for the blocks of a single-camera planar IR (`cam`, `dist`,
/// `sensor`, `pose/<view>`), with distortion coefficients named by
/// `distortion`.
pub(crate) fn planar_block_labels(block: &ParamBlock, distortion: &[&str]) -> BlockLabels {
    match block.name.as_str() {
        "cam" => BlockLabels::shared("intrinsics", &["fx", "fy", "cx", "cy"]),
        "dist" => BlockLabels::shared("distortion", distortion),
        "sensor" => BlockLabels::shared("sensor", &["tilt_x", "tilt_y"]),
        name => {
            let view = name
                .strip_prefix("pose/")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("unexpected planar IR block {name}"));
            BlockLabels::view_pose(view, "camera_se3_target")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::covariance::{CovarianceEstimate, FamilyNoise};
    use crate::ir::FixedMask;
    use nalgebra::DMatrix;
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
            .map(|b| planar_block_labels(b, &["k1", "k2", "k3", "p1", "p2"]))
            .collect()
    }

    #[test]
    fn splits_shared_and_per_view_coordinates() {
        let ir = ir();
        let mut coordinates: Vec<Coordinate> =
            (0..4).map(|index| Coordinate { block: 0, index }).collect();
        coordinates.extend([0, 1, 3, 4].map(|index| Coordinate { block: 1, index }));
        coordinates.extend((0..6).map(|index| Coordinate { block: 3, index }));
        let n = coordinates.len();
        let solve = ParamUncertainty::Available(Box::new(CovarianceEstimate {
            coordinates,
            matrix: DMatrix::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 }),
            families: vec![FamilyNoise {
                family: ResidualKind::Reprojection,
                num_residuals: 100,
                sigma: 0.25,
            }],
            num_residuals: 100,
            dof: 100 - n,
        }));
        let Uncertainty::Available(c) = labelled(&ir, &solve, &labels(&ir)) else {
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
}

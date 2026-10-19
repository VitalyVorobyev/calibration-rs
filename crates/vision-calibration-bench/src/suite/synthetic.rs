//! Synthetic entries: a scene of the solver benchmark's matrix, frozen into
//! an [`ObservationSet`] and its [`Truth`].
//!
//! The scene is generated from its id alone (its seed is a hash of the id),
//! but the frozen files, not the generator, are the entry: `suite freeze`
//! writes them once and `SHA256SUMS` pins them, so a later change to the
//! generator cannot change the data an entry is checked on.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};

use super::truth::Truth;
use super::{OBSERVATION_SET_VERSION, ObservationSet, ObservedCamera, ProblemInput};
use crate::solver::metrics::CameraParamsSet;
use crate::solver::scenes::{Preset, Problem, Scene, SceneData, scene_specs};

/// The scene with id `scene_id` (as `calib-bench solver` names it, e.g.
/// `planar_intrinsics/pinhole/medium/n0.5/clean`) as observations and
/// truth.
///
/// # Errors
///
/// Fails when no scene of the full matrix has that id or it cannot be
/// built.
pub fn synthesize(scene_id: &str) -> Result<(ObservationSet, Truth)> {
    let spec = scene_specs(Preset::Full)
        .into_iter()
        .find(|spec| spec.id() == scene_id)
        .with_context(|| format!("no benchmark scene {scene_id:?}"))?;
    let scene = Scene::build(&spec)?;
    let input = match scene.data.clone() {
        SceneData::Planar { input, .. } => ProblemInput::PlanarIntrinsics(input),
        SceneData::Scheimpflug { input, .. } => ProblemInput::ScheimpflugIntrinsics(input),
        SceneData::SingleCamHandeye { input, .. } => ProblemInput::SingleCamHandeye(input),
        SceneData::Laserline { input, .. } => ProblemInput::LaserlineDevice(input),
        SceneData::Rig { input, .. } => ProblemInput::RigExtrinsics(input),
        SceneData::RigHandeye { input, .. } => ProblemInput::RigHandeye(input),
        SceneData::RigLaserline { input, .. } => ProblemInput::RigLaserlineDevice(input),
        SceneData::RigHandeyeLaserline { input, .. } => ProblemInput::RigHandeyeLaserline(input),
    };
    ensure!(
        !scene.image_sizes.is_empty(),
        "scene {scene_id:?} has no cameras"
    );
    let cameras = scene
        .image_sizes
        .iter()
        .enumerate()
        .map(|(i, &image_size)| ObservedCamera {
            id: format!("cam{i}"),
            image_size,
        })
        .collect();
    let set = ObservationSet {
        version: OBSERVATION_SET_VERSION,
        cameras,
        input,
    };
    Ok((set, truth(&scene)))
}

/// The scene's ground truth under the labels its result's covariance uses.
fn truth(scene: &Scene) -> Truth {
    let gt = &scene.truth;
    let problem = scene.spec.problem;
    let rig = matches!(
        problem,
        Problem::RigExtrinsics
            | Problem::RigHandeye
            | Problem::RigLaserlineDevice
            | Problem::RigHandeyeLaserline
    );
    // A rig result names a camera's parameters `cam<i>.…` once it has more
    // than one camera.
    let prefix = |i: usize| {
        if rig && scene.image_sizes.len() > 1 {
            format!("cam{i}.")
        } else {
            String::new()
        }
    };
    let mut parameters = BTreeMap::new();
    for (i, camera) in gt.cameras.iter().enumerate() {
        for (name, value) in camera_values(camera) {
            parameters.insert(format!("{}{name}", prefix(i)), value);
        }
    }
    for (i, plane) in gt.planes_cam.iter().enumerate() {
        let n = plane.normal.into_inner();
        for (name, value) in [("nx", n.x), ("ny", n.y), ("nz", n.z), ("d", plane.distance)] {
            parameters.insert(format!("{}plane.{name}", prefix(i)), value);
        }
    }
    let mut poses = BTreeMap::new();
    for (i, pose) in gt.cam_se3_rig.iter().enumerate() {
        poses.insert(format!("{}cam_se3_rig", prefix(i)), *pose);
    }
    if let Some(handeye) = gt.handeye {
        // The scenes are eye-in-hand.
        let name = if rig {
            "gripper_se3_rig"
        } else {
            "gripper_se3_camera"
        };
        poses.insert(name.to_string(), handeye);
    }
    if let Some(target) = gt.base_se3_target {
        poses.insert("base_se3_target".to_string(), target);
    }
    Truth { parameters, poses }
}

/// A camera's parameters by their unprefixed covariance labels.
fn camera_values(c: &CameraParamsSet) -> Vec<(&'static str, f64)> {
    let mut out = vec![
        ("fx", c.k.fx),
        ("fy", c.k.fy),
        ("cx", c.k.cx),
        ("cy", c.k.cy),
        ("k1", c.dist.k1),
        ("k2", c.dist.k2),
        ("k3", c.dist.k3),
        ("p1", c.dist.p1),
        ("p2", c.dist.p2),
    ];
    if let Some(sensor) = c.sensor {
        out.extend([("tilt_x", sensor.tilt_x), ("tilt_y", sensor.tilt_y)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rig_scene_names_each_camera() {
        let (set, truth) = synthesize("rig_handeye/pinhole/small/n0.1/clean").unwrap();
        assert_eq!(set.input.problem_name(), "rig_handeye");
        assert_eq!(set.cameras.len(), 2);
        assert_eq!(set.cameras[1].image_size, [1280, 960]);
        assert!(truth.parameters.contains_key("cam1.fx"));
        assert!(truth.poses.contains_key("cam1.cam_se3_rig"));
        assert!(truth.poses.contains_key("gripper_se3_rig"));
        assert!(truth.poses.contains_key("base_se3_target"));
    }

    #[test]
    fn a_laser_scene_names_its_plane() {
        let (_, truth) = synthesize("laserline_device/pinhole/small/n0.1/clean").unwrap();
        assert!(truth.parameters.contains_key("fx"));
        assert!(truth.parameters.contains_key("plane.d"));
    }

    #[test]
    fn an_unknown_scene_is_an_error() {
        assert!(synthesize("planar_intrinsics/pinhole/huge/n0.1/clean").is_err());
    }
}

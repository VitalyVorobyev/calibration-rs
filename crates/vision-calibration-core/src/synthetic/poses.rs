//! Deterministic pose generators and pose-error helpers.
//!
//! Everything here is geometry only: board poses in front of a camera, robot
//! stations for hand-eye scenes, rig camera layouts, and a rotation/translation
//! error metric. Seeded generators use an explicit seed and a fixed RNG, so a
//! `(count, seed)` pair always yields the same poses.

use crate::{Iso3, Real};
use nalgebra::{Rotation3, Translation3, UnitQuaternion, Vector3};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Build an isometry from roll/pitch/yaw Euler angles (radians, the
/// `nalgebra::Rotation3::from_euler_angles` convention) and a translation.
pub fn make_iso(angles: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
    let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
    Iso3::from_parts(
        Translation3::new(t.0, t.1, t.2),
        UnitQuaternion::from_rotation_matrix(&rot),
    )
}

/// Rotation and translation discrepancy between two poses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseError {
    /// Angle of the relative rotation `a.rotation⁻¹ · b.rotation`, in degrees.
    pub rot_deg: Real,
    /// Distance between the two translation vectors, in the pose's length unit.
    pub trans: Real,
}

impl PoseError {
    /// The rotation error in radians.
    pub fn rot_rad(&self) -> Real {
        self.rot_deg.to_radians()
    }
}

/// Discrepancy between two poses: rotation angle of the relative rotation and
/// the distance between the translation vectors.
pub fn pose_error(a: &Iso3, b: &Iso3) -> PoseError {
    let trans = (a.translation.vector - b.translation.vector).norm();
    let rot_deg = (a.rotation.inverse() * b.rotation).angle().to_degrees();
    PoseError { rot_deg, trans }
}

/// Board poses (`cam_se3_target`) from explicit `(pitch, yaw)` tilts.
///
/// Pose `i` rotates the board by `Rotation3::from_euler_angles(pitch_i, yaw_i, 0)`
/// and translates it to `(-origin_xy.0, -origin_xy.1, z_start + z_step · i)`.
/// Passing half the board extent as `origin_xy` centres a board whose origin
/// is a corner on the optical axis.
pub fn tilted_board_poses(
    tilts: &[(Real, Real)],
    origin_xy: (Real, Real),
    z_start: Real,
    z_step: Real,
) -> Vec<Iso3> {
    tilts
        .iter()
        .enumerate()
        .map(|(i, &(pitch, yaw))| {
            Iso3::from_parts(
                Translation3::new(-origin_xy.0, -origin_xy.1, z_start + z_step * i as Real),
                Rotation3::from_euler_angles(pitch, yaw, 0.0).into(),
            )
        })
        .collect()
}

/// `n` seeded `(pitch, yaw)` tilts, each uniform in `[-max_abs_rad, max_abs_rad]`.
///
/// The first tilt is always `(0, 0)` so the sequence starts fronto-parallel.
pub fn seeded_tilts(n: usize, max_abs_rad: Real, seed: u64) -> Vec<(Real, Real)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            if i == 0 {
                (0.0, 0.0)
            } else {
                (
                    rng.random_range(-max_abs_rad..=max_abs_rad),
                    rng.random_range(-max_abs_rad..=max_abs_rad),
                )
            }
        })
        .collect()
}

/// Orientation and depth offset of one board pose for
/// [`centered_board_poses`]: Euler angles in radians (the
/// `nalgebra::Rotation3::from_euler_angles` convention) and a depth offset in
/// the scene's length unit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoardPoseSpec {
    /// Rotation about the board's x axis, radians.
    pub pitch: Real,
    /// Rotation about the board's y axis, radians.
    pub yaw: Real,
    /// Rotation about the board's z axis, radians.
    pub roll: Real,
    /// Depth offset added to the nominal distance.
    pub dz: Real,
}

impl BoardPoseSpec {
    /// A spec from its four components.
    pub const fn new(pitch: Real, yaw: Real, roll: Real, dz: Real) -> Self {
        Self {
            pitch,
            yaw,
            roll,
            dz,
        }
    }
}

/// Board poses whose *centre* sits on the optical axis.
///
/// Each spec rotates the board by its Euler angles and places the point
/// `board_center` (in target coordinates, `z = 0`) at depth `z0 + dz` on the
/// optical axis, for every rotation. Roll diversity keeps a laser stripe from
/// lying on a single line across views.
pub fn centered_board_poses(
    specs: &[BoardPoseSpec],
    board_center: (Real, Real),
    z0: Real,
) -> Vec<Iso3> {
    specs
        .iter()
        .map(|s| {
            let r = Rotation3::from_euler_angles(s.pitch, s.yaw, s.roll);
            let center_local = Vector3::new(board_center.0, board_center.1, 0.0);
            let t = Vector3::new(0.0, 0.0, z0 + s.dz) - r * center_local;
            Iso3::from_parts(Translation3::from(t), r.into())
        })
        .collect()
}

/// `n` seeded specs for [`centered_board_poses`]: pitch and yaw uniform in
/// `[-max_tilt_rad, max_tilt_rad]`, roll within ±0.6 rad, depth offset within
/// ±0.015 (15 mm for a scene in metres). The first spec is the identity.
pub fn seeded_board_pose_specs(n: usize, max_tilt_rad: Real, seed: u64) -> Vec<BoardPoseSpec> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            if i == 0 {
                BoardPoseSpec::default()
            } else {
                BoardPoseSpec {
                    pitch: rng.random_range(-max_tilt_rad..=max_tilt_rad),
                    yaw: rng.random_range(-max_tilt_rad..=max_tilt_rad),
                    roll: rng.random_range(-0.6..=0.6),
                    dz: rng.random_range(-0.015..=0.015),
                }
            }
        })
        .collect()
}

/// `(euler angles, translation)` of the nine base robot stations.
type Station = ((Real, Real, Real), (Real, Real, Real));

/// Nine robot stations with strongly diverse rotation axes (roll, pitch, yaw
/// and mixes). Hand-eye identifiability needs at least two relative motions
/// with non-parallel rotation axes; this set supplies them.
const BASE_STATIONS: [Station; 9] = [
    ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00)),
    ((0.30, 0.00, 0.00), (0.10, 0.00, 0.00)),
    ((0.00, 0.30, 0.00), (0.00, 0.10, 0.00)),
    ((0.00, 0.00, 0.30), (0.00, 0.00, 0.08)),
    ((0.22, 0.20, 0.00), (0.05, -0.05, 0.02)),
    ((-0.24, 0.00, 0.20), (-0.05, 0.05, 0.03)),
    ((0.16, -0.18, 0.12), (0.02, -0.04, 0.05)),
    ((-0.15, 0.22, -0.10), (-0.03, 0.03, 0.04)),
    ((0.20, -0.10, 0.24), (0.04, 0.02, 0.01)),
];

/// `n` robot stations (`base_se3_gripper`) with diverse rotation axes.
///
/// The first nine stations are a fixed table; station `i ≥ 9` repeats
/// station `i mod 9` with a seeded jitter of up to ±0.05 rad and ±20 mm, so
/// the set stays well conditioned for any `n`.
pub fn robot_stations(n: usize, seed: u64) -> Vec<Iso3> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let (a, t) = BASE_STATIONS[i % BASE_STATIONS.len()];
            if i < BASE_STATIONS.len() {
                return make_iso(a, t);
            }
            let mut j = |amp: Real| rng.random_range(-amp..=amp);
            make_iso(
                (a.0 + j(0.05), a.1 + j(0.05), a.2 + j(0.05)),
                (t.0 + j(0.02), t.1 + j(0.02), t.2 + j(0.02)),
            )
        })
        .collect()
}

/// Camera layout for an `n_cameras` rig as `cam_se3_rig` (`T_C_R`).
///
/// Camera 0 defines the rig frame (identity). Camera `i` sits at
/// `(baseline · i, 0, 0)` in the rig frame and is yawed about +Y by
/// `-toe_in_rad · i`, so a positive `toe_in_rad` converges the optical axes.
pub fn rig_layout(n_cameras: usize, baseline: Real, toe_in_rad: Real) -> Vec<Iso3> {
    (0..n_cameras)
        .map(|i| {
            let rig_se3_cam = Iso3::from_parts(
                Translation3::new(baseline * i as Real, 0.0, 0.0),
                UnitQuaternion::from_scaled_axis(Vector3::y() * (-toe_in_rad * i as Real)),
            );
            rig_se3_cam.inverse()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_error_is_zero_for_equal_poses_and_measures_rotation() {
        let a = make_iso((0.1, -0.2, 0.3), (0.1, 0.2, 0.3));
        let e = pose_error(&a, &a);
        assert!(e.rot_deg < 1e-9 && e.trans < 1e-12);

        let b = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let c = make_iso((0.0, 0.0, 0.1), (0.0, 0.0, 0.5));
        let e = pose_error(&b, &c);
        assert!((e.rot_deg - 0.1_f64.to_degrees()).abs() < 1e-9);
        assert!((e.trans - 0.5).abs() < 1e-12);
        assert!((e.rot_rad() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn tilted_board_poses_follow_the_ramp() {
        let poses = tilted_board_poses(&[(0.0, 0.0), (0.1, -0.1)], (0.1, 0.05), 0.5, 0.05);
        assert_eq!(poses.len(), 2);
        assert!((poses[1].translation.vector - Vector3::new(-0.1, -0.05, 0.55)).norm() < 1e-12);
    }

    #[test]
    fn seeded_generators_are_deterministic_and_seed_sensitive() {
        assert_eq!(seeded_tilts(6, 0.2, 7), seeded_tilts(6, 0.2, 7));
        assert_ne!(seeded_tilts(6, 0.2, 7), seeded_tilts(6, 0.2, 8));
        assert_eq!(seeded_tilts(3, 0.2, 7)[0], (0.0, 0.0));
        assert!(seeded_tilts(50, 0.2, 1).iter().all(|t| t.0.abs() <= 0.2));
        assert_eq!(
            seeded_board_pose_specs(5, 0.3, 3),
            seeded_board_pose_specs(5, 0.3, 3)
        );
        assert!(
            seeded_board_pose_specs(50, 0.3, 1)
                .iter()
                .all(|s| s.pitch.abs() <= 0.3 && s.yaw.abs() <= 0.3)
        );
        let a = robot_stations(20, 5);
        let b = robot_stations(20, 5);
        assert!(a.iter().zip(&b).all(|(x, y)| x == y));
    }

    #[test]
    fn robot_stations_prefix_is_the_fixed_table() {
        let a = robot_stations(9, 1);
        let b = robot_stations(20, 99);
        assert!(a.iter().zip(&b).all(|(x, y)| x == y));
        assert_eq!(b.len(), 20);
    }

    #[test]
    fn centered_board_poses_put_the_centre_on_the_axis() {
        let specs = [BoardPoseSpec::new(0.1, -0.05, 0.4, 0.01)];
        let pose = centered_board_poses(&specs, (0.12, 0.09), 0.5)[0];
        let centre = pose.transform_point(&nalgebra::Point3::new(0.12, 0.09, 0.0));
        assert!(centre.x.abs() < 1e-12 && centre.y.abs() < 1e-12);
        assert!((centre.z - 0.51).abs() < 1e-12);
    }

    #[test]
    fn rig_layout_places_cameras_along_x() {
        let layout = rig_layout(3, 0.1, 0.02);
        assert_eq!(layout.len(), 3);
        assert!(layout[0].translation.vector.norm() < 1e-15);
        // Camera 2's origin, expressed in the rig frame, is (0.2, 0, 0).
        let origin = layout[2].inverse().translation.vector;
        assert!((origin - Vector3::new(0.2, 0.0, 0.0)).norm() < 1e-12);
    }
}

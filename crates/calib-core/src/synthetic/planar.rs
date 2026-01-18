//! Synthetic planar target helpers.
//!
//! The functions here build simple planar point grids (Z=0), generate simple
//! camera poses, and project the target into the camera to produce
//! [`crate::CorrespondenceView`] instances.

use crate::{
    models::{DistortionModel, IntrinsicsModel, ProjectionModel, SensorModel},
    Camera, CorrespondenceView, Iso3, Pt2, Pt3, Real,
};
use anyhow::Result;
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use std::ops::RangeInclusive;

/// Generate a planar grid of 3D points (Z=0) with `nx * ny` points.
///
/// Points are ordered deterministically in row-major order (Y major):
/// `(x = 0..nx-1, y = 0..ny-1)`.
pub fn grid_points(nx: usize, ny: usize, spacing: Real) -> Vec<Pt3> {
    grid_points_range(
        0..=(nx.saturating_sub(1) as i32),
        0..=(ny.saturating_sub(1) as i32),
        spacing,
    )
}

/// Generate a planar grid of 2D points with `nx * ny` points.
///
/// Points are ordered deterministically in row-major order (Y major):
/// `(x = 0..nx-1, y = 0..ny-1)`.
pub fn grid_points_2d(nx: usize, ny: usize, spacing: Real) -> Vec<Pt2> {
    grid_points_range_2d(
        0..=(nx.saturating_sub(1) as i32),
        0..=(ny.saturating_sub(1) as i32),
        spacing,
    )
}

/// Generate a planar grid of 3D points (Z=0) over integer index ranges.
///
/// The output order is deterministic in row-major order (Y major).
pub fn grid_points_range(
    x: RangeInclusive<i32>,
    y: RangeInclusive<i32>,
    spacing: Real,
) -> Vec<Pt3> {
    let nx = (*x.end() as i64 - *x.start() as i64 + 1).max(0) as usize;
    let ny = (*y.end() as i64 - *y.start() as i64 + 1).max(0) as usize;
    let mut points = Vec::with_capacity(nx.saturating_mul(ny));

    for j in y {
        for i in x.clone() {
            points.push(Pt3::new(i as Real * spacing, j as Real * spacing, 0.0));
        }
    }
    points
}

/// Generate a planar grid of 2D points over integer index ranges.
///
/// The output order is deterministic in row-major order (Y major).
pub fn grid_points_range_2d(
    x: RangeInclusive<i32>,
    y: RangeInclusive<i32>,
    spacing: Real,
) -> Vec<Pt2> {
    let nx = (*x.end() as i64 - *x.start() as i64 + 1).max(0) as usize;
    let ny = (*y.end() as i64 - *y.start() as i64 + 1).max(0) as usize;
    let mut points = Vec::with_capacity(nx.saturating_mul(ny));

    for j in y {
        for i in x.clone() {
            points.push(Pt2::new(i as Real * spacing, j as Real * spacing));
        }
    }
    points
}

/// Generate `n_views` poses with a yaw rotation around the +Y axis and a Z translation ramp.
///
/// This is a convenient default for planar targets: the board stays in front of the camera
/// while varying viewpoint.
pub fn poses_yaw_y_z(
    n_views: usize,
    yaw_start_rad: Real,
    yaw_step_rad: Real,
    z_start: Real,
    z_step: Real,
) -> Vec<Iso3> {
    (0..n_views)
        .map(|view_idx| {
            let yaw = yaw_start_rad + yaw_step_rad * view_idx as Real;
            let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 1.0, 0.0) * yaw);
            let translation = Vector3::new(0.0, 0.0, z_start + z_step * view_idx as Real);
            Iso3::from_parts(Translation3::from(translation), rotation)
        })
        .collect()
}

/// Project a planar target into the camera, requiring every point to be projectable.
///
/// `cam_from_target` must map target-frame points into the camera frame.
pub fn project_view_all<P, D, Sm, K>(
    camera: &Camera<Real, P, D, Sm, K>,
    cam_from_target: &Iso3,
    target_points: &[Pt3],
) -> Result<CorrespondenceView>
where
    P: ProjectionModel<Real>,
    D: DistortionModel<Real>,
    Sm: SensorModel<Real>,
    K: IntrinsicsModel<Real>,
{
    let mut pixels = Vec::with_capacity(target_points.len());
    for (idx, pw) in target_points.iter().enumerate() {
        let pc = cam_from_target.transform_point(pw);
        let Some(uv) = camera.project_point(&pc) else {
            anyhow::bail!("point {idx} not projectable (z={:.6})", pc.z);
        };
        pixels.push(uv);
    }

    CorrespondenceView::new(target_points.to_vec(), pixels)
}

/// Project a planar target into the camera, keeping only projectable points.
///
/// This is convenient for tests where the pose may place some points behind the camera.
pub fn project_view_visible<P, D, Sm, K>(
    camera: &Camera<Real, P, D, Sm, K>,
    cam_from_target: &Iso3,
    target_points: &[Pt3],
) -> CorrespondenceView
where
    P: ProjectionModel<Real>,
    D: DistortionModel<Real>,
    Sm: SensorModel<Real>,
    K: IntrinsicsModel<Real>,
{
    let mut points_3d = Vec::with_capacity(target_points.len());
    let mut points_2d = Vec::with_capacity(target_points.len());

    for pw in target_points {
        let pc = cam_from_target.transform_point(pw);
        let Some(uv) = camera.project_point(&pc) else {
            continue;
        };
        points_3d.push(*pw);
        points_2d.push(uv);
    }

    CorrespondenceView {
        points_3d,
        points_2d,
        weights: None,
    }
}

/// Project multiple views, requiring every point to be projectable in every view.
pub fn project_views_all<P, D, Sm, K>(
    camera: &Camera<Real, P, D, Sm, K>,
    target_points: &[Pt3],
    cam_from_target: &[Iso3],
) -> Result<Vec<CorrespondenceView>>
where
    P: ProjectionModel<Real>,
    D: DistortionModel<Real>,
    Sm: SensorModel<Real>,
    K: IntrinsicsModel<Real>,
{
    cam_from_target
        .iter()
        .map(|pose| project_view_all(camera, pose, target_points))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FxFyCxCySkew, IdentitySensor, NoDistortion, Pinhole};

    #[test]
    fn grid_points_order_is_stable() {
        let pts = grid_points(2, 3, 0.5);
        assert_eq!(pts.len(), 6);
        assert_eq!(pts[0], Pt3::new(0.0, 0.0, 0.0));
        assert_eq!(pts[1], Pt3::new(0.5, 0.0, 0.0));
        assert_eq!(pts[2], Pt3::new(0.0, 0.5, 0.0));
    }

    #[test]
    fn project_view_all_produces_matching_correspondences() {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);
        let board = grid_points(3, 2, 0.05);
        let pose = Iso3::from_parts(Translation3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());

        let view = project_view_all(&cam, &pose, &board).unwrap();
        assert_eq!(view.points_3d.len(), board.len());
        assert_eq!(view.points_2d.len(), board.len());
    }
}

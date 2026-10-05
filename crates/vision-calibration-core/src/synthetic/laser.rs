//! Synthetic laser-line stripes on a planar target.
//!
//! A laser plane intersects the planar target (`z = 0` in the target frame)
//! in a straight line. [`laser_stripe_pixels`] samples that line inside the
//! physical board and projects it through a camera model.

use crate::{
    Camera, Iso3, Pt2, Pt3, Real, Vec2, Vec3,
    models::{DistortionModel, IntrinsicsModel, ProjectionModel, SensorModel},
};

use super::noise::UniformPixelNoise;

/// Axis-aligned board extent in target coordinates (`z = 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardExtent {
    /// Minimum `(x, y)` corner.
    pub min: [Real; 2],
    /// Maximum `(x, y)` corner.
    pub max: [Real; 2],
}

impl BoardExtent {
    /// Extent of an `nx × ny` grid with `spacing`, anchored at the origin
    /// (matches [`super::planar::grid_points`]).
    pub fn from_grid(nx: usize, ny: usize, spacing: Real) -> Self {
        Self {
            min: [0.0, 0.0],
            max: [
                nx.saturating_sub(1) as Real * spacing,
                ny.saturating_sub(1) as Real * spacing,
            ],
        }
    }

    /// Centre of the board.
    pub fn center(&self) -> [Real; 2] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
        ]
    }
}

/// Number of samples along the visible stripe.
const STRIPE_SAMPLES: usize = 41;

/// Pixels of the laser stripe on a planar target for one view.
///
/// The laser plane `normal · p + distance = 0` is given in the camera frame;
/// `cam_se3_target` maps target points into the camera frame. The plane is
/// intersected with the target plane, the line is clipped to `board`, and
/// 41 points along it (inset 2 % from the ends) are projected
/// through `camera` and perturbed with `noise` (keyed on `view_idx` and a
/// point index disjoint from the corner-noise key space).
///
/// Returns an empty vector for degenerate views: the plane parallel to the
/// board, the stripe missing the board, or a stripe shorter than 0.05 (5 cm
/// for a scene in metres).
pub fn laser_stripe_pixels<P, D, Sm, K>(
    camera: &Camera<Real, P, D, Sm, K>,
    cam_se3_target: &Iso3,
    plane_normal: &Vec3,
    plane_distance: Real,
    board: &BoardExtent,
    view_idx: usize,
    noise: &UniformPixelNoise,
) -> Vec<Pt2>
where
    P: ProjectionModel<Real>,
    D: DistortionModel<Real>,
    Sm: SensorModel<Real>,
    K: IntrinsicsModel<Real>,
{
    // Laser plane expressed in the target frame (board is z = 0 there).
    let n = cam_se3_target
        .rotation
        .inverse_transform_vector(plane_normal);
    let d = plane_normal.dot(&cam_se3_target.translation.vector) + plane_distance;

    let n_xy = Vec2::new(n.x, n.y);
    let horiz = n_xy.norm();
    if horiz < 1e-9 {
        return Vec::new();
    }
    let dir = Vec2::new(-n.y, n.x) / horiz;
    // Foot of the perpendicular from the board centre onto the stripe line;
    // any point on the line serves as the clip anchor.
    let c = board.center();
    let center = Vec2::new(c[0], c[1]);
    let signed = (n_xy.dot(&center) + d) / (horiz * horiz);
    let p0 = center - signed * n_xy;

    // Clip the infinite line q(t) = p0 + t·dir to the board rectangle
    // (parametric slab clipping).
    let mut t_lo = Real::NEG_INFINITY;
    let mut t_hi = Real::INFINITY;
    for axis in 0..2 {
        let (origin, delta) = (p0[axis], dir[axis]);
        let (lo, hi) = (board.min[axis], board.max[axis]);
        if delta.abs() < 1e-12 {
            if origin < lo || origin > hi {
                return Vec::new();
            }
        } else {
            let mut ta = (lo - origin) / delta;
            let mut tb = (hi - origin) / delta;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t_lo = t_lo.max(ta);
            t_hi = t_hi.min(tb);
        }
    }
    if t_hi - t_lo < 0.05 {
        return Vec::new();
    }
    let margin = 0.02 * (t_hi - t_lo);
    let (a, b) = (t_lo + margin, t_hi - margin);

    let last = (STRIPE_SAMPLES - 1) as Real;
    let mut pixels = Vec::with_capacity(STRIPE_SAMPLES);
    for i in 0..STRIPE_SAMPLES {
        let t = a + (b - a) * (i as Real) / last;
        let p_cam =
            cam_se3_target.transform_point(&Pt3::new(p0.x + t * dir.x, p0.y + t * dir.y, 0.0));
        if let Some(px) = camera.project_point(&p_cam) {
            let noisy = noise.apply(view_idx, 10_000 + i, Vec2::new(px.x, px.y));
            pixels.push(Pt2::new(noisy.x, noisy.y));
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FxFyCxCySkew, IdentitySensor, NoDistortion, Pinhole};

    fn camera() -> Camera<Real, Pinhole, NoDistortion, IdentitySensor, FxFyCxCySkew<Real>> {
        Camera::new(
            Pinhole,
            NoDistortion,
            IdentitySensor,
            FxFyCxCySkew {
                fx: 900.0,
                fy: 900.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
        )
    }

    #[test]
    fn stripe_lies_on_the_laser_plane_and_inside_the_board() {
        let cam = camera();
        let board = BoardExtent::from_grid(9, 7, 0.03);
        let pose = crate::synthetic::poses::centered_board_poses(
            &[crate::synthetic::poses::BoardPoseSpec::new(
                0.1, -0.05, 0.3, 0.0,
            )],
            board.center().into(),
            0.5,
        )[0];
        let n = Vec3::new(0.3, 0.0, 1.0).normalize();
        let plane_d = -n.z * 0.5;
        let noise = UniformPixelNoise::default();
        let px = laser_stripe_pixels(&cam, &pose, &n, plane_d, &board, 0, &noise);
        assert_eq!(px.len(), STRIPE_SAMPLES);

        // Back-project each pixel to the target plane; it must satisfy the
        // laser plane equation and lie within the board rectangle.
        let target_se3_cam = pose.inverse();
        for p in &px {
            let ray = Vec3::new((p.x - 640.0) / 900.0, (p.y - 360.0) / 900.0, 1.0);
            // Intersect the ray with the target plane z_t = 0.
            let origin_t = target_se3_cam.translation.vector;
            let dir_t = target_se3_cam.rotation * ray;
            let s = -origin_t.z / dir_t.z;
            let hit_t = origin_t + s * dir_t;
            assert!(hit_t.x >= -1e-9 && hit_t.x <= board.max[0] + 1e-9);
            assert!(hit_t.y >= -1e-9 && hit_t.y <= board.max[1] + 1e-9);
            let hit_c = pose.transform_point(&Pt3::from(hit_t));
            assert!((n.dot(&hit_c.coords) + plane_d).abs() < 1e-9);
        }
    }

    #[test]
    fn parallel_plane_yields_no_stripe_and_noise_is_deterministic() {
        let cam = camera();
        let board = BoardExtent::from_grid(9, 7, 0.03);
        let pose = crate::synthetic::poses::make_iso((0.0, 0.0, 0.0), (-0.12, -0.09, 0.5));
        let flat = laser_stripe_pixels(
            &cam,
            &pose,
            &Vec3::new(0.0, 0.0, 1.0),
            -0.5,
            &board,
            0,
            &UniformPixelNoise::default(),
        );
        assert!(flat.is_empty());

        let noise = UniformPixelNoise {
            seed: 3,
            max_abs_px: 0.3,
        };
        let n = Vec3::new(0.3, 0.0, 1.0).normalize();
        let a = laser_stripe_pixels(&cam, &pose, &n, -0.45, &board, 2, &noise);
        let b = laser_stripe_pixels(&cam, &pose, &n, -0.45, &board, 2, &noise);
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }
}

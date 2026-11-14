//! The tilted camera, its parameterizations and the Fisher information of a
//! scene.
//!
//! The camera is the library's: pinhole, thin-prism distortion (Brown-Conrady
//! plus `s1..s4`, so every distortion term the study frees is one model), the
//! Scheimpflug sensor and `K`, evaluated through the library's generic
//! `DistortionKind::distort` and `SensorKind::to_sensor`.
//!
//! The tilt homography factors exactly as `K·H(τ) = K'(τ)·R(τ)`: a pinhole
//! `K'` (the foot of the sensor normal as its principal point, the focal
//! lengths scaled by `R₂₂`) behind a rotation `R(τ)`. The parameterizations
//! differ in which of the two they make coordinates:
//!
//! - [`Param::Raw`]: the library's `fx, fy, cx, cy` (the optical axis foot,
//!   the distortion centre) and poses in the lens frame;
//! - [`Param::TanShift`]: `c̃ = c − f·tan(τ)` per axis, poses in the lens
//!   frame (the substitution tried on a real rig);
//! - [`Param::Effective`]: `K'` exactly, poses in the lens frame;
//! - [`Param::Gauge`]: `K'` and poses in the sensor-normal frame
//!   (`P' = R·P`), so that without distortion the tilt changes nothing.

use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Point2, RealField, Rotation3, Translation3,
    UnitQuaternion, Vector3,
};
use vision_calibration_core::{DistortionKind, Iso3, SensorKind};

use crate::scene::Scene;

/// Camera coordinates: 4 intrinsics, 9 distortion terms, 2 tilts.
pub const NCAM: usize = 15;
pub const TX: usize = 13;
pub const TY: usize = 14;

/// The library's names of the 15 raw coordinates.
pub const RAW_NAMES: [&str; NCAM] = [
    "fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4", "tilt_x",
    "tilt_y",
];

/// A camera in the library's (raw) parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    /// `[fx, fy, cx, cy]`.
    pub k: [f64; 4],
    /// `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`.
    pub d: [f64; 9],
    /// `[tilt_x, tilt_y]`, radians.
    pub tilt: [f64; 2],
}

impl Cam {
    pub fn raw(&self) -> [f64; NCAM] {
        let mut v = [0.0; NCAM];
        v[..4].copy_from_slice(&self.k);
        v[4..13].copy_from_slice(&self.d);
        v[TX] = self.tilt[0];
        v[TY] = self.tilt[1];
        v
    }

    pub fn from_raw(v: &[f64]) -> Self {
        let mut c = Cam {
            k: [0.0; 4],
            d: [0.0; 9],
            tilt: [v[TX], v[TY]],
        };
        c.k.copy_from_slice(&v[..4]);
        c.d.copy_from_slice(&v[4..13]);
        c
    }
}

/// The rotation `R(τ) = Ry(τy)·Rx(τx)` of the library's tilt homography
/// (OpenCV's sensor-frame sign convention).
pub fn tilt_rotation<T: RealField>(tx: T, ty: T) -> Matrix3<T> {
    let (sx, cx) = (tx.clone().sin(), tx.cos());
    let (sy, cy) = (ty.clone().sin(), ty.cos());
    let (o, z) = (T::one(), T::zero());
    let rx = Matrix3::new(
        o.clone(),
        z.clone(),
        z.clone(),
        z.clone(),
        cx.clone(),
        sx.clone(),
        z.clone(),
        -sx,
        cx,
    );
    let ry = Matrix3::new(
        cy.clone(),
        z.clone(),
        -sy.clone(),
        z.clone(),
        o,
        z.clone(),
        sy,
        z,
        cy,
    );
    ry * rx
}

/// Pixel of a lens-frame point through the raw camera `k, d, tilt`; `None`
/// behind the camera.
pub fn project<T: RealField>(k: &[T], d: &[T], tilt: &[T], xc: &Vector3<T>) -> Option<[T; 2]> {
    if xc.z <= T::from_f64(1e-9).unwrap() {
        return None;
    }
    let n = Point2::new(xc.x.clone() / xc.z.clone(), xc.y.clone() / xc.z.clone());
    let nd = DistortionKind::ThinPrism9.distort(d, &n);
    let s = SensorKind::Scheimpflug.to_sensor(tilt, &nd);
    Some([
        k[0].clone() * s.x.clone() + k[2].clone(),
        k[1].clone() * s.y.clone() + k[3].clone(),
    ])
}

/// How the camera's intrinsics and the view poses are coordinatized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    Raw,
    TanShift,
    Effective,
    Gauge,
}

impl Param {
    pub const ALL: [Param; 4] = [Param::Raw, Param::TanShift, Param::Effective, Param::Gauge];

    pub fn name(self) -> &'static str {
        match self {
            Param::Raw => "raw (current)",
            Param::TanShift => "c̃ = c − f·tan τ",
            Param::Effective => "K' (sensor-normal pinhole)",
            Param::Gauge => "K' + poses in sensor frame",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Param::Raw => "P0",
            Param::TanShift => "P1",
            Param::Effective => "P2",
            Param::Gauge => "P3",
        }
    }

    /// The names of the four intrinsic coordinates.
    pub fn intrinsic_names(self) -> [&'static str; 4] {
        match self {
            Param::Raw => ["fx", "fy", "cx", "cy"],
            Param::TanShift => ["fx", "fy", "c~x", "c~y"],
            Param::Effective | Param::Gauge => ["fx'", "fy'", "cx'", "cy'"],
        }
    }

    pub fn coord_name(self, i: usize) -> &'static str {
        if i < 4 {
            self.intrinsic_names()[i]
        } else {
            RAW_NAMES[i]
        }
    }

    /// Raw intrinsics `[fx, fy, cx, cy]` from this parameterization's.
    pub fn raw_intrinsics<T: RealField>(self, k: &[T], tilt: &[T]) -> [T; 4] {
        let (tx, ty) = (tilt[0].clone(), tilt[1].clone());
        match self {
            Param::Raw => [k[0].clone(), k[1].clone(), k[2].clone(), k[3].clone()],
            Param::TanShift => [
                k[0].clone(),
                k[1].clone(),
                k[2].clone() - k[0].clone() * ty.tan(),
                k[3].clone() + k[1].clone() * tx.tan(),
            ],
            Param::Effective | Param::Gauge => {
                let r22 = tx.clone().cos() * ty.clone().cos();
                let fx = k[0].clone() / r22.clone();
                let fy = k[1].clone() / r22;
                let cx = k[2].clone() - fx.clone() * ty.sin() * tx.clone().cos();
                let cy = k[3].clone() + fy.clone() * tx.sin();
                [fx, fy, cx, cy]
            }
        }
    }

    /// This parameterization's intrinsics from raw ones.
    pub fn intrinsics_of_raw(self, k: &[f64; 4], tilt: &[f64; 2]) -> [f64; 4] {
        let (tx, ty) = (tilt[0], tilt[1]);
        match self {
            Param::Raw => *k,
            Param::TanShift => [k[0], k[1], k[2] + k[0] * ty.tan(), k[3] - k[1] * tx.tan()],
            Param::Effective | Param::Gauge => {
                let r22 = tx.cos() * ty.cos();
                [
                    k[0] * r22,
                    k[1] * r22,
                    k[2] + k[0] * ty.sin() * tx.cos(),
                    k[3] - k[1] * tx.sin(),
                ]
            }
        }
    }

    /// The coordinates of a raw camera.
    pub fn coords(self, cam: &Cam) -> [f64; NCAM] {
        let mut v = cam.raw();
        v[..4].copy_from_slice(&self.intrinsics_of_raw(&cam.k, &cam.tilt));
        v
    }

    /// The raw camera of coordinates `v`.
    pub fn camera(self, v: &[f64]) -> Cam {
        let k = self.raw_intrinsics(&v[..4], &v[TX..=TY]);
        let mut c = Cam::from_raw(v);
        c.k = k;
        c
    }

    /// A lens-frame pose in this parameterization's frame.
    pub fn pose_to(self, cam: &Cam, pose: &Iso3) -> Iso3 {
        match self {
            Param::Gauge => {
                let r = tilt_rotation(cam.tilt[0], cam.tilt[1]);
                let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
                Iso3::from_parts(Translation3::identity(), q) * pose
            }
            _ => *pose,
        }
    }

    /// Pixel of a target point: coordinates `v` (15), the view's pose in this
    /// parameterization's frame.
    pub fn project<T: RealField>(
        self,
        v: &[T],
        pose: &Isometry3<T>,
        pw: &Vector3<T>,
    ) -> Option<[T; 2]> {
        let tilt = &v[TX..=TY];
        let k = self.raw_intrinsics(&v[..4], tilt);
        let xp = pose.rotation.transform_vector(pw) + pose.translation.vector.clone();
        let xl = match self {
            Param::Gauge => tilt_rotation(tilt[0].clone(), tilt[1].clone()).transpose() * xp,
            _ => xp,
        };
        project(&k, &v[4..13], tilt, &xl)
    }
}

/// The Fisher information of a scene's camera coordinates, poses
/// eliminated, at one pixel of noise per coordinate.
#[derive(Clone, Debug)]
pub struct Fisher {
    /// Free camera coordinates (indices into the 15).
    pub free: Vec<usize>,
    /// Schur complement of the poses: the camera coordinates' information.
    pub info: DMatrix<f64>,
    /// Its inverse: the camera coordinates' covariance at σ = 1 px; `None`
    /// when the information is singular.
    pub cov: Option<DMatrix<f64>>,
    /// Condition number of the Jacobi-scaled full normal matrix (camera and
    /// poses), as the solver's scaled step sees it.
    pub cond_full: f64,
    /// Condition number of the Jacobi-scaled camera information.
    pub cond_cam: f64,
}

impl Fisher {
    fn pos(&self, coord: usize) -> Option<usize> {
        self.free.iter().position(|&c| c == coord)
    }

    /// Covariance of two coordinates at one pixel of noise; `None` when
    /// either is fixed or the information singular.
    pub fn cov_of(&self, a: usize, b: usize) -> Option<f64> {
        match (&self.cov, self.pos(a), self.pos(b)) {
            (Some(c), Some(i), Some(j)) => Some(c[(i, j)]),
            _ => None,
        }
    }

    /// σ of a coordinate at pixel noise `noise`; `INFINITY` when singular.
    pub fn sigma(&self, coord: usize, noise: f64) -> f64 {
        match (&self.cov, self.pos(coord)) {
            (Some(c), Some(i)) => noise * c[(i, i)].max(0.0).sqrt(),
            (None, Some(_)) => f64::INFINITY,
            _ => f64::NAN,
        }
    }

    /// Correlation of two coordinates; NaN when either is fixed or the
    /// information singular.
    pub fn corr(&self, a: usize, b: usize) -> f64 {
        match (&self.cov, self.pos(a), self.pos(b)) {
            (Some(c), Some(i), Some(j)) => c[(i, j)] / (c[(i, i)] * c[(j, j)]).sqrt(),
            _ => f64::NAN,
        }
    }

    /// How much its correlations with the other camera coordinates inflate
    /// a coordinate's σ: `σ_marginal / σ_conditional = 1/√(1 − R²)`.
    pub fn inflation(&self, coord: usize) -> f64 {
        match (&self.cov, self.pos(coord)) {
            (Some(c), Some(i)) => (c[(i, i)] * self.info[(i, i)]).sqrt(),
            (None, Some(_)) => f64::INFINITY,
            _ => f64::NAN,
        }
    }

    /// The largest |correlation| between `coord` and another free camera
    /// coordinate, with that coordinate.
    pub fn max_corr(&self, coord: usize) -> (f64, usize) {
        let mut best = (0.0, coord);
        for &o in &self.free {
            if o != coord {
                let r = self.corr(coord, o);
                if r.abs() > best.0 {
                    best = (r.abs(), o);
                }
            }
        }
        best
    }
}

/// The 6 tangent coordinates of a pose: `exp(δ)·P`, `δ = (ω, v)`.
pub fn perturb(pose: &Iso3, delta: &[f64; 6]) -> Iso3 {
    let rot = UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2]));
    Iso3::from_parts(Translation3::new(delta[3], delta[4], delta[5]), rot) * pose
}

/// The view's pixels (2 per point) at coordinates `v` and pose `pose`.
fn view_pixels(param: Param, v: &[f64], pose: &Iso3, board: &[[f64; 3]], out: &mut Vec<f64>) {
    out.clear();
    for b in board {
        let px = param
            .project(v, pose, &Vector3::new(b[0], b[1], b[2]))
            .unwrap_or([f64::NAN, f64::NAN]);
        out.extend_from_slice(&px);
    }
}

/// Central-difference step of each camera coordinate.
fn camera_step(i: usize, v: &[f64]) -> f64 {
    match i {
        0..=3 => 1e-6 * v[0].abs().max(1.0),
        _ => 1e-7,
    }
}

/// Condition number of a symmetric positive semi-definite matrix after
/// Jacobi scaling; `INFINITY` when singular to working precision.
pub fn scaled_cond(m: &DMatrix<f64>) -> f64 {
    let n = m.nrows();
    let d: Vec<f64> = (0..n).map(|i| 1.0 / m[(i, i)].max(1e-300).sqrt()).collect();
    let s = DMatrix::from_fn(n, n, |i, j| m[(i, j)] * d[i] * d[j]);
    let eig = s.symmetric_eigen();
    let max = eig.eigenvalues.max();
    let min = eig.eigenvalues.min();
    // The central-difference Jacobian carries relative errors near 1e-10,
    // so a smaller eigenvalue ratio is a singular matrix.
    if min <= max * 1e-11 {
        f64::INFINITY
    } else {
        max / min
    }
}

/// The Fisher information of `scene` in `param`, with the camera
/// coordinates `free` estimated and every pose free.
pub fn fisher(scene: &Scene, cam: &Cam, poses: &[Iso3], param: Param, free: &[usize]) -> Fisher {
    let v0 = param.coords(cam);
    let nc = free.len();
    let nv = scene.views.len();
    let n = nc + 6 * nv;
    let mut full = DMatrix::<f64>::zeros(n, n);
    let mut info = DMatrix::<f64>::zeros(nc, nc);
    let (mut plus, mut minus) = (Vec::new(), Vec::new());
    for (vi, view) in scene.views.iter().enumerate() {
        let pose = param.pose_to(cam, &poses[vi]);
        let rows = 2 * view.board.len();
        let mut jc = DMatrix::<f64>::zeros(rows, nc);
        let mut jp = DMatrix::<f64>::zeros(rows, 6);
        for (c, &coord) in free.iter().enumerate() {
            let h = camera_step(coord, &v0);
            let mut vp = v0;
            vp[coord] += h;
            view_pixels(param, &vp, &pose, &view.board, &mut plus);
            vp[coord] -= 2.0 * h;
            view_pixels(param, &vp, &pose, &view.board, &mut minus);
            for r in 0..rows {
                jc[(r, c)] = (plus[r] - minus[r]) / (2.0 * h);
            }
        }
        let depth = pose.translation.vector.norm().max(1e-3);
        for k in 0..6 {
            let h = if k < 3 { 1e-7 } else { 1e-7 * depth };
            let mut d = [0.0; 6];
            d[k] = h;
            view_pixels(param, &v0, &perturb(&pose, &d), &view.board, &mut plus);
            d[k] = -h;
            view_pixels(param, &v0, &perturb(&pose, &d), &view.board, &mut minus);
            for r in 0..rows {
                jp[(r, k)] = (plus[r] - minus[r]) / (2.0 * h);
            }
        }
        let ncc = jc.transpose() * &jc;
        let ncp = jc.transpose() * &jp;
        let npp = jp.transpose() * &jp;
        let o = nc + 6 * vi;
        for i in 0..nc {
            for j in 0..nc {
                full[(i, j)] += ncc[(i, j)];
            }
        }
        full.view_mut((0, o), (nc, 6)).copy_from(&ncp);
        full.view_mut((o, 0), (6, nc)).copy_from(&ncp.transpose());
        full.view_mut((o, o), (6, 6)).copy_from(&npp);
        let npp_inv = npp
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::zeros(6, 6));
        info += ncc - &ncp * npp_inv * ncp.transpose();
    }
    let info = 0.5 * (&info + info.transpose());
    let cond_cam = scaled_cond(&info);
    let cov = if cond_cam.is_finite() {
        info.clone().try_inverse()
    } else {
        None
    };
    Fisher {
        free: free.to_vec(),
        info,
        cov,
        cond_full: scaled_cond(&full),
        cond_cam,
    }
}

/// Free coordinate sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeSet {
    /// The library default under a tilt: `K, k1, k2, τ`.
    Default,
    /// Default plus `p1, p2`.
    Tangential,
    /// Default plus `s1..s4`.
    Prism,
    /// Default plus `p1, p2, s1..s4`.
    TangentialPrism,
    /// Default plus `k3`.
    K3,
    /// Default without the tilt (`τ` held at its true value).
    NoTilt,
    /// Tangential without the tilt.
    TangentialNoTilt,
    /// Thin prism without the tilt.
    PrismNoTilt,
}

impl FreeSet {
    pub fn indices(self) -> Vec<usize> {
        let base = [0, 1, 2, 3, 4, 5];
        let tilt = [TX, TY];
        let mut v: Vec<usize> = base.to_vec();
        match self {
            FreeSet::Default | FreeSet::NoTilt => {}
            FreeSet::Tangential | FreeSet::TangentialNoTilt => v.extend([7, 8]),
            FreeSet::Prism | FreeSet::PrismNoTilt => v.extend([9, 10, 11, 12]),
            FreeSet::TangentialPrism => v.extend([7, 8, 9, 10, 11, 12]),
            FreeSet::K3 => v.push(6),
        }
        if !matches!(
            self,
            FreeSet::NoTilt | FreeSet::TangentialNoTilt | FreeSet::PrismNoTilt
        ) {
            v.extend(tilt);
        }
        v
    }

    pub fn name(self) -> &'static str {
        match self {
            FreeSet::Default => "default (K, k1, k2, τ)",
            FreeSet::Tangential => "+ p1, p2",
            FreeSet::Prism => "+ s1..s4",
            FreeSet::TangentialPrism => "+ p1, p2, s1..s4",
            FreeSet::K3 => "+ k3",
            FreeSet::NoTilt => "τ held",
            FreeSet::TangentialNoTilt => "τ held, + p1, p2",
            FreeSet::PrismNoTilt => "τ held, + s1..s4",
        }
    }
}

/// Max |pixel difference| between `K·H(τ)` and `K'(τ)·R(τ)` applied to
/// distorted normalized points over a grid: the exact factorization.
pub fn factorization_error(cam: &Cam) -> f64 {
    let kp = Param::Effective.intrinsics_of_raw(&cam.k, &cam.tilt);
    let r = tilt_rotation(cam.tilt[0], cam.tilt[1]);
    let mut worst: f64 = 0.0;
    for i in -10..=10 {
        for j in -10..=10 {
            let n = Vector3::new(0.08 * i as f64, 0.06 * j as f64, 1.0);
            let a = project(&cam.k, &[0.0; 9], &cam.tilt, &n).expect("in front");
            let q = r * n;
            let b = [kp[0] * q.x / q.z + kp[2], kp[1] * q.y / q.z + kp[3]];
            worst = worst.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs());
        }
    }
    worst
}

/// The camera coordinates' columns at the truth over the scene, as column
/// norms (pixels per unit), for the scaling diagnostic.
pub fn column_norms(scene: &Scene, cam: &Cam, poses: &[Iso3], param: Param) -> DVector<f64> {
    let v0 = param.coords(cam);
    let mut sums = DVector::<f64>::zeros(NCAM);
    let (mut plus, mut minus) = (Vec::new(), Vec::new());
    for (vi, view) in scene.views.iter().enumerate() {
        let pose = param.pose_to(cam, &poses[vi]);
        for (c, s) in sums.iter_mut().enumerate() {
            let h = camera_step(c, &v0);
            let mut vp = v0;
            vp[c] += h;
            view_pixels(param, &vp, &pose, &view.board, &mut plus);
            vp[c] -= 2.0 * h;
            view_pixels(param, &vp, &pose, &view.board, &mut minus);
            *s += plus
                .iter()
                .zip(&minus)
                .map(|(a, b)| ((a - b) / (2.0 * h)).powi(2))
                .sum::<f64>();
        }
    }
    sums.map(f64::sqrt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{CamSpec, ViewDesign, build_scene, camera_params};
    use nalgebra::Point3;

    fn spec(tilt: (f64, f64), dist: f64) -> CamSpec {
        CamSpec {
            tilt_deg: tilt,
            corner_distortion: dist,
            ..CamSpec::baseline()
        }
    }

    /// The generic projection is the library camera's.
    #[test]
    fn projection_matches_the_library_camera() {
        let mut cam = spec((7.0, -4.0), -0.08).camera();
        cam.d[1] = 0.03;
        cam.d[3] = 4e-4;
        cam.d[4] = -3e-4;
        cam.d[5] = 1e-4;
        cam.d[8] = -2e-4;
        let lib = camera_params(&cam).build().unwrap();
        for &(x, y, z) in &[(0.1, -0.2, 1.0), (-0.3, 0.25, 1.2), (0.4, 0.3, 0.9)] {
            let a = lib.project_point(&Point3::new(x, y, z)).unwrap();
            let b = project(&cam.k, &cam.d, &cam.tilt, &Vector3::new(x, y, z)).unwrap();
            assert!((a.x - b[0]).abs() < 1e-9 && (a.y - b[1]).abs() < 1e-9);
        }
    }

    /// `K·H(τ) = K'(τ)·R(τ)` holds to rounding for any tilt.
    #[test]
    fn tilt_factors_into_a_rotated_pinhole() {
        for &t in &[(1.0, 0.0), (5.0, -3.0), (20.0, 20.0), (-12.0, 7.0)] {
            let cam = spec(t, 0.0).camera();
            assert!(factorization_error(&cam) < 1e-9, "{t:?}");
        }
    }

    /// Every parameterization describes the same camera: projecting the
    /// truth through each gives the same pixels.
    #[test]
    fn parameterizations_agree_at_the_truth() {
        let scene = build_scene(spec((8.0, -5.0), -0.05), &ViewDesign::baseline()).unwrap();
        for p in Param::ALL {
            let v = p.coords(&scene.cam);
            assert!(
                (p.camera(&v).raw().iter().zip(scene.cam.raw())).all(|(a, b)| (a - b).abs() < 1e-9)
            );
            for (pose, view) in scene.poses.iter().zip(&scene.views) {
                let pp = p.pose_to(&scene.cam, pose);
                for (b, px) in view.board.iter().zip(&view.pixels) {
                    let q = p.project(&v, &pp, &Vector3::new(b[0], b[1], b[2])).unwrap();
                    assert!((q[0] - px[0]).abs() < 1e-7 && (q[1] - px[1]).abs() < 1e-7);
                }
            }
        }
    }

    /// Without distortion the tilt is a gauge: singular information in the
    /// library's parameterization, and an exactly zero tilt column in the
    /// gauge-aligned one.
    #[test]
    fn without_distortion_the_tilt_is_unobservable() {
        let scene = build_scene(spec((5.0, 2.0), 0.0), &ViewDesign::baseline()).unwrap();
        let f = fisher(
            &scene,
            &scene.cam,
            &scene.poses,
            Param::Raw,
            &FreeSet::Default.indices(),
        );
        assert!(f.cond_cam > 1e12, "cond {}", f.cond_cam);
        // The tilt moves every pixel in the library's coordinates and none in
        // the gauge-aligned ones (to the central difference's rounding).
        let raw = column_norms(&scene, &scene.cam, &scene.poses, Param::Raw);
        let gauge = column_norms(&scene, &scene.cam, &scene.poses, Param::Gauge);
        for c in [TX, TY] {
            assert!(gauge[c] < 1e-8 * raw[c], "{} vs {}", gauge[c], raw[c]);
        }
    }

    /// With distortion the tilt is observable, and freeing the tangential
    /// terms inflates its σ several times (the decentering degeneracy).
    #[test]
    fn tangential_terms_absorb_the_tilt_signal() {
        let scene = build_scene(spec((5.0, 0.0), -0.05), &ViewDesign::baseline()).unwrap();
        let free = |s: FreeSet| fisher(&scene, &scene.cam, &scene.poses, Param::Raw, &s.indices());
        let held = free(FreeSet::Default);
        let tang = free(FreeSet::Tangential);
        let s_held = held.sigma(TX, 0.2);
        let s_tang = tang.sigma(TX, 0.2);
        assert!(
            s_held.is_finite() && s_tang > 3.0 * s_held,
            "{s_held} {s_tang}"
        );
        assert!(tang.corr(TX, 7).abs() > 0.9, "{}", tang.corr(TX, 7));
        // A reparameterization cannot change the marginal σ of a kept
        // coordinate.
        let gauge = fisher(
            &scene,
            &scene.cam,
            &scene.poses,
            Param::Gauge,
            &FreeSet::Default.indices(),
        );
        let ratio = gauge.sigma(TX, 0.2) / s_held;
        assert!((ratio - 1.0).abs() < 1e-3, "{ratio}");
    }
}

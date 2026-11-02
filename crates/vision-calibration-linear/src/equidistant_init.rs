//! Planar intrinsics initialization for a Kannala-Brandt fisheye camera
//! (equidistant projection).
//!
//! Zhang's method rests on the board-to-pixel map being a homography, which
//! a fisheye lens breaks. This initializer works on bearings instead: for a
//! candidate focal length `f`, principal point `c` and distortion, every
//! pixel back-projects to a bearing, each view's pose follows from its
//! bearings ([`estimate_planar_pose_from_bearings`]), and the board
//! reprojects to pixels; the pixel error scores the candidate.
//!
//! 1. A log-spaced sweep over `f` without distortion, the principal point
//!    at the centre of the observations. The sweep covers fields of view
//!    from a few degrees to beyond 270°, so no image size or nominal focal
//!    length is needed.
//! 2. A simplex search over `(f, cx, cy)`, still without distortion.
//! 3. A linear fit of `f`, `k1` and `k2` to the observed radii
//!    `r = f θ (1 + k1 θ² + k2 θ⁴)` at the incidence angles `θ` of the
//!    current poses, then a simplex search over all five together.
//!
//! Pixels are square (`fx = fy`) and `k3`, `k4` start at zero; the
//! non-linear refinement frees them.

use crate::Error;
use crate::planar_pose::estimate_planar_pose_from_bearings;
use nalgebra::{Matrix3, Point2, Point3, Vector3};
use vision_calibration_core::{
    DistortionModel, Equidistant, FxFyCxCy, Iso3, KannalaBrandt4, PlanarDataset, ProjectionModel,
    Pt2, Real,
};

/// Which Kannala-Brandt coefficients the linear stage fits; a coefficient
/// it does not fit stays at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct EquidistantInitOptions {
    /// Fit `k1`, the `θ³` coefficient.
    pub fit_k1: bool,
    /// Fit `k2`, the `θ⁵` coefficient.
    pub fit_k2: bool,
}

impl Default for EquidistantInitOptions {
    /// Fit both: together they carry most of a real fisheye lens's
    /// departure from the equidistant model.
    fn default() -> Self {
        Self {
            fit_k1: true,
            fit_k2: true,
        }
    }
}

impl EquidistantInitOptions {
    /// Options fitting `k1` and `k2` as given.
    pub fn new(fit_k1: bool, fit_k2: bool) -> Self {
        Self { fit_k1, fit_k2 }
    }
}

/// Initial intrinsics, distortion and poses of a Kannala-Brandt camera.
#[derive(Debug, Clone)]
pub struct EquidistantIntrinsicsLinearInit {
    /// Focal length (`fx = fy`, pixels per radian) and principal point.
    pub intrinsics: FxFyCxCy<Real>,
    /// Kannala-Brandt coefficients; `k3` and `k4` are zero.
    pub distortion: KannalaBrandt4<Real>,
    /// Per-view board pose (`cam_se3_target`), one per dataset view.
    pub poses: Vec<Iso3>,
    /// Root-mean-square pixel error of the fit.
    pub rms_error: Real,
}

/// Number of focal lengths in the initial sweep.
const FOCAL_SWEEP: usize = 40;
/// The sweep's focal range, as the observed radius over the angle it spans:
/// from `θ_max = 2.5` rad (143°) down to `θ_max = 0.05` rad (3°).
const SWEEP_MAX_ANGLE: Real = 2.5;
const SWEEP_MIN_ANGLE: Real = 0.05;
/// Simplex search budget and convergence tolerance (relative spread of the
/// simplex's costs).
const SIMPLEX_EVALUATIONS: usize = 600;
const SIMPLEX_TOLERANCE: Real = 1e-12;
/// Initial simplex steps: the principal point as a fraction of the observed
/// radius, and the distortion coefficients.
const CENTRE_STEP: Real = 0.05;
const K1_STEP: Real = 0.01;
const K2_STEP: Real = 0.002;

/// Estimate the intrinsics, distortion and per-view poses of a
/// Kannala-Brandt camera from planar views (see the module documentation).
///
/// # Errors
///
/// - [`Error::InsufficientData`] for fewer than three views.
/// - [`Error::InvalidInput`] if the observations span no area.
/// - [`Error::Numerical`] when no candidate focal length fits every view.
pub fn estimate_equidistant_intrinsics(
    dataset: &PlanarDataset,
    opts: EquidistantInitOptions,
) -> Result<EquidistantIntrinsicsLinearInit, Error> {
    if dataset.views.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: dataset.views.len(),
        });
    }
    let views: Vec<(Vec<Pt2>, &[Pt2])> = dataset
        .views
        .iter()
        .map(|view| {
            let board = view
                .obs
                .points_3d
                .iter()
                .map(|p| Pt2::new(p.x, p.y))
                .collect();
            (board, view.obs.points_2d.as_slice())
        })
        .collect();

    // The observations' bounding-box centre seeds the principal point, and
    // their radius about it scales the focal sweep.
    let (mut lo, mut hi) = (
        Pt2::new(Real::INFINITY, Real::INFINITY),
        Pt2::new(Real::NEG_INFINITY, Real::NEG_INFINITY),
    );
    for (_, pixels) in &views {
        for p in *pixels {
            lo = Pt2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Pt2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
    }
    let centre = Pt2::new(0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y));
    let radius = 0.5 * (hi - lo).norm();
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::invalid_input(
            "the observations span no area; a camera needs points across the image",
        ));
    }
    let search = SearchSpace { centre, radius };

    // 1. The focal sweep, without distortion.
    let (ln_lo, ln_hi) = (
        (radius / SWEEP_MAX_ANGLE).ln(),
        (radius / SWEEP_MIN_ANGLE).ln(),
    );
    let sweep_step = (ln_hi - ln_lo) / (FOCAL_SWEEP - 1) as Real;
    let mut x = [0.0; 5];
    let mut best = Real::INFINITY;
    for i in 0..FOCAL_SWEEP {
        let candidate = [ln_lo + sweep_step * i as Real, 0.0, 0.0, 0.0, 0.0];
        let cost = search.cost(&views, &candidate);
        if cost < best {
            (x, best) = (candidate, cost);
        }
    }
    if !best.is_finite() {
        return Err(Error::numerical(
            "no focal length fits every view with an equidistant camera",
        ));
    }

    // 2. Focal length and principal point, without distortion.
    let steps = [sweep_step, CENTRE_STEP, CENTRE_STEP, K1_STEP, K2_STEP];
    x = nelder_mead(x, steps, &[0, 1, 2], &|x| search.cost(&views, x));

    // 3. The radial fit seeds the distortion, then all of them together.
    let mut free = vec![0, 1, 2];
    free.extend(opts.fit_k1.then_some(3));
    free.extend(opts.fit_k2.then_some(4));
    if free.len() > 3 {
        let (k, distortion) = search.camera(&x);
        if let Some((_, poses)) = fit(&views, &k, &distortion)
            && let Some((f, k1, k2)) = fit_radial(&views, &poses, &k, opts)
        {
            (x[0], x[3], x[4]) = (f.ln(), k1, k2);
        }
        x = nelder_mead(x, steps, &free, &|x| search.cost(&views, x));
    }

    let (intrinsics, distortion) = search.camera(&x);
    let (sse, poses) = fit(&views, &intrinsics, &distortion)
        .ok_or_else(|| Error::numerical("the refined fisheye fit lost a view"))?;
    let count: usize = views.iter().map(|(_, pixels)| pixels.len()).sum();
    Ok(EquidistantIntrinsicsLinearInit {
        intrinsics,
        distortion,
        poses,
        rms_error: (sse / count as Real).sqrt(),
    })
}

/// The search coordinates `x = (ln f, Δcx / R, Δcy / R, k1, k2)` about the
/// centre and radius `R` of the observations.
struct SearchSpace {
    centre: Pt2,
    radius: Real,
}

impl SearchSpace {
    fn camera(&self, x: &[Real; 5]) -> (FxFyCxCy<Real>, KannalaBrandt4<Real>) {
        let f = x[0].exp();
        let intrinsics = FxFyCxCy {
            fx: f,
            fy: f,
            cx: self.centre.x + x[1] * self.radius,
            cy: self.centre.y + x[2] * self.radius,
        };
        let distortion = KannalaBrandt4 {
            k1: x[3],
            k2: x[4],
            k3: 0.0,
            k4: 0.0,
            iters: 0,
        };
        (intrinsics, distortion)
    }

    fn cost(&self, views: &[(Vec<Pt2>, &[Pt2])], x: &[Real; 5]) -> Real {
        let (k, distortion) = self.camera(x);
        fit(views, &k, &distortion).map_or(Real::INFINITY, |(sse, _)| sse)
    }
}

/// Sum of squared pixel errors and the per-view poses of a camera, or
/// `None` when a view's pose is degenerate.
fn fit(
    views: &[(Vec<Pt2>, &[Pt2])],
    k: &FxFyCxCy<Real>,
    distortion: &KannalaBrandt4<Real>,
) -> Option<(Real, Vec<Iso3>)> {
    let mut sse = 0.0;
    let mut poses = Vec::with_capacity(views.len());
    let mut bearings = Vec::new();
    for (board, pixels) in views {
        bearings.clear();
        bearings.extend(pixels.iter().map(|p| {
            let n_d = Point2::new((p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy);
            let n = distortion.undistort(&n_d);
            ProjectionModel::<Real>::unproject(&Equidistant, &n).direction
        }));
        let pose = estimate_planar_pose_from_bearings(board, &bearings).ok()?;
        for (b, p) in board.iter().zip(*pixels) {
            let p_c = pose.transform_point(&Point3::new(b.x, b.y, 0.0));
            let n = ProjectionModel::<Real>::project(&Equidistant, &p_c.coords);
            let n_d = distortion.distort(&n);
            let (du, dv) = (k.fx * n_d.x + k.cx - p.x, k.fy * n_d.y + k.cy - p.y);
            sse += du * du + dv * dv;
        }
        poses.push(pose);
    }
    sse.is_finite().then_some((sse, poses))
}

/// Least-squares `f`, `k1`, `k2` of `r = f θ + f k1 θ³ + f k2 θ⁵` from each
/// point's observed radius about the principal point and its incidence angle
/// under `poses`; a coefficient `opts` does not fit stays zero. `None` when
/// the system is singular or the focal length comes out non-positive.
fn fit_radial(
    views: &[(Vec<Pt2>, &[Pt2])],
    poses: &[Iso3],
    k: &FxFyCxCy<Real>,
    opts: EquidistantInitOptions,
) -> Option<(Real, Real, Real)> {
    let active = [true, opts.fit_k1, opts.fit_k2];
    let mut ata = Matrix3::<Real>::zeros();
    let mut atb = Vector3::<Real>::zeros();
    for ((board, pixels), pose) in views.iter().zip(poses) {
        for (b, p) in board.iter().zip(*pixels) {
            let p_c = pose.transform_point(&Point3::new(b.x, b.y, 0.0));
            let theta = p_c.coords.xy().norm().atan2(p_c.z);
            let r = (p.x - k.cx).hypot(p.y - k.cy);
            let t2 = theta * theta;
            let row = Vector3::new(theta, theta * t2, theta * t2 * t2);
            let row = Vector3::from_fn(|i, _| if active[i] { row[i] } else { 0.0 });
            ata += row * row.transpose();
            atb += row * r;
        }
    }
    for (i, &on) in active.iter().enumerate() {
        if !on {
            ata[(i, i)] = 1.0;
        }
    }
    let a = ata.cholesky()?.solve(&atb);
    let f = a[0];
    (f > 0.0 && f.is_finite()).then(|| (f, a[1] / f, a[2] / f))
}

/// Minimize `cost` with the Nelder–Mead simplex from `start`, moving only
/// the coordinates in `free`, each with its initial step from `step`.
fn nelder_mead<const N: usize>(
    start: [Real; N],
    step: [Real; N],
    free: &[usize],
    cost: &impl Fn(&[Real; N]) -> Real,
) -> [Real; N] {
    let mut simplex: Vec<([Real; N], Real)> = std::iter::once(start)
        .chain(free.iter().map(|&i| {
            let mut x = start;
            x[i] += step[i];
            x
        }))
        .map(|x| (x, cost(&x)))
        .collect();
    let n = free.len();
    let mut evaluations = simplex.len();
    let lerp = |a: &[Real; N], b: &[Real; N], t: Real| -> [Real; N] {
        std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
    };
    while evaluations < SIMPLEX_EVALUATIONS {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (best, worst) = (simplex[0].1, simplex[n].1);
        if (worst - best).abs() <= SIMPLEX_TOLERANCE * best.abs().max(Real::MIN_POSITIVE) {
            break;
        }
        let centroid: [Real; N] = std::array::from_fn(|i| {
            simplex[..n].iter().map(|(x, _)| x[i]).sum::<Real>() / n as Real
        });
        let reflected = lerp(&centroid, &simplex[n].0, -1.0);
        let reflected_cost = cost(&reflected);
        evaluations += 1;
        if reflected_cost < simplex[0].1 {
            let expanded = lerp(&centroid, &simplex[n].0, -2.0);
            let expanded_cost = cost(&expanded);
            evaluations += 1;
            simplex[n] = if expanded_cost < reflected_cost {
                (expanded, expanded_cost)
            } else {
                (reflected, reflected_cost)
            };
        } else if reflected_cost < simplex[n - 1].1 {
            simplex[n] = (reflected, reflected_cost);
        } else {
            let contracted = lerp(&centroid, &simplex[n].0, 0.5);
            let contracted_cost = cost(&contracted);
            evaluations += 1;
            if contracted_cost < simplex[n].1 {
                simplex[n] = (contracted, contracted_cost);
            } else {
                let anchor = simplex[0].0;
                for vertex in &mut simplex[1..] {
                    vertex.0 = lerp(&anchor, &vertex.0, 0.5);
                    vertex.1 = cost(&vertex.0);
                }
                evaluations += n;
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex[0].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use vision_calibration_core::{
        Camera, CorrespondenceView, IdentitySensor, KannalaBrandt4, NoDistortion, Pt3, View,
    };

    fn fisheye_dataset<D>(distortion: D, k: FxFyCxCy<Real>) -> (PlanarDataset, Vec<Iso3>)
    where
        D: vision_calibration_core::DistortionModel<Real> + Clone,
    {
        let camera = Camera::new(Equidistant, distortion, IdentitySensor, k);
        let board: Vec<Pt3> = (0..9)
            .flat_map(|i| (0..7).map(move |j| Pt3::new(i as Real * 0.05, j as Real * 0.05, 0.0)))
            .collect();
        // Boards close to the camera and far off-axis, some beyond 90°.
        let poses: Vec<Iso3> = [
            ([0.0, 0.0, 0.0], [-0.2, -0.15, 0.35]),
            ([0.3, -0.5, 0.1], [-0.4, -0.1, 0.25]),
            ([-0.4, 0.6, -0.2], [0.15, -0.2, 0.3]),
            ([0.0, 1.4, 0.0], [0.3, -0.15, 0.05]),
            ([0.2, -1.5, 0.3], [-0.3, -0.1, 0.0]),
            ([1.3, 0.2, 0.0], [-0.2, 0.25, 0.05]),
        ]
        .iter()
        .map(|(r, t)| {
            Iso3::from_parts(
                Translation3::new(t[0], t[1], t[2]),
                UnitQuaternion::from_euler_angles(r[0], r[1], r[2]),
            )
        })
        .collect();
        let views = poses
            .iter()
            .map(|pose| {
                let pixels = board
                    .iter()
                    .map(|p| {
                        let p_c = pose.transform_point(p);
                        camera.project_point(&p_c).expect("in the fisheye domain")
                    })
                    .collect();
                View::without_meta(CorrespondenceView::new(board.clone(), pixels).unwrap())
            })
            .collect();
        (PlanarDataset::new(views).unwrap(), poses)
    }

    fn truth() -> FxFyCxCy<Real> {
        FxFyCxCy {
            fx: 330.0,
            fy: 330.0,
            cx: 645.0,
            cy: 395.0,
        }
    }

    #[test]
    fn exact_equidistant_views_recover_intrinsics_and_poses() {
        let (dataset, poses) = fisheye_dataset(NoDistortion, truth());
        let max_angle = dataset
            .views
            .iter()
            .zip(&poses)
            .flat_map(|(v, pose)| {
                v.obs.points_3d.iter().map(move |p| {
                    let p_c = pose.transform_point(p);
                    p_c.coords.xy().norm().atan2(p_c.z)
                })
            })
            .fold(0.0, Real::max);
        assert!(
            max_angle > 1.7,
            "the scene reaches beyond 90° ({max_angle})"
        );

        let init =
            estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::default()).unwrap();
        let k = init.intrinsics;
        assert!((k.fx - 330.0).abs() < 1e-3, "{k:?}");
        assert!(
            (k.cx - 645.0).abs() < 1e-3 && (k.cy - 395.0).abs() < 1e-3,
            "{k:?}"
        );
        assert!(init.rms_error < 1e-4, "{}", init.rms_error);
        for (est, gt) in init.poses.iter().zip(&poses) {
            assert!((est.translation.vector - gt.translation.vector).norm() < 1e-5);
            assert!(est.rotation.angle_to(&gt.rotation) < 1e-5);
        }
    }

    #[test]
    fn kannala_brandt_views_land_near_the_truth() {
        // A wide lens compressing the edge by about 5 % at 115°: the linear
        // model has no distortion, so it lands near, not on, the truth; the
        // refinement closes the gap.
        let distortion = KannalaBrandt4 {
            k1: -0.02,
            k2: 0.002,
            k3: -0.0001,
            k4: 0.0,
            iters: 20,
        };
        let (dataset, poses) = fisheye_dataset(distortion, truth());
        let init =
            estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::default()).unwrap();
        let (k, d) = (init.intrinsics, init.distortion);
        assert!((k.fx / 330.0 - 1.0).abs() < 0.01, "{k:?}");
        assert!(
            (k.cx - 645.0).abs() < 1.0 && (k.cy - 395.0).abs() < 1.0,
            "{k:?}"
        );
        assert!((d.k1 + 0.02).abs() < 0.005, "{d:?}");
        assert!(init.rms_error < 0.5, "{}", init.rms_error);
        for (est, gt) in init.poses.iter().zip(&poses) {
            assert!(est.rotation.angle_to(&gt.rotation) < 0.01);
        }

        // A coefficient the caller holds stays at zero.
        let held =
            estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::new(true, false))
                .unwrap();
        assert_eq!(held.distortion.k2, 0.0);
        assert!(held.distortion.k1 < 0.0);
        let none =
            estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::new(false, false))
                .unwrap();
        assert_eq!((none.distortion.k1, none.distortion.k2), (0.0, 0.0));
    }

    #[test]
    fn a_narrow_lens_is_found_by_the_sweep() {
        // A 2400 px/rad lens sees the board within about 6° of the axis.
        let k = FxFyCxCy {
            fx: 2400.0,
            fy: 2400.0,
            cx: 640.0,
            cy: 480.0,
        };
        let camera = Camera::new(Equidistant, NoDistortion, IdentitySensor, k);
        let board: Vec<Pt3> = (0..8)
            .flat_map(|i| (0..6).map(move |j| Pt3::new(i as Real * 0.03, j as Real * 0.03, 0.0)))
            .collect();
        let views = (0..5)
            .map(|v| {
                let a = v as Real * 0.25;
                let pose = Iso3::from_parts(
                    Translation3::new(-0.1, -0.08, 1.5 + 0.1 * v as Real),
                    UnitQuaternion::from_euler_angles(0.3 * a.cos(), 0.3 * a.sin(), 0.1 * a),
                );
                let pixels = board
                    .iter()
                    .map(|p| camera.project_point(&pose.transform_point(p)).unwrap())
                    .collect();
                View::without_meta(CorrespondenceView::new(board.clone(), pixels).unwrap())
            })
            .collect();
        let dataset = PlanarDataset::new(views).unwrap();
        let init =
            estimate_equidistant_intrinsics(&dataset, EquidistantInitOptions::new(false, false))
                .unwrap();
        assert!(
            (init.intrinsics.fx / 2400.0 - 1.0).abs() < 1e-3,
            "{:?}",
            init.intrinsics
        );
    }

    #[test]
    fn too_few_views_are_rejected() {
        let (dataset, _) = fisheye_dataset(NoDistortion, truth());
        let two = PlanarDataset::new(dataset.views[..2].to_vec()).unwrap();
        assert!(matches!(
            estimate_equidistant_intrinsics(&two, EquidistantInitOptions::default()),
            Err(Error::InsufficientData { need: 3, got: 2 })
        ));
    }
}

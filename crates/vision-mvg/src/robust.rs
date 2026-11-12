//! Robust estimation using RANSAC for epipolar and homography models.
//!
//! Provides RANSAC-based estimators that work with [`PixelMatch`] (calibrated)
//! and [`Correspondence2D`] (homography):
//!
//! - [`estimate_essential`]: 5-point algorithm on bearings + covariance-weighted
//!   Sampson error, refined on its inliers with the covariance of `E`
//! - [`estimate_homography`]: 4-point DLT + symmetric transfer error
//! - [`recover_relative_pose_robust`]: full pipeline from matched pixels to
//!   pose, with the covariance of the relative pose
//!
//! # The two-view refinement and its gauge
//!
//! The consensus model is refined on its inliers to the least squares of the
//! weighted Sampson error, through the public problem builder of
//! `vision-calibration-optim`, as a relative pose: `E = [t]× R` with the
//! rotation `R` (the rotation of `second_se3_first`) and the unit translation
//! direction `t`. Two views do not observe the scale of the translation, so
//! `|t| = 1` is the gauge: the pose has five degrees of freedom, three of
//! rotation and two of direction.
//!
//! The covariance of the relative pose is labelled `rotation.rx`,
//! `rotation.ry`, `rotation.rz` (rad, the right-perturbation tangent
//! `R·Exp(δ)`) and `direction.x`, `direction.y`, `direction.z` (the unit
//! translation's components, in the second camera's frame): the direction
//! block is rank 2, with no variance along `t`. The covariance of `E` is that
//! of its nine entries (`essential.e00` … `essential.e22`, row-major, `E`
//! scaled to `|t| = 1`), rank 5.

use crate::bearing::{bearing_with_jacobian, camera_model};
use crate::cheirality::{cheirality_count, select_pose};
use crate::residuals;
use crate::triangulation::sub_covariance;
use crate::types::{Correspondence2D, PixelMatch};
use crate::{Error, Result};
use nalgebra::{
    DVector, Matrix2, Matrix6, RealField, Rotation3, SMatrix, Translation3, UnitQuaternion, Vector3,
};
use serde::Serialize;
use vision_calibration_core::{
    CalibratedRig, Estimator, Iso3, Mat3, RansacOptions, Real, RunControl, Uncertainty, Vec3,
    ransac_fit,
};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    Manifold, Noise, Problem, Residual, SolveOptions, so3, solve,
};

/// The labels of the relative pose's covariance, in order.
const POSE_LABELS: [&str; 6] = [
    "rotation.rx",
    "rotation.ry",
    "rotation.rz",
    "direction.x",
    "direction.y",
    "direction.z",
];

/// The label of the essential matrix's entry `(row, col)`.
fn essential_label(row: usize, col: usize) -> String {
    format!("essential.e{row}{col}")
}

/// Result of robust essential matrix estimation.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct EssentialEstimate {
    /// The estimated essential matrix: `b2ᵀ E b1 = 0` on the bearings of
    /// matched pixels, scaled to a unit translation (`‖E‖_F = √2`).
    pub essential: Mat3,
    /// Indices of the inlier matches: those the estimate is fitted to.
    pub inliers: Vec<usize>,
    /// RMS of the weighted Sampson error over inliers, in σ: about 1 when the
    /// stated pixel noise describes the pixels.
    pub inlier_rms: Real,
    /// The covariance of `essential`'s nine entries, labelled
    /// `essential.e00` … `essential.e22` (row-major), from the inliers'
    /// stated pixel noise: rank 5, as `E` has five degrees of freedom at its
    /// scale. Or why it is unavailable (fewer than six inliers, a refinement
    /// that stopped on its iteration budget).
    pub uncertainty: Uncertainty,
}

impl EssentialEstimate {
    /// The 9×9 covariance of `essential`'s entries in row-major order; `None`
    /// when the uncertainty is unavailable.
    pub fn essential_covariance(&self) -> Option<SMatrix<Real, 9, 9>> {
        let labels: [String; 9] = std::array::from_fn(|k| essential_label(k / 3, k % 3));
        sub_covariance(&self.uncertainty, labels.clone(), labels)
    }
}

/// Result of robust homography estimation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HomographyEstimate {
    /// The estimated homography matrix.
    pub homography: Mat3,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// RMS symmetric transfer error over inliers.
    pub inlier_rms: Real,
}

/// Result of robust relative pose recovery.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct RobustRelativePose {
    /// Pose of the first camera's frame in the second's (`second_se3_first`),
    /// with a unit-norm translation: the scale is not observable.
    pub second_se3_first: Iso3,
    /// The essential matrix used: `b2ᵀ E b1 = 0` on the bearings.
    pub essential: Mat3,
    /// Indices of the inlier matches: those the pose is fitted to.
    pub inliers: Vec<usize>,
    /// RMS of the weighted Sampson error over inliers, in σ: about 1 when the
    /// stated pixel noise describes the pixels.
    pub inlier_rms: Real,
    /// The covariance of the relative pose in the gauge `|t| = 1`, from the
    /// inliers' stated pixel noise: `rotation.rx`, `.ry`, `.rz` (rad, the
    /// right-perturbation tangent `R·Exp(δ)` of the rotation of
    /// `second_se3_first`) and `direction.x`, `.y`, `.z` (the components of
    /// its unit translation; rank 2, no variance along the translation). Or
    /// why it is unavailable (fewer than six inliers, a refinement that
    /// stopped on its iteration budget).
    pub uncertainty: Uncertainty,
}

impl RobustRelativePose {
    /// The 6×6 covariance over `[rx ry rz x y z]`: the rotation's
    /// right-perturbation tangent (rad) and the unit translation's
    /// components; `None` when the uncertainty is unavailable.
    pub fn pose_covariance(&self) -> Option<Matrix6<Real>> {
        let labels = POSE_LABELS.map(String::from);
        sub_covariance(&self.uncertainty, labels.clone(), labels)
    }
}

// --- Essential matrix RANSAC estimator ---

/// The unit bearings of one match, in the two cameras' frames, with the
/// covariances of the bearings (3×3, rank 2) their pixels' noise maps to.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BearingPair {
    pub(crate) b1: Vec3,
    pub(crate) b2: Vec3,
    pub(crate) m1: Mat3,
    pub(crate) m2: Mat3,
}

impl BearingPair {
    /// The weighted Sampson error of `e` on this pair, in σ.
    pub(crate) fn error(&self, e: &Mat3) -> Real {
        residuals::weighted_sampson_error(e, &self.b1, &self.b2, &self.m1, &self.m2)
    }
}

/// A relative pose: `p2 = R p1 + t` with a unit `t`, and its essential
/// matrix `E = [t]× R`.
#[derive(Debug, Clone, Copy)]
struct Motion {
    rotation: UnitQuaternion<Real>,
    direction: Vec3,
}

impl Motion {
    /// The motion of a rotation matrix and a translation (normalized).
    fn new(r: &Mat3, t: &Vec3) -> Self {
        let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r));
        Self {
            rotation: UnitQuaternion::new_normalize(rotation.into_inner()),
            direction: t.normalize(),
        }
    }

    fn rotation_matrix(&self) -> Mat3 {
        self.rotation.to_rotation_matrix().into_inner()
    }

    fn essential(&self) -> Mat3 {
        self.direction.cross_matrix() * self.rotation_matrix()
    }

    fn iso(&self) -> Iso3 {
        Iso3::from_parts(Translation3::from(self.direction), self.rotation)
    }
}

/// The weighted Sampson error of one bearing pair, `g / σ_g` with
/// `E = [t]× R`, over a rotation block (`R`) and a unit-vector block (`t`).
/// `σ_g` is evaluated at the blocks, so the Jacobian carries its dependence
/// on `E`; the residual is already in σ (`Noise::Stated(1.0)`).
struct SampsonResidual(BearingPair);

impl Residual for SampsonResidual {
    fn dim(&self) -> usize {
        1
    }

    fn block_dims(&self) -> Vec<usize> {
        vec![4, 3]
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let lift = |c: f64| T::from_f64(c).unwrap();
        let p = &self.0;
        let r = so3(&blocks[0]).to_rotation_matrix().into_inner();
        let t = Vector3::new(
            blocks[1][0].clone(),
            blocks[1][1].clone(),
            blocks[1][2].clone(),
        );
        let e = t.cross_matrix() * r;
        let (g, var) = residuals::epipolar_constraint_variance(
            &e,
            &p.b1.map(lift),
            &p.b2.map(lift),
            &p.m1.map(lift),
            &p.m2.map(lift),
        );
        DVector::from_element(1, g / var.sqrt())
    }
}

/// A motion refined on its matches, with its covariance (labels
/// [`POSE_LABELS`]).
#[derive(Debug, Clone)]
struct Fit {
    motion: Motion,
    uncertainty: Uncertainty,
}

/// `start` refined on the matches `used` to the least squares of the
/// weighted Sampson error, through the problem builder.
///
/// The linear (8-point) solution is not that minimizer: noise in the design
/// matrix biases it, the more so when the two cameras' angular noise differs
/// (a fisheye matched with a pinhole) or the stated covariances are
/// anisotropic, and its error then shows as an inlier RMS well above 1 σ. A
/// match whose constraint carries no information at `start` (a bearing at
/// an epipole: its σ vanishes) is left out of the fit.
fn refine(
    pairs: &[BearingPair],
    used: &[usize],
    start: &Motion,
    control: &RunControl,
) -> Result<Fit> {
    let fail = |e: vision_calibration_optim::Error| match e {
        vision_calibration_optim::Error::Cancelled => Error::Cancelled,
        e => Error::numerical(format!("refining the relative pose: {e}")),
    };
    let mut problem = Problem::new();
    let q = start.rotation.as_ref().coords;
    let rotation = problem
        .add_block("rotation", &[q.x, q.y, q.z, q.w], Manifold::So3)
        .map_err(fail)?;
    let t = start.direction;
    let direction = problem
        .add_block("direction", &[t.x, t.y, t.z], Manifold::UnitVector)
        .map_err(fail)?;
    problem
        .set_labels(direction, &["x", "y", "z"])
        .map_err(fail)?;
    let e = start.essential();
    for &i in used {
        if pairs[i].error(&e).is_finite() {
            problem
                .add_residual(
                    SampsonResidual(pairs[i]),
                    &[rotation, direction],
                    RobustLoss::None,
                    Noise::Stated(1.0),
                )
                .map_err(fail)?;
        }
    }
    // `SolveOptions` is non-exhaustive: start from its default.
    let mut options = SolveOptions::default();
    options.control = control.clone();
    let solution = solve(&problem, &options).map_err(fail)?;
    let (Some(rotation), Some(direction)) =
        (solution.rotation(rotation), solution.value(direction))
    else {
        return Err(Error::numerical(
            "the relative pose is missing from the solution",
        ));
    };
    Ok(Fit {
        motion: Motion {
            rotation,
            direction: Vec3::new(direction[0], direction[1], direction[2]).normalize(),
        },
        uncertainty: solution.uncertainty,
    })
}

/// The decomposition of `e` with the most of the `used` matches ahead along
/// both bearings; the first decomposition when none has any (the estimate of
/// `E` does not depend on the choice).
fn physical_motion(e: &Mat3, pairs: &[BearingPair], used: &[usize]) -> Result<Motion> {
    let candidates = vision_geometry::epipolar::decompose_essential(e)?;
    let (b1, b2) = bearings(pairs, used);
    let (r, t) = select_pose(&candidates, &b1, &b2)
        .map(|(r, t, _)| (r, t))
        .or_else(|| candidates.first().map(|c| (c.rotation, c.translation)))
        .ok_or_else(|| Error::numerical("the essential matrix has no decomposition"))?;
    Ok(Motion::new(&r, &t))
}

/// The bearings of the matches `used`, first and second view.
fn bearings(pairs: &[BearingPair], used: &[usize]) -> (Vec<Vec3>, Vec<Vec3>) {
    used.iter().map(|&i| (pairs[i].b1, pairs[i].b2)).unzip()
}

/// The RMS weighted Sampson error of `e` over the matches `used`, in σ.
fn rms(pairs: &[BearingPair], used: &[usize], e: &Mat3) -> Real {
    let sum: Real = used.iter().map(|&i| pairs[i].error(e).powi(2)).sum();
    (sum / used.len() as Real).sqrt()
}

/// The covariance of `E = [t]× R`'s entries from the motion's: `J Σ Jᵀ`
/// with `∂E/∂δωₖ = [t]× R [eₖ]×` (`R·Exp(δω)`) and `∂E/∂tₖ = [eₖ]× R`.
fn essential_uncertainty(fit: &Fit) -> Uncertainty {
    let labels = POSE_LABELS.map(String::from);
    let (Some(covariance), Some(sigma)) = (
        fit.uncertainty.covariance(),
        sub_covariance::<6>(&fit.uncertainty, labels.clone(), labels),
    ) else {
        return fit.uncertainty.clone();
    };
    let r = fit.motion.rotation_matrix();
    let tx = fit.motion.direction.cross_matrix();
    let columns: [Mat3; 6] = std::array::from_fn(|k| {
        let axis = Vec3::ith(k % 3, 1.0).cross_matrix();
        if k < 3 { tx * r * axis } else { axis * r }
    });
    let jacobian = SMatrix::<Real, 9, 6>::from_fn(|row, k| columns[k][(row / 3, row % 3)]);
    let matrix = jacobian * sigma * jacobian.transpose();
    let mut out = covariance.clone();
    out.labels = (0..9).map(|k| essential_label(k / 3, k % 3)).collect();
    out.covariance = (0..9)
        .map(|a| {
            (0..9)
                .map(|b| 0.5 * (matrix[(a, b)] + matrix[(b, a)]))
                .collect()
        })
        .collect();
    Uncertainty::Available(out)
}

struct EssentialEstimator;

impl Estimator for EssentialEstimator {
    type Datum = BearingPair;
    type Model = Mat3;

    const MIN_SAMPLES: usize = 5;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        let n = sample_indices.len();

        // Shared helper: pick the candidate E that minimises the total weighted
        // Sampson error over *all* sample_indices (not just the 5 used to
        // construct the candidates).  Used for both the minimal and the 6-7
        // point paths.
        let best_of = |candidates: Vec<Mat3>| -> Option<Mat3> {
            let total =
                |e: &Mat3| -> Real { sample_indices.iter().map(|&i| data[i].error(e)).sum() };
            candidates.into_iter().min_by(|a, b| {
                total(a)
                    .partial_cmp(&total(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        };

        if n >= 8 {
            // Overdetermined case (refit on inliers): the linear solver has a
            // 1-dimensional null space and projects onto the essential manifold.
            let b1: Vec<_> = sample_indices.iter().map(|&i| data[i].b1).collect();
            let b2: Vec<_> = sample_indices.iter().map(|&i| data[i].b2).collect();
            vision_geometry::epipolar::essential_linear(&b1, &b2).ok()
        } else if n >= 5 {
            // Minimal or near-minimal case (5, 6, or 7 points): use the 5-point
            // solver on the first five matches, then score all candidates
            // against the full sample_indices set to pick the best one.
            // RANSAC guarantees n >= MIN_SAMPLES == 5, so n < 5 is unreachable.
            let b1: Vec<_> = sample_indices[..5].iter().map(|&i| data[i].b1).collect();
            let b2: Vec<_> = sample_indices[..5].iter().map(|&i| data[i].b2).collect();
            let candidates = vision_geometry::epipolar::essential_5point(&b1, &b2).ok()?;
            best_of(candidates)
        } else {
            None
        }
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        datum.error(model)
    }

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
        // Three coplanar first-camera rays (a vanishing triple product of their
        // bearings) make the sample degenerate.
        if sample_indices.len() < 3 {
            return false;
        }
        let b0 = data[sample_indices[0]].b1;
        let b1 = data[sample_indices[1]].b1;
        let b2 = data[sample_indices[2]].b1;
        b0.dot(&b1.cross(&b2)).abs() < 1e-9
    }
}

// --- Homography RANSAC estimator ---

struct HomographyEstimator;

impl Estimator for HomographyEstimator {
    type Datum = Correspondence2D;
    type Model = Mat3;

    const MIN_SAMPLES: usize = 4;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        let mut src = Vec::with_capacity(sample_indices.len());
        let mut dst = Vec::with_capacity(sample_indices.len());
        for &idx in sample_indices {
            src.push(data[idx].pt1);
            dst.push(data[idx].pt2);
        }
        vision_geometry::homography::dlt_homography(&src, &dst).ok()
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        residuals::symmetric_transfer_error(model, datum)
    }

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
        if sample_indices.len() < 3 {
            return false;
        }
        let p0 = data[sample_indices[0]].pt1;
        let p1 = data[sample_indices[1]].pt1;
        let p2 = data[sample_indices[2]].pt1;
        let area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        area.abs() < 1e-9
    }
}

/// The bearing of `pixel` with the covariance its pixel noise maps to.
fn bearing_and_covariance(
    model: &vision_calibration_core::CameraModel,
    pixel: &vision_calibration_core::Pt2,
    noise: &vision_calibration_core::PointCovariance,
    index: usize,
) -> Result<(Vec3, Mat3)> {
    let (b, j) = bearing_with_jacobian(model, pixel, index)?;
    let sigma = Matrix2::new(noise.xx(), noise.xy(), noise.xy(), noise.yy());
    Ok((b, j * sigma * j.transpose()))
}

/// The matches as bearing pairs through the rig cameras `cameras`.
pub(crate) fn bearing_pairs(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[PixelMatch],
) -> Result<Vec<BearingPair>> {
    let (model1, model2) = (
        camera_model(rig, cameras[0])?,
        camera_model(rig, cameras[1])?,
    );
    matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let (b1, m1) = bearing_and_covariance(model1, &m.pt1, &m.noise1, i)?;
            let (b2, m2) = bearing_and_covariance(model2, &m.pt2, &m.noise2, i)?;
            Ok(BearingPair { b1, b2, m1, m2 })
        })
        .collect()
}

/// Estimate an essential matrix from matched pixels using RANSAC.
///
/// `matches[i].pt1` is a pixel of rig camera `cameras[0]` in the first
/// capture and `matches[i].pt2` of `cameras[1]` in the second (`[0, 0]` for one
/// moving camera). Pixels are unprojected through each camera's own model into
/// bearings; the 5-point algorithm is the minimal solver and the
/// covariance-weighted Sampson error the residual: the epipolar constraint
/// divided by its first-order standard deviation, with each pixel's stated
/// covariance mapped onto its bearing through the camera model's projection
/// Jacobian. The consensus search scores minimal-sample models; the consensus
/// set is then fitted linearly and refined as a relative pose to the least
/// squares of that error (see the [module docs](self)), and the inliers are
/// re-selected and the fit refined again until they settle: the result is
/// the estimate of its own inliers, with the covariance of `E`.
///
/// **Threshold unit:** `opts.thresh` is in **σ**, dimensionless, and does not
/// depend on the focal length or the camera model: with Gaussian noise as
/// stated, `3.0` keeps about 99.7 % of the inliers. The stated noise is the
/// scale: a pixel noise understated by a factor of two doubles every residual.
/// `inlier_rms` of the result is the RMS of the same error over the inliers,
/// about 1 when the stated noise describes the pixels.
///
/// The refinement reports each iteration to `control`'s observers, and an
/// observer cannot change the result.
///
/// # Errors
///
/// [`Error::Core`] if `opts.thresh` is not finite and positive,
/// [`Error::InsufficientData`] for fewer than 5 matches,
/// [`Error::InvalidInput`] for a camera index out of range or a pixel with no
/// finite viewing ray, [`Error::NoConsensus`] if no model reaches
/// `opts.min_inliers`, and [`Error::Cancelled`] when `control`'s token is
/// cancelled.
pub fn estimate_essential(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[PixelMatch],
    opts: &RansacOptions,
    control: &RunControl,
) -> Result<EssentialEstimate> {
    let pairs = checked_pairs(rig, cameras, matches, opts)?;
    let settled = settle(&pairs, opts, control)?;
    Ok(EssentialEstimate {
        essential: settled.fit.motion.essential(),
        uncertainty: essential_uncertainty(&settled.fit),
        inlier_rms: settled.inlier_rms,
        inliers: settled.inliers,
    })
}

/// The matches as bearing pairs, after the checks every robust two-view
/// estimate makes before any work.
fn checked_pairs(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[PixelMatch],
    opts: &RansacOptions,
) -> Result<Vec<BearingPair>> {
    opts.validate()?;
    if matches.len() < 5 {
        return Err(Error::InsufficientData {
            need: 5,
            got: matches.len(),
        });
    }
    bearing_pairs(rig, cameras, matches)
}

/// The settled two-view estimate: the refined motion, the inliers it is
/// fitted to and their RMS error.
struct Settled {
    fit: Fit,
    inliers: Vec<usize>,
    inlier_rms: Real,
}

/// RANSAC, then the refinement of the consensus set, re-selected until it
/// settles.
fn settle(pairs: &[BearingPair], opts: &RansacOptions, control: &RunControl) -> Result<Settled> {
    let result = ransac_fit::<EssentialEstimator>(pairs, opts);
    if !result.success {
        return Err(Error::NoConsensus);
    }
    let minimal = result.model.expect("success guarantees model");
    let mut inliers = result.inliers;

    // The consensus set came from a minimal sample: fit it linearly, start
    // from the decomposition with the points ahead of both cameras, refine,
    // and re-select the inliers until the set settles, so the result is the
    // refined model of its own inliers. A re-selection that shrinks the set
    // ends the rounds, keeping the fit of the set before it.
    let linear = EssentialEstimator::fit(pairs, &inliers);
    let start = match linear.map(|e| physical_motion(&e, pairs, &inliers)) {
        Some(Ok(motion)) => motion,
        _ => physical_motion(&minimal, pairs, &inliers)?,
    };
    let mut fit = refine(pairs, &inliers, &start, control)?;
    for _ in 1..FINAL_REFITS {
        let e = fit.motion.essential();
        let next: Vec<usize> = (0..pairs.len())
            .filter(|&i| pairs[i].error(&e) <= opts.thresh)
            .collect();
        if next == inliers || next.len() < inliers.len().max(opts.min_inliers) {
            break;
        }
        inliers = next;
        fit = refine(pairs, &inliers, &fit.motion, control)?;
    }
    let inlier_rms = rms(pairs, &inliers, &fit.motion.essential());
    Ok(Settled {
        fit,
        inliers,
        inlier_rms,
    })
}

/// Refinements of the consensus set, each on the inliers the previous one
/// selects; the set settles in one or two.
const FINAL_REFITS: usize = 5;

/// Estimate a homography from correspondences using RANSAC.
///
/// Uses the 4-point DLT as the minimal solver and symmetric transfer
/// error as the residual metric.
///
/// **Threshold unit:** `opts.thresh` is compared with the (non-squared)
/// symmetric transfer error `sqrt(|H x1 - x2|^2 + |H^-1 x2 - x1|^2)`, in the
/// unit of the input points (pixels for pixel correspondences). The two
/// transfer directions add in quadrature, so a point with an error `e` in each
/// direction scores `sqrt(2) e`.
///
/// # Errors
///
/// [`Error::Core`] if `opts.thresh` is not finite and positive,
/// [`Error::InsufficientData`] for fewer than 4 correspondences and
/// [`Error::NoConsensus`] if no model reaches `opts.min_inliers`.
pub fn estimate_homography(
    corrs: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<HomographyEstimate> {
    opts.validate()?;
    if corrs.len() < 4 {
        return Err(Error::InsufficientData {
            need: 4,
            got: corrs.len(),
        });
    }

    let result = ransac_fit::<HomographyEstimator>(corrs, opts);
    if !result.success {
        return Err(Error::NoConsensus);
    }

    Ok(HomographyEstimate {
        homography: result.model.expect("success guarantees model"),
        inliers: result.inliers,
        inlier_rms: result.inlier_rms,
    })
}

/// Recover relative pose robustly from matched pixels.
///
/// Combines essential matrix RANSAC and its refinement (as
/// [`estimate_essential`], with the same arguments: the rig, the camera of
/// each capture, the matches and the run control) with cheirality: the
/// refinement starts from the decomposition of `E` that puts the most inliers
/// ahead of both cameras, so the refined parameters are the pose. The
/// returned `second_se3_first` has a unit-norm translation, and its
/// covariance is in that gauge (see the [module docs](self)).
///
/// **Threshold unit:** as for [`estimate_essential`], `opts.thresh` and
/// `inlier_rms` are in σ of the stated pixel noise.
///
/// # Errors
///
/// The errors of [`estimate_essential`], plus the cheirality errors of the
/// pose decomposition.
pub fn recover_relative_pose_robust(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[PixelMatch],
    opts: &RansacOptions,
    control: &RunControl,
) -> Result<RobustRelativePose> {
    let pairs = checked_pairs(rig, cameras, matches, opts)?;
    let settled = settle(&pairs, opts, control)?;
    let fit = ahead_of_both(&pairs, &settled.inliers, settled.fit, control)?;
    Ok(RobustRelativePose {
        second_se3_first: fit.motion.iso(),
        essential: fit.motion.essential(),
        inlier_rms: rms(&pairs, &settled.inliers, &fit.motion.essential()),
        inliers: settled.inliers,
        uncertainty: fit.uncertainty,
    })
}

/// `fit` as the pose of the decomposition of its `E` with the most of the
/// `inliers` ahead of both cameras.
///
/// Cheirality on the inliers only, so outliers don't corrupt the
/// positive-depth vote. The refinement started from the winning
/// decomposition and does not leave it, so `fit` is returned as it is;
/// should another decomposition win (a start with no point ahead), the pose
/// is refined again from it, so the covariance is the pose's.
fn ahead_of_both(
    pairs: &[BearingPair],
    inliers: &[usize],
    fit: Fit,
    control: &RunControl,
) -> Result<Fit> {
    let (b1, b2) = bearings(pairs, inliers);
    let candidates = vision_geometry::epipolar::decompose_essential(&fit.motion.essential())?;
    let (r, t, best) = select_pose(&candidates, &b1, &b2).ok_or(Error::NoValidPose)?;
    let own = cheirality_count(
        &fit.motion.rotation_matrix(),
        &fit.motion.direction,
        &b1,
        &b2,
    );
    if best > own {
        refine(pairs, inliers, &Motion::new(&r, &t), control)
    } else {
        Ok(fit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose_recovery::iso_from_rt;
    use crate::test_support::{Lcg, iso, match_of, pinhole_rig, pixel_of};
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3};

    /// The ground-truth pose of the second view.
    fn gt_pose() -> Iso3 {
        iso(0.05, -0.03, 0.02, Vec3::new(0.3, 0.01, 0.005))
    }

    fn rig() -> CalibratedRig {
        // Distortion makes the pixel path matter.
        pinhole_rig(-0.15)
    }

    /// 15 noise-free matches of an 800 px camera plus 3 gross outliers.
    fn stereo_matches_with_outliers() -> Vec<PixelMatch> {
        let rig = rig();
        let gt = gt_pose();
        let world = [
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
            Pt3::new(-0.3, -0.4, 3.5),
            Pt3::new(0.1, 0.6, 4.5),
            Pt3::new(0.4, -0.5, 6.0),
            Pt3::new(-0.2, 0.3, 3.0),
            Pt3::new(0.3, 0.5, 5.5),
            Pt3::new(-0.5, -0.1, 4.0),
            Pt3::new(0.2, -0.4, 3.2),
            Pt3::new(0.0, 0.0, 2.5),
            Pt3::new(-0.3, 0.5, 4.8),
            Pt3::new(0.4, 0.1, 3.8),
            Pt3::new(-0.1, -0.3, 5.2),
            Pt3::new(0.5, -0.2, 4.3),
        ];
        let mut matches: Vec<_> = world
            .iter()
            .map(|p| {
                match_of(
                    pixel_of(&rig, 0, &p.coords),
                    pixel_of(&rig, 0, &(gt * p).coords),
                )
            })
            .collect();
        // Add outliers.
        matches.push(match_of(Pt2::new(700.0, 500.0), Pt2::new(100.0, 90.0)));
        matches.push(match_of(Pt2::new(400.0, 520.0), Pt2::new(900.0, 100.0)));
        matches.push(match_of(Pt2::new(660.0, 400.0), Pt2::new(300.0, 700.0)));
        matches
    }

    fn rotation_error_deg(a: &Iso3, b: &Iso3) -> f64 {
        a.rotation.angle_to(&b.rotation).to_degrees()
    }

    #[test]
    fn essential_ransac_with_outliers() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions::new(3.0)
            .with_max_iters(500)
            .with_min_inliers(10)
            .with_confidence(0.99)
            .with_seed(42)
            .with_refit_on_inliers(true);

        let est =
            estimate_essential(&rig, [0, 0], &matches, &opts, &RunControl::default()).unwrap();
        assert!(
            est.inliers.len() >= 12,
            "expected ≥12 inliers, got {}",
            est.inliers.len()
        );
        assert!(est.inliers.iter().all(|&i| i < 15), "an outlier survived");
        assert!(est.inlier_rms < 1e-3, "inlier rms {} σ", est.inlier_rms);

        // Verify decomposition recovers pose (using inliers only).
        let pairs = bearing_pairs(&rig, [0, 0], &matches).unwrap();
        let b1: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b1).collect();
        let b2: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b2).collect();
        let (r_est, t_est) =
            crate::cheirality::recover_pose_from_essential(&est.essential, &b1, &b2).unwrap();
        let est_pose = iso_from_rt(&r_est, &t_est);
        assert!(rotation_error_deg(&est_pose, &gt_pose()) < 0.1);
        let cos_t = t_est
            .normalize()
            .dot(&gt_pose().translation.vector.normalize());
        assert!(
            (cos_t - 1.0).abs() < 1e-3,
            "translation direction: cos = {cos_t}"
        );
    }

    /// The 15 noise-free inliers of the fixture plus 6 gross outliers (29 %);
    /// the inliers keep indices `0..15`.
    fn thirty_percent_outliers() -> (Vec<PixelMatch>, usize) {
        const N_IN: usize = 15;
        let mut matches = stereo_matches_with_outliers();
        matches.truncate(N_IN);
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        for _ in 0..6 {
            let mut px = || Pt2::new(640.0 + 400.0 * rng.next(), 480.0 + 300.0 * rng.next());
            let (a, b) = (px(), px());
            matches.push(match_of(a, b));
        }
        (matches, N_IN)
    }

    #[test]
    fn essential_ransac_rejects_outliers_at_sigma_threshold() {
        let rig = rig();
        let (matches, n_in) = thirty_percent_outliers();
        let opts = RansacOptions::new(3.0)
            .with_max_iters(2000)
            .with_min_inliers(8)
            .with_seed(7);
        let est =
            estimate_essential(&rig, [0, 0], &matches, &opts, &RunControl::default()).unwrap();
        assert_eq!(est.inliers.len(), n_in, "inlier count");
        assert!(est.inliers.iter().all(|&i| i < n_in), "an outlier survived");

        // A threshold of 1e5 σ accepts everything.
        let loose = RansacOptions::new(1.0e5)
            .with_max_iters(50)
            .with_min_inliers(8)
            .with_seed(7);
        let est =
            estimate_essential(&rig, [0, 0], &matches, &loose, &RunControl::default()).unwrap();
        assert_eq!(est.inliers.len(), matches.len());
    }

    #[test]
    fn ransac_threshold_must_be_finite_and_positive() {
        let rig = rig();
        let (matches, _) = thirty_percent_outliers();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let opts = RansacOptions::new(bad);
            assert!(
                matches!(
                    estimate_essential(&rig, [0, 0], &matches, &opts, &RunControl::default()),
                    Err(Error::Core(_))
                ),
                "essential accepted thresh {bad}"
            );
            let plain: Vec<_> = matches
                .iter()
                .map(|m| Correspondence2D::new(m.pt1, m.pt2))
                .collect();
            assert!(
                matches!(estimate_homography(&plain, &opts), Err(Error::Core(_))),
                "homography accepted thresh {bad}"
            );
            assert!(
                matches!(
                    recover_relative_pose_robust(
                        &rig,
                        [0, 0],
                        &matches,
                        &opts,
                        &RunControl::default()
                    ),
                    Err(Error::Core(_))
                ),
                "pose accepted thresh {bad}"
            );
        }
    }

    #[test]
    fn camera_index_out_of_range_is_invalid_input() {
        let (matches, _) = thirty_percent_outliers();
        let opts = RansacOptions::new(3.0);
        assert!(matches!(
            estimate_essential(&rig(), [0, 1], &matches, &opts, &RunControl::default()),
            Err(Error::InvalidInput { .. })
        ));
    }

    #[test]
    fn homography_ransac_with_outliers() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let d = 3.0;
        let h = crate::homography::homography_from_pose_and_plane(&r, &t, &n, d);

        let plane_pts = [
            Pt3::new(0.5, 0.3, d),
            Pt3::new(-0.4, 0.2, d),
            Pt3::new(0.6, -0.3, d),
            Pt3::new(-0.3, -0.4, d),
            Pt3::new(0.1, 0.6, d),
            Pt3::new(0.4, -0.5, d),
            Pt3::new(-0.2, 0.3, d),
            Pt3::new(0.3, 0.5, d),
            Pt3::new(-0.5, -0.1, d),
            Pt3::new(0.2, -0.4, d),
        ];

        let mut corrs: Vec<_> = plane_pts
            .iter()
            .map(|pw| {
                let pt1 = Pt2::new(pw.x / pw.z, pw.y / pw.z);
                let pt2 = crate::homography::homography_transfer(&h, &pt1);
                Correspondence2D::new(pt1, pt2)
            })
            .collect();

        // Add outliers.
        corrs.push(Correspondence2D::new(
            Pt2::new(0.1, 0.2),
            Pt2::new(5.0, -3.0),
        ));
        corrs.push(Correspondence2D::new(
            Pt2::new(-0.3, 0.1),
            Pt2::new(-2.0, 4.0),
        ));

        let opts = RansacOptions::new(0.001)
            .with_max_iters(500)
            .with_min_inliers(6)
            .with_confidence(0.99)
            .with_seed(42)
            .with_refit_on_inliers(true);

        let est = estimate_homography(&corrs, &opts).unwrap();
        assert!(
            est.inliers.len() >= 8,
            "expected ≥8 inliers, got {}",
            est.inliers.len()
        );
        assert!(
            est.inlier_rms < 1e-6,
            "inlier RMS too large: {}",
            est.inlier_rms
        );
    }

    #[test]
    fn recover_pose_robust_with_outliers() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions::new(3.0)
            .with_max_iters(500)
            .with_min_inliers(10)
            .with_confidence(0.99)
            .with_seed(42)
            .with_refit_on_inliers(true);

        let result =
            recover_relative_pose_robust(&rig, [0, 0], &matches, &opts, &RunControl::default())
                .unwrap();

        assert!(rotation_error_deg(&result.second_se3_first, &gt_pose()) < 0.1);
        let t = result.second_se3_first.translation.vector;
        assert!((t.norm() - 1.0).abs() < 1e-12);
        let cos_t = t.dot(&gt_pose().translation.vector.normalize());
        assert!(
            (cos_t - 1.0).abs() < 1e-3,
            "translation direction: cos = {cos_t}"
        );
        assert!(result.inliers.len() >= 12);
    }

    #[test]
    fn essential_ransac_deterministic_with_same_seed() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions::new(3.0)
            .with_max_iters(200)
            .with_min_inliers(10)
            .with_confidence(0.99)
            .with_seed(123)
            .with_refit_on_inliers(true);

        let r1 = estimate_essential(&rig, [0, 0], &matches, &opts, &RunControl::default()).unwrap();
        let r2 = estimate_essential(&rig, [0, 0], &matches, &opts, &RunControl::default()).unwrap();

        assert_eq!(r1.inliers, r2.inliers);
        assert!((r1.inlier_rms - r2.inlier_rms).abs() < 1e-15);
    }

    /// A fit of another decomposition of `E` than the one ahead of both
    /// cameras is refined again from that one, so the pose and its
    /// covariance are the pose's; the physical fit is kept as it is.
    #[test]
    fn the_pose_is_the_decomposition_ahead_of_both_cameras() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();
        let pairs = bearing_pairs(&rig, [0, 0], &matches).unwrap();
        let inliers: Vec<usize> = (0..15).collect();
        let gt = gt_pose();
        let truth = Motion::new(
            gt.rotation.to_rotation_matrix().matrix(),
            &gt.translation.vector,
        );
        let control = RunControl::default();
        let (b1, b2) = bearings(&pairs, &inliers);
        let candidates =
            vision_geometry::epipolar::decompose_essential(&truth.essential()).unwrap();
        // Each decomposition that is not the truth: a twisted rotation or a
        // flipped direction.
        let others: Vec<Motion> = candidates
            .iter()
            .map(|c| Motion::new(&c.rotation, &c.translation))
            .filter(|m| {
                m.rotation.angle_to(&truth.rotation) > 1e-3
                    || m.direction.dot(&truth.direction) < 0.0
            })
            .collect();
        assert_eq!(others.len(), 3);
        for wrong in others {
            assert!(cheirality_count(&wrong.rotation_matrix(), &wrong.direction, &b1, &b2) < 15);
            let fit = Fit {
                motion: wrong,
                uncertainty: Uncertainty::default(),
            };
            let pose = ahead_of_both(&pairs, &inliers, fit, &control).unwrap();
            assert!(pose.motion.rotation.angle_to(&truth.rotation) < 1e-6);
            assert!((pose.motion.direction - truth.direction).norm() < 1e-6);
            let labels = &pose.uncertainty.covariance().expect("covariance").labels;
            assert_eq!(labels[0], "rotation.rx");
        }
        // The physical fit comes back unchanged.
        let fit = Fit {
            motion: truth,
            uncertainty: Uncertainty::default(),
        };
        let kept = ahead_of_both(&pairs, &inliers, fit, &control).unwrap();
        assert_eq!(kept.motion.rotation, truth.rotation);
        assert!(kept.uncertainty.covariance().is_none());
    }

    /// Regression test for the RANSAC refit no-op bug.
    ///
    /// When `refit_on_inliers: true` and the consensus set has >5 points, the
    /// old code called `essential_5point` with all inliers → `Err` → silent
    /// no-op (model tied to a single 5-point sample).  The fix routes the
    /// overdetermined case to `essential_linear`, which accepts any ≥5 points.
    #[test]
    fn essential_ransac_refit_on_inliers_uses_linear_solver() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts_refit = RansacOptions::new(3.0)
            .with_max_iters(500)
            .with_min_inliers(10)
            .with_confidence(0.99)
            .with_seed(77)
            .with_refit_on_inliers(true);

        let est = estimate_essential(&rig, [0, 0], &matches, &opts_refit, &RunControl::default())
            .expect("RANSAC with refit must succeed");
        assert!(
            est.inliers.len() >= 12,
            "expected ≥12 inliers after refit, got {}",
            est.inliers.len()
        );

        // The refit model must be geometrically valid: decompose and check pose.
        let pairs = bearing_pairs(&rig, [0, 0], &matches).unwrap();
        let b1: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b1).collect();
        let b2: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b2).collect();
        let (r_est, t_est) =
            crate::cheirality::recover_pose_from_essential(&est.essential, &b1, &b2)
                .expect("pose decomposition must succeed");
        assert!(rotation_error_deg(&iso_from_rt(&r_est, &t_est), &gt_pose()) < 0.1);
        let cos_t = t_est
            .normalize()
            .dot(&gt_pose().translation.vector.normalize());
        assert!((cos_t - 1.0).abs() < 1e-3, "cos_t = {cos_t}");
    }

    /// The 6-/7-point dispatch path in `EssentialEstimator::fit` produces a
    /// valid essential matrix on bearings, including ones beyond 90° off the
    /// axis.
    #[test]
    fn essential_fit_7pts_uses_5point_scored_on_all() {
        let rot = Rotation3::from_euler_angles(0.08, -0.04, 0.15);
        let t = Vec3::new(0.25, 0.05, 0.01);

        let world = [
            Pt3::new(0.4, 0.2, 3.0),
            Pt3::new(-0.3, 0.1, 4.0),
            Pt3::new(0.5, -0.2, 3.5),
            Pt3::new(-0.2, -0.3, 2.8),
            Pt3::new(0.1, 0.4, 4.5),
            Pt3::new(0.3, -0.4, 5.0),
            Pt3::new(-0.4, 0.3, -3.2),
        ];

        let pairs: Vec<BearingPair> = world
            .iter()
            .map(|pw| BearingPair {
                b1: pw.coords.normalize(),
                b2: (rot * pw.coords + t).normalize(),
                m1: Mat3::identity() * 1e-6,
                m2: Mat3::identity() * 1e-6,
            })
            .collect();

        let indices: Vec<usize> = (0..7).collect();
        let e =
            EssentialEstimator::fit(&pairs, &indices).expect("fit with 7 points must return Some");

        // 1. Weighted Sampson error near zero for all 7 points.
        assert!(e.norm() > 0.0, "fit returned zero matrix");
        for p in &pairs {
            let res = p.error(&e);
            assert!(res < 1e-5, "epipolar residual too large: {:.3e}", res);
        }

        // 2. Essential manifold check: singular values ≈ (σ, σ, 0).
        let sv = e.svd(false, false).singular_values;
        let s_avg = (sv[0] + sv[1]) * 0.5;
        assert!(
            (sv[0] / s_avg - 1.0).abs() < 1e-4,
            "sv[0] not close to sv[1]: {:?}",
            sv
        );
        assert!(sv[2] / s_avg < 1e-4, "sv[2] not near zero: {:?}", sv);
    }

    /// Monte Carlo: with pixels drawn from the stated covariances (a fisheye
    /// with an anisotropic, correlated one; a pinhole with an isotropic one),
    /// the weighted Sampson error at the true `E` has an RMS of 1 σ, and it
    /// halves when every stated σ is doubled.
    #[test]
    fn weighted_error_at_the_true_essential_matrix_is_one_sigma() {
        use crate::test_support::{fisheye_params, pinhole_params};
        use vision_calibration_core::{PointCovariance, Uncertainty};

        let cam1_se3_rig = iso(0.02, -0.12, 0.03, Vec3::new(-0.5, 0.05, 0.1));
        let rig = CalibratedRig::new(
            vec![fisheye_params(), pinhole_params(-0.2)],
            vec![Iso3::identity(), cam1_se3_rig],
            Uncertainty::default(),
        )
        .unwrap();
        let truth = [
            PointCovariance::new(0.25, 0.08, 0.09).unwrap(),
            PointCovariance::isotropic(0.4).unwrap(),
        ];
        let mut rng = Lcg(0xC0FFEE);
        let mut uniform = || (rng.next() + 1.0) * 0.5 * (1.0 - 1e-12) + 5e-13;
        let mut normal = || {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        };
        let mut offset = |c: &PointCovariance| {
            let l00 = c.xx().sqrt();
            let l10 = c.xy() / l00;
            let l11 = (c.yy() - l10 * l10).sqrt();
            let (a, b) = (normal(), normal());
            Pt2::new(l00 * a, l10 * a + l11 * b)
        };

        let mut obs = Vec::new();
        for i in 0..4000 {
            let z = 3.0 + 5.0 * (f64::from(i) * 0.618_034).fract();
            let x = z * (0.6 * ((f64::from(i) * 0.371).fract() - 0.5));
            let y = z * (0.5 * ((f64::from(i) * 0.713).fract() - 0.5));
            let p = Pt3::new(x, y, z);
            let (m0, m1) = (rig.model(0).unwrap(), rig.model(1).unwrap());
            if let (Some(u1), Some(u2)) =
                (m0.project_point(&p), m1.project_point(&(cam1_se3_rig * p)))
            {
                obs.push((u1 + offset(&truth[0]).coords, u2 + offset(&truth[1]).coords));
            }
        }
        assert!(obs.len() >= 3000, "{} matches", obs.len());

        let t = cam1_se3_rig.translation.vector;
        let skew = Mat3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
        let e = skew * cam1_se3_rig.rotation.to_rotation_matrix().into_inner();

        let rms_at_true_e = |scale: f64| -> f64 {
            let state = |c: &PointCovariance| {
                let s2 = scale * scale;
                PointCovariance::new(c.xx() * s2, c.xy() * s2, c.yy() * s2).unwrap()
            };
            let matches: Vec<_> = obs
                .iter()
                .map(|&(a, b)| PixelMatch::new(a, state(&truth[0]), b, state(&truth[1])))
                .collect();
            let pairs = bearing_pairs(&rig, [0, 1], &matches).unwrap();
            let sum: Real = pairs.iter().map(|p| p.error(&e).powi(2)).sum();
            (sum / pairs.len() as Real).sqrt()
        };
        let rms = rms_at_true_e(1.0);
        let half = rms_at_true_e(2.0);
        eprintln!("RMS at the true E: {rms:.4} sigma; with doubled stated sigma {half:.4}");
        assert!((0.95..=1.05).contains(&rms), "RMS {rms} sigma");
        assert!((half / rms - 0.5).abs() < 1e-9, "{half} vs {rms}");
    }
}

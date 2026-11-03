//! Measurements over points with their joint covariance: distances, angles,
//! plane fits and point-to-plane distances, each with its standard deviation.
//!
//! The input is a [`PointSet`]: points and the joint covariance of all their
//! coordinates, as [`triangulate`](crate::triangulation::triangulate) and
//! [`bundle_adjust`](crate::bundle_adjust::bundle_adjust) return it. Points
//! measured with one calibration share that calibration's uncertainty, so
//! their errors are correlated, and a length between two of them is more (or
//! less) certain than independent errors would say. Every σ here uses the
//! full joint covariance, cross terms included, so all the points of a
//! measurement must come from one call that produced them.
//!
//! A measurement `f(x)` over the coordinates `x` of its points has the
//! standard deviation `√(J Σ Jᵀ)`, with `J = ∂f/∂x` from dual numbers (exact,
//! no finite-difference step) and `Σ` the block of the joint covariance for
//! the points involved. This is the first-order (linearized) propagation:
//! it holds while the points' errors are small against the geometry they
//! measure.
//!
//! # Example
//!
//! ```
//! use nalgebra::DMatrix;
//! use vision_calibration_core::Pt3;
//! use vision_mvg::metrology::{PointSet, distance};
//!
//! // Two points 0.1 m apart, each known to 0.2 mm per axis, independently.
//! let points = vec![Pt3::new(0.0, 0.0, 0.7), Pt3::new(0.1, 0.0, 0.7)];
//! let set = PointSet::new(points, DMatrix::identity(6, 6) * 4.0e-8)?;
//! let d = distance(&set, 0, 1)?;
//! assert!((d.value - 0.1).abs() < 1e-12);
//! // Only the error along the segment counts: sqrt(2) * 0.2 mm.
//! assert!((d.sigma - 2.0_f64.sqrt() * 2.0e-4).abs() < 1e-9);
//! # Ok::<(), vision_mvg::MvgError>(())
//! ```

use crate::bundle_adjust::BundleAdjustment;
use crate::laser::LaserTriangulation;
use crate::triangulation::Triangulation;
use crate::{MvgError, Result};
use nalgebra::{DMatrix, DVector, RealField, SMatrix, SVector, SymmetricEigen, Vector3};
use num_dual::{DualDVec64, DualSVec64, gradient, jacobian};
use serde::Serialize;
use vision_calibration_core::{Mat3, Pt3, Real, Uncertainty, Vec3};

/// The smallest scatter eigenvalue ratio below which points count as collinear.
const COLLINEAR_RATIO: Real = 1e-12;

/// A measured scalar and its standard deviation, in the measurement's unit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[non_exhaustive]
pub struct Scalar {
    /// The measured value.
    pub value: Real,
    /// Its standard deviation, from the points' joint covariance.
    pub sigma: Real,
}

/// Points with their joint covariance, the input of every measurement.
#[derive(Debug, Clone)]
pub struct PointSet {
    points: Vec<Pt3>,
    /// 3N × 3N, rows and columns ordered `p0.x, p0.y, p0.z, p1.x, ...`, m².
    covariance: DMatrix<Real>,
}

impl PointSet {
    /// Points and the joint covariance of their coordinates (3N × 3N, m²,
    /// ordered `x, y, z` per point).
    ///
    /// The covariance must be finite, symmetric (within 1e-9 of its largest
    /// entry) and have a non-negative diagonal.
    pub fn new(points: Vec<Pt3>, covariance: DMatrix<Real>) -> Result<Self> {
        let n = 3 * points.len();
        if covariance.nrows() != n || covariance.ncols() != n {
            return Err(MvgError::InvalidInput {
                reason: format!(
                    "{} points need a {n}x{n} covariance, got {}x{}",
                    points.len(),
                    covariance.nrows(),
                    covariance.ncols()
                ),
            });
        }
        if points
            .iter()
            .any(|p| !p.coords.iter().all(|c| c.is_finite()))
        {
            return Err(MvgError::InvalidInput {
                reason: "a point is not finite".into(),
            });
        }
        if !covariance.iter().all(|c| c.is_finite()) {
            return Err(MvgError::InvalidInput {
                reason: "the covariance is not finite".into(),
            });
        }
        let scale = covariance.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for r in 0..n {
            if covariance[(r, r)] < 0.0 {
                return Err(MvgError::InvalidInput {
                    reason: format!("the covariance has a negative variance at row {r}"),
                });
            }
            for c in 0..r {
                if (covariance[(r, c)] - covariance[(c, r)]).abs() > 1e-9 * scale {
                    return Err(MvgError::InvalidInput {
                        reason: format!("the covariance is not symmetric at ({r}, {c})"),
                    });
                }
            }
        }
        Ok(Self { points, covariance })
    }

    /// Points with independent errors: `covariances[i]` is point `i`'s 3×3
    /// covariance and the cross terms are zero.
    pub fn independent(points: Vec<Pt3>, covariances: &[Mat3]) -> Result<Self> {
        if points.len() != covariances.len() {
            return Err(MvgError::CountMismatch {
                expected: points.len(),
                got: covariances.len(),
            });
        }
        let mut covariance = DMatrix::zeros(3 * points.len(), 3 * points.len());
        for (i, c) in covariances.iter().enumerate() {
            covariance.fixed_view_mut::<3, 3>(3 * i, 3 * i).copy_from(c);
        }
        Self::new(points, covariance)
    }

    /// The points of a triangulation with their joint covariance. Every point
    /// must have one: otherwise an `InvalidInput` names the first point
    /// without and, when the whole uncertainty is unavailable, the reason.
    pub fn from_triangulation(triangulation: &Triangulation) -> Result<Self> {
        let points = triangulation.points.iter().map(|p| p.point).collect();
        Self::from_uncertainty(points, &triangulation.uncertainty)
    }

    /// The points of a laser triangulation (rig frame) with their joint
    /// covariance. As for [`from_triangulation`](Self::from_triangulation).
    pub fn from_laser_triangulation(laser: &LaserTriangulation) -> Result<Self> {
        let points = laser.points.iter().map(|p| p.point).collect();
        Self::from_uncertainty(points, &laser.uncertainty)
    }

    /// The points of a bundle adjustment with their joint covariance, in the
    /// adjustment's gauge. As for [`from_triangulation`](Self::from_triangulation).
    pub fn from_bundle_adjustment(adjustment: &BundleAdjustment) -> Result<Self> {
        Self::from_uncertainty(adjustment.points.clone(), &adjustment.uncertainty)
    }

    /// Reads the joint covariance of `p{i}.x/y/z` out of `uncertainty`.
    fn from_uncertainty(points: Vec<Pt3>, uncertainty: &Uncertainty) -> Result<Self> {
        let Some(c) = uncertainty.covariance() else {
            let reason = match uncertainty {
                Uncertainty::Unavailable { reason } => format!("{reason:?}"),
                Uncertainty::Available(_) => {
                    unreachable!("an available uncertainty has a covariance")
                }
            };
            return Err(MvgError::InvalidInput {
                reason: format!(
                    "point 0 has no covariance: the uncertainty is unavailable ({reason})"
                ),
            });
        };
        let mut index = Vec::with_capacity(3 * points.len());
        for i in 0..points.len() {
            for axis in ["x", "y", "z"] {
                let label = format!("p{i}.{axis}");
                index.push(c.index(&label).ok_or_else(|| MvgError::InvalidInput {
                    reason: format!("point {i} has no covariance ({label} is not in it)"),
                })?);
            }
        }
        let covariance = DMatrix::from_fn(index.len(), index.len(), |r, s| {
            c.covariance[index[r]][index[s]]
        });
        Self::new(points, covariance)
    }

    /// The number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the set has no point.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Point `i`; `None` when out of range.
    pub fn point(&self, i: usize) -> Option<Pt3> {
        self.points.get(i).copied()
    }

    /// Point `i`'s own 3×3 covariance, m²; `None` when out of range.
    pub fn covariance(&self, i: usize) -> Option<Mat3> {
        (i < self.len()).then(|| {
            self.covariance
                .fixed_view::<3, 3>(3 * i, 3 * i)
                .into_owned()
        })
    }

    /// The joint covariance of the points `involved`, in that order.
    fn block(&self, involved: &[usize]) -> DMatrix<Real> {
        DMatrix::from_fn(3 * involved.len(), 3 * involved.len(), |r, s| {
            self.covariance[(3 * involved[r / 3] + r % 3, 3 * involved[s / 3] + s % 3)]
        })
    }

    /// `√(g Σ gᵀ)` for a gradient over the coordinates of `involved`.
    fn sigma(&self, involved: &[usize], gradient: &DVector<Real>) -> Real {
        let sigma = self.block(involved);
        (gradient.dot(&(sigma * gradient))).max(0.0).sqrt()
    }

    /// Checks `indices` are in range and distinct.
    fn check_indices(&self, indices: &[usize]) -> Result<()> {
        for (k, &i) in indices.iter().enumerate() {
            if i >= self.len() {
                return Err(MvgError::InvalidInput {
                    reason: format!("point index {i} is out of range for {} points", self.len()),
                });
            }
            if indices[..k].contains(&i) {
                return Err(MvgError::InvalidInput {
                    reason: format!("point index {i} is repeated"),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scalar functions, written once for every scalar type
// ---------------------------------------------------------------------------

fn v3<T: RealField>(x: &[T], at: usize) -> Vector3<T> {
    Vector3::new(x[at].clone(), x[at + 1].clone(), x[at + 2].clone())
}

fn lift<T: RealField>(v: &Vec3) -> Vector3<T> {
    v.map(|c| T::from_f64(c).expect("an f64 converts to every RealField"))
}

/// Distance between the points `x[0..3]` and `x[3..6]`.
fn distance_fn<T: RealField>(x: &[T]) -> T {
    (v3(x, 0) - v3(x, 3)).norm()
}

/// Angle at `x[3..6]` between the rays to `x[0..3]` and `x[6..9]`, as the
/// half-angle form `2 atan(|u × w| / (|u||w| + u·w))`: it keeps its slope
/// near π/2, where `acos` and the dual-number `atan2` (an `atan` of a ratio)
/// both break down. Only a straight angle (0 or π) is singular.
fn angle_fn<T: RealField>(x: &[T]) -> T {
    let u = v3(x, 0) - v3(x, 3);
    let w = v3(x, 6) - v3(x, 3);
    let two = T::from_f64(2.0).expect("an f64 converts to every RealField");
    two * (u.cross(&w).norm() / (u.norm() * w.norm() + u.dot(&w))).atan()
}

/// The plane's normal `normalize(n0 + α t1 + β t2)`.
fn plane_normal<T: RealField>(alpha: &T, beta: &T, frame: &PlaneFrame) -> (Vector3<T>, T) {
    let u = lift::<T>(&frame.n0)
        + lift::<T>(&frame.t[0]) * alpha.clone()
        + lift::<T>(&frame.t[1]) * beta.clone();
    let norm = u.norm();
    (u / norm.clone(), norm)
}

/// The gradient of `F(θ, p) = Σ (n(θ)·pᵢ − d)²` over `θ = (α, β, d)`, with
/// `x = [α, β, d, p₀, p₁, ...]`. Differentiating it over all of `x` gives
/// the two blocks of the implicit-function theorem.
fn plane_gradient<T: RealField>(x: &[T], frame: &PlaneFrame) -> DVector<T> {
    let (n, norm) = plane_normal(&x[0], &x[1], frame);
    let t = [lift::<T>(&frame.t[0]), lift::<T>(&frame.t[1])];
    // ∂n/∂α = (t1 − n (n·t1)) / |u|, likewise for β.
    let dn = t.map(|t| (t.clone() - n.clone() * n.dot(&t)) / norm.clone());
    let two = T::from_f64(2.0).expect("an f64 converts to every RealField");
    let (mut ga, mut gb, mut gd) = (T::zero(), T::zero(), T::zero());
    for i in 0..(x.len() - 3) / 3 {
        let p = v3(x, 3 + 3 * i);
        let r = n.dot(&p) - x[2].clone();
        ga += r.clone() * dn[0].dot(&p);
        gb += r.clone() * dn[1].dot(&p);
        gd += r;
    }
    DVector::from_vec(vec![two.clone() * ga, two.clone() * gb, -two * gd])
}

/// The signed distance `n(θ)·p − d`, with `x = [α, β, d, p]`.
fn signed_distance_fn<T: RealField>(x: &[T], frame: &PlaneFrame) -> T {
    let (n, _) = plane_normal(&x[0], &x[1], frame);
    n.dot(&v3(x, 3)) - x[2].clone()
}

/// Value and gradient of `f` over `N` coordinates, by dual numbers.
fn value_and_gradient<const N: usize>(
    f: impl Fn(&[DualSVec64<N>]) -> DualSVec64<N>,
    x: [Real; N],
) -> (Real, SVector<Real, N>) {
    gradient(
        |v: SVector<DualSVec64<N>, N>| f(v.as_slice()),
        &SVector::from(x),
    )
}

// ---------------------------------------------------------------------------
// Distance and angle
// ---------------------------------------------------------------------------

fn coords(set: &PointSet, indices: &[usize]) -> Vec<Real> {
    indices
        .iter()
        .flat_map(|&i| set.points[i].coords.iter().copied())
        .collect()
}

/// Value and gradient of the distance of `a` and `b`, over their six
/// coordinates.
fn distance_gradient(set: &PointSet, a: usize, b: usize) -> Result<(Real, DVector<Real>)> {
    set.check_indices(&[a, b])?;
    let x: [Real; 6] = coords(set, &[a, b])
        .try_into()
        .expect("two points are six coordinates");
    let (value, grad) = value_and_gradient(distance_fn, x);
    if value <= 0.0 {
        return Err(MvgError::Degenerate {
            reason: format!("points {a} and {b} coincide: the distance has no gradient"),
        });
    }
    Ok((value, DVector::from_column_slice(grad.as_slice())))
}

/// The distance between points `a` and `b`, metres.
pub fn distance(set: &PointSet, a: usize, b: usize) -> Result<Scalar> {
    let (value, grad) = distance_gradient(set, a, b)?;
    Ok(Scalar {
        value,
        sigma: set.sigma(&[a, b], &grad),
    })
}

/// Value and gradient of the angle at `vertex`, over the coordinates of
/// `a`, `vertex`, `b`.
fn angle_gradient(
    set: &PointSet,
    a: usize,
    vertex: usize,
    b: usize,
) -> Result<(Real, DVector<Real>)> {
    set.check_indices(&[a, vertex, b])?;
    let x: [Real; 9] = coords(set, &[a, vertex, b])
        .try_into()
        .expect("three points are nine coordinates");
    for (p, q) in [(a, vertex), (b, vertex)] {
        if (set.points[p] - set.points[q]).norm() == 0.0 {
            return Err(MvgError::Degenerate {
                reason: format!("points {p} and {q} coincide: the angle is undefined"),
            });
        }
    }
    let (u, w) = (
        set.points[a] - set.points[vertex],
        set.points[b] - set.points[vertex],
    );
    if u.cross(&w).norm() == 0.0 {
        return Err(MvgError::Degenerate {
            reason: format!(
                "points {a}, {vertex} and {b} are collinear: the angle (0 or pi) has no gradient"
            ),
        });
    }
    let (value, grad) = value_and_gradient(angle_fn, x);
    Ok((value, DVector::from_column_slice(grad.as_slice())))
}

/// The angle at `vertex` between the rays to `a` and `b`, radians, in
/// `(0, π)`. Collinear points (an angle of exactly 0 or π) are a
/// `Degenerate` error: the angle is not differentiable there.
pub fn angle(set: &PointSet, a: usize, vertex: usize, b: usize) -> Result<Scalar> {
    let (value, grad) = angle_gradient(set, a, vertex, b)?;
    Ok(Scalar {
        value,
        sigma: set.sigma(&[a, vertex, b], &grad),
    })
}

// ---------------------------------------------------------------------------
// Planes
// ---------------------------------------------------------------------------

/// The nominal normal and a tangent basis, the point the plane's
/// parameters `(α, β)` are measured from.
#[derive(Debug, Clone)]
struct PlaneFrame {
    n0: Vec3,
    t: [Vec3; 2],
}

impl PlaneFrame {
    fn new(n0: Vec3) -> Self {
        // The axis least aligned with the normal makes the best-conditioned
        // first tangent; the rest follows, so the basis is deterministic.
        let axis = (0..3)
            .min_by(|&a, &b| n0[a].abs().total_cmp(&n0[b].abs()))
            .expect("three axes");
        let mut e = Vec3::zeros();
        e[axis] = 1.0;
        let t1 = (e - n0 * n0.dot(&e)).normalize();
        let t2 = n0.cross(&t1);
        Self { n0, t: [t1, t2] }
    }
}

/// A plane `n · x = d` (unit `n`, `d` in metres) fitted to points, with its
/// covariance.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct PlaneFit {
    /// The unit normal, oriented so that `distance >= 0`.
    pub normal: Vec3,
    /// The plane's distance from the origin along `normal`, metres.
    pub distance: Real,
    /// The covariance of `[α, β, d]`: the normal's rotation along the two
    /// vectors of `tangent_basis` (radians) and the distance (metres), so its
    /// units are rad², rad·m and m². It comes from the points' joint
    /// covariance, not from how well the points fit the plane.
    pub covariance: Mat3,
    /// Two unit vectors perpendicular to `normal` (and to each other) that
    /// `covariance` is expressed in.
    pub tangent_basis: [Vec3; 2],
    /// RMS orthogonal distance of the points to the plane, metres. A planarity
    /// diagnostic; it is not part of `covariance`.
    pub rms: Real,
    /// The points the plane was fitted to, in the order given.
    pub indices: Vec<usize>,
}

/// A fitted plane and its sensitivity to the points that made it.
struct PlaneCore {
    fit: PlaneFit,
    frame: PlaneFrame,
    /// `∂θ/∂p` (3 × 3k) over the coordinates of `fit.indices`.
    sensitivity: DMatrix<Real>,
}

fn plane_core(set: &PointSet, indices: &[usize]) -> Result<PlaneCore> {
    set.check_indices(indices)?;
    let k = indices.len();
    if k < 3 {
        return Err(MvgError::InvalidInput {
            reason: format!("a plane needs at least 3 points, got {k}"),
        });
    }
    let centroid = indices
        .iter()
        .fold(Vec3::zeros(), |s, &i| s + set.points[i].coords)
        / k as Real;
    let scatter = indices
        .iter()
        .fold(SMatrix::<Real, 3, 3>::zeros(), |s, &i| {
            let q = set.points[i].coords - centroid;
            s + q * q.transpose()
        });
    let eigen = SymmetricEigen::new(scatter);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let [smallest, middle, largest] = order.map(|i| eigen.eigenvalues[i]);
    if largest <= 0.0 || middle < COLLINEAR_RATIO * largest {
        return Err(MvgError::Degenerate {
            reason: "the plane's points are collinear".into(),
        });
    }
    let mut normal = eigen.eigenvectors.column(order[0]).into_owned();
    let mut d = normal.dot(&centroid);
    if d < 0.0 {
        normal = -normal;
        d = -d;
    }
    let frame = PlaneFrame::new(normal);

    // Implicit-function theorem on g(θ, p) = ∂F/∂θ = 0 at θ = (0, 0, d):
    // ∂θ/∂p = −(∂g/∂θ)⁻¹ ∂g/∂p, both from one dual pass over (θ, p).
    let mut x = DVector::zeros(3 + 3 * k);
    x[2] = d;
    x.rows_mut(3, 3 * k)
        .copy_from(&DVector::from_vec(coords(set, indices)));
    let (_, jac) = jacobian(
        |v: DVector<DualDVec64>| plane_gradient(v.as_slice(), &frame),
        &x,
    );
    let hessian: SMatrix<Real, 3, 3> = jac.fixed_view::<3, 3>(0, 0).into_owned();
    let inverse = hessian.try_inverse().ok_or_else(|| MvgError::Degenerate {
        reason: "the plane fit is singular (its points do not determine a plane)".into(),
    })?;
    let sensitivity =
        -DMatrix::from_iterator(3, 3, inverse.iter().copied()) * jac.columns(3, 3 * k);
    if !sensitivity.iter().all(|s| s.is_finite()) {
        return Err(MvgError::Degenerate {
            reason: "the plane fit is singular (its points do not determine a plane)".into(),
        });
    }

    let block = set.block(indices);
    let cov = &sensitivity * block * sensitivity.transpose();
    let covariance = Mat3::from_fn(|r, c| 0.5 * (cov[(r, c)] + cov[(c, r)]));
    Ok(PlaneCore {
        fit: PlaneFit {
            normal,
            distance: d,
            covariance,
            tangent_basis: frame.t,
            rms: (smallest.max(0.0) / k as Real).sqrt(),
            indices: indices.to_vec(),
        },
        frame,
        sensitivity,
    })
}

/// The total-least-squares plane through the points `indices` (at least 3,
/// not collinear), with the covariance of its parameters.
///
/// The covariance holds the points as consider parameters: the plane moves
/// with the points as `∂θ/∂p` says, and `Σ_θ = (∂θ/∂p) Σ_p (∂θ/∂p)ᵀ` with
/// `Σ_p` the points' joint covariance. Points that fit the plane badly do not
/// widen it; [`PlaneFit::rms`] reports that.
pub fn fit_plane(set: &PointSet, indices: &[usize]) -> Result<PlaneFit> {
    plane_core(set, indices).map(|core| core.fit)
}

/// Value and gradient of the signed distance from point `p` to the plane
/// fitted through `plane`, over the coordinates of `involved`
/// (`plane` then `p` when `p` is not in it).
fn point_to_plane_gradient(
    set: &PointSet,
    plane: &[usize],
    p: usize,
) -> Result<(Real, Vec<usize>, DVector<Real>)> {
    set.check_indices(&[p])?;
    let core = plane_core(set, plane)?;
    let mut involved = plane.to_vec();
    let slot = involved.iter().position(|&i| i == p).unwrap_or_else(|| {
        involved.push(p);
        involved.len() - 1
    });
    let frame = &core.frame;
    let q = set.points[p].coords;
    let x = [0.0, 0.0, core.fit.distance, q.x, q.y, q.z];
    let (value, grad) = value_and_gradient(|v| signed_distance_fn(v, frame), x);
    // J = ∂s/∂p_direct + ∂s/∂θ · S.
    let mut j = DVector::zeros(3 * involved.len());
    let dtheta = grad.fixed_rows::<3>(0).into_owned();
    let through_plane =
        core.sensitivity.transpose() * DVector::from_column_slice(dtheta.as_slice());
    j.rows_mut(0, through_plane.len()).copy_from(&through_plane);
    for axis in 0..3 {
        j[3 * slot + axis] += grad[3 + axis];
    }
    Ok((value, involved, j))
}

/// The signed distance of point `p` to the plane fitted through `plane`,
/// metres, positive on the side the plane's normal points to.
///
/// The σ is joint: the plane is correlated with the points that made it, and
/// with `p`. It uses every involved point's covariance and the cross terms.
/// `p` may be one of `plane`.
pub fn point_to_plane(set: &PointSet, plane: &[usize], p: usize) -> Result<Scalar> {
    let (value, involved, grad) = point_to_plane_gradient(set, plane, p)?;
    Ok(Scalar {
        value,
        sigma: set.sigma(&involved, &grad),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic smooth pseudo-random in [-1, 1].
    fn wobble(i: usize, salt: Real) -> Real {
        ((i as Real + 1.0) * 12.9898 + salt * 78.233).sin()
    }

    fn scene() -> Vec<Pt3> {
        (0..7)
            .map(|i| {
                Pt3::new(
                    0.05 * wobble(i, 1.0),
                    0.04 * wobble(i, 2.0),
                    0.7 + 0.03 * wobble(i, 3.0),
                )
            })
            .collect()
    }

    fn with_coord(set: &PointSet, i: usize, axis: usize, h: Real) -> PointSet {
        let mut points = set.points.clone();
        points[i].coords[axis] += h;
        PointSet {
            points,
            covariance: set.covariance.clone(),
        }
    }

    fn set() -> PointSet {
        let points = scene();
        PointSet::new(points.clone(), DMatrix::identity(21, 21) * 1e-8).unwrap()
    }

    /// Central-difference gradient of `f` over the coordinates of `involved`.
    fn numeric_gradient(
        set: &PointSet,
        involved: &[usize],
        f: impl Fn(&PointSet) -> Real,
    ) -> DVector<Real> {
        let h = 1e-6;
        DVector::from_fn(3 * involved.len(), |k, _| {
            let (i, axis) = (involved[k / 3], k % 3);
            (f(&with_coord(set, i, axis, h)) - f(&with_coord(set, i, axis, -h))) / (2.0 * h)
        })
    }

    fn relative_error(a: &DVector<Real>, b: &DVector<Real>) -> Real {
        (a - b).norm() / b.norm()
    }

    #[test]
    fn dual_gradients_match_central_differences() {
        let s = set();
        let (_, g) = distance_gradient(&s, 1, 4).unwrap();
        let n = numeric_gradient(&s, &[1, 4], |s| distance(s, 1, 4).unwrap().value);
        let e_distance = relative_error(&g, &n);

        let (_, g) = angle_gradient(&s, 0, 2, 5).unwrap();
        let n = numeric_gradient(&s, &[0, 2, 5], |s| angle(s, 0, 2, 5).unwrap().value);
        let e_angle = relative_error(&g, &n);

        // A point outside the plane, and one inside it.
        let plane = [0, 1, 2, 3, 4];
        let (_, inv, g) = point_to_plane_gradient(&s, &plane, 6).unwrap();
        let n = numeric_gradient(&s, &inv, |s| point_to_plane(s, &plane, 6).unwrap().value);
        let e_outside = relative_error(&g, &n);
        let (_, inv, g) = point_to_plane_gradient(&s, &plane, 2).unwrap();
        let n = numeric_gradient(&s, &inv, |s| point_to_plane(s, &plane, 2).unwrap().value);
        let e_inside = relative_error(&g, &n);

        eprintln!(
            "gradient vs central differences, relative error: distance {e_distance:.2e}, \
             angle {e_angle:.2e}, point-to-plane {e_outside:.2e} (outside), {e_inside:.2e} (inside)"
        );
        for e in [e_distance, e_angle, e_outside, e_inside] {
            assert!(e < 1e-7, "gradient error {e:e}");
        }
    }

    #[test]
    fn plane_sensitivity_matches_refitting() {
        let s = set();
        let plane = [0, 1, 2, 3, 4, 5];
        let core = plane_core(&s, &plane).unwrap();
        let h = 1e-6;
        // θ of a refit, in the nominal frame: (t1·n', t2·n', d').
        let theta = |s: &PointSet| {
            let f = fit_plane(s, &plane).unwrap();
            let n = if f.normal.dot(&core.fit.normal) < 0.0 {
                -f.normal
            } else {
                f.normal
            };
            let sign = if f.normal.dot(&core.fit.normal) < 0.0 {
                -1.0
            } else {
                1.0
            };
            SVector::<Real, 3>::new(
                core.fit.tangent_basis[0].dot(&n),
                core.fit.tangent_basis[1].dot(&n),
                sign * f.distance,
            )
        };
        let mut worst: Real = 0.0;
        for (col, (&i, axis)) in plane
            .iter()
            .flat_map(|i| (0..3).map(move |a| (i, a)))
            .enumerate()
        {
            let fd = (theta(&with_coord(&s, i, axis, h)) - theta(&with_coord(&s, i, axis, -h)))
                / (2.0 * h);
            let analytic = core.sensitivity.column(col);
            worst = worst.max((fd - analytic).norm() / analytic.norm().max(1e-3));
        }
        eprintln!("plane sensitivity vs refitting, worst relative error {worst:.2e}");
        assert!(worst < 1e-6, "sensitivity error {worst:e}");
    }
}

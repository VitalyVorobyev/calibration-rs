//! The unit-sphere S² retraction both backends use for laser-plane normals.
//!
//! Tangent coordinates are expressed in a deterministic orthonormal basis
//! `(b1, b2)` at the base point; `plus` is the exponential map and `minus`
//! its inverse.

use nalgebra as na;

/// Build an orthonormal tangent basis (b1, b2) at x (assumed unit-ish).
///
/// Deterministic: chooses a reference axis far enough from x to avoid degeneracy,
/// then Gram–Schmidt and cross product.
#[inline]
pub(crate) fn tangent_basis<T: na::RealField>(
    x_unit: &na::Vector3<T>,
) -> (na::Vector3<T>, na::Vector3<T>) {
    // Pick a reference axis well away from x.
    let ax_x = na::Vector3::new(T::one(), T::zero(), T::zero());
    let ax_z = na::Vector3::new(T::zero(), T::zero(), T::one());

    // If |x.z| is small, z-axis is safe; otherwise use x-axis.
    let a = if x_unit[2].clone().abs() < T::from_f64(0.9).unwrap() {
        ax_z
    } else {
        ax_x
    };

    // b1 = normalize( a - x (x·a) )
    let proj = x_unit.clone() * x_unit.dot(&a);
    let mut b1 = (a - proj).normalize();
    // b2 = x × b1 (already unit if x,b1 unit and orthogonal)
    let b2 = x_unit.cross(&b1);

    // Extra guard (rare): if normalization produced NaNs due to a bad input x, fall back.
    // (We don't try too hard here; state is expected to be a unit vector.)
    if b1.norm_squared() == T::zero() {
        b1 = na::Vector3::new(T::zero(), T::one(), T::zero());
    }

    (b1, b2)
}

/// The S² exponential map at `x` (normalized first) for a step `delta` in the
/// tangent basis [`tangent_basis`]; the result is renormalized.
pub(crate) fn plus<T: na::RealField>(x: &na::Vector3<T>, delta: &na::Vector2<T>) -> na::Vector3<T> {
    const EPS: f64 = 1e-6;
    let x0 = x.normalize();
    let d = delta;

    let (b1, b2) = tangent_basis(&x0);

    // Tangent vector v in R³ at x0.
    let v = b1 * d[0].clone() + b2 * d[1].clone();
    let theta2 = v.norm_squared();

    let (cos_t, sin_over_t) = if theta2 < T::from_f64(EPS * EPS).unwrap() {
        // series:
        // cos θ ≈ 1 - θ²/2
        // sin θ / θ ≈ 1 - θ²/6
        (
            T::one() - theta2.clone() / T::from_f64(2.0).unwrap(),
            T::one() - theta2 / T::from_f64(6.0).unwrap(),
        )
    } else {
        let theta = theta2.sqrt();
        let (sin_t, cos_t) = theta.clone().sin_cos();
        let sin_over_t = sin_t / theta;
        (cos_t, sin_over_t)
    };

    let x1 = (x0 * cos_t) + (v * sin_over_t);

    // Re-normalize to stay on the sphere even with numeric drift.
    x1.normalize()
}

/// The S² logarithm at `x`: the tangent step (in [`tangent_basis`]) that
/// [`plus`] maps to `y`.
pub(crate) fn minus<T: na::RealField>(y: &na::Vector3<T>, x: &na::Vector3<T>) -> na::Vector2<T> {
    const EPS: f64 = 1e-6;
    let x0 = x.normalize();
    let y0 = y.normalize();

    let (b1, b2) = tangent_basis(&x0);

    // Log map on S² at x0.
    //
    // sinθ = ||x×y||, cosθ = x·y, θ = atan2(sinθ, cosθ)
    // u = y - (x·y)x is the tangent direction with ||u|| = sinθ
    // w = (θ / sinθ) u   (if sinθ != 0)
    let dot = x0.dot(&y0);
    let cross = x0.cross(&y0);
    let sin_t = cross.norm();

    let theta = sin_t.clone().atan2(dot.clone());

    let u = y0 - x0.clone() * dot.clone();

    let w = if sin_t < T::from_f64(EPS).unwrap() {
        // Near-parallel or near-antipodal.
        // Parallel (dot>0): log is ~0.
        // Antipodal (dot<0): direction is not unique; pick deterministic b1 * π.
        if dot > T::zero() {
            na::Vector3::zeros()
        } else {
            b1.clone() * T::from_f64(std::f64::consts::PI).unwrap()
        }
    } else {
        u * (theta / sin_t)
    };

    let d0 = b1.dot(&w);
    let d1 = b2.dot(&w);

    na::Vector2::new(d0, d1)
}

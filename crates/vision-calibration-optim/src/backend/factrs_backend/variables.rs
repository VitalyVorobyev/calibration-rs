//! factrs variables for the IR parameter blocks.
//!
//! - camera blocks (intrinsics ⊕ distortion ⊕ sensor) fuse into one
//!   `VectorVar<N>`;
//! - an SE3 block is a factrs [`SE3`];
//! - a laser plane (unit normal ⊕ distance) is a [`PlaneVar`];
//! - a robot-pose correction is a `VectorVar<6>`.

use std::fmt;

use factrs::dtype;
use factrs::linalg::{Const, Numeric, SupersetOf, Vector2, Vector3, VectorViewX, VectorX};
use factrs::variables::{SE3, SO3, Variable};
use nalgebra::DVector;

use crate::backend::s2;

/// Dimensions a fused vector variable may take: the camera models' sizes
/// (`4 + distortion + sensor`) and the 6-D robot-pose correction.
pub(super) const VECTOR_DIMS: [usize; 10] = [4, 5, 6, 7, 9, 11, 12, 13, 14, 15];

/// Run `$body` with `$N` bound to the `usize` constant equal to `$n`, one of
/// [`VECTOR_DIMS`]; `$fallback` handles any other value.
macro_rules! with_vector_dim {
    ($n:expr, $N:ident => $body:expr, _ => $fallback:expr) => {
        match $n {
            4 => {
                const $N: usize = 4;
                $body
            }
            5 => {
                const $N: usize = 5;
                $body
            }
            6 => {
                const $N: usize = 6;
                $body
            }
            7 => {
                const $N: usize = 7;
                $body
            }
            9 => {
                const $N: usize = 9;
                $body
            }
            11 => {
                const $N: usize = 11;
                $body
            }
            12 => {
                const $N: usize = 12;
                $body
            }
            13 => {
                const $N: usize = 13;
                $body
            }
            14 => {
                const $N: usize = 14;
                $body
            }
            15 => {
                const $N: usize = 15;
                $body
            }
            _ => $fallback,
        }
    };
}
pub(super) use with_vector_dim;

/// A laser plane `n·p + d = 0`: a unit normal on S² and a signed distance.
///
/// factrs reaches a variable's manifold only through `x.compose(exp(δ))`:
/// its right `oplus` and its forward-mode dual seeding. So `exp(δ)` builds a
/// carrier for the tangent step `δ = [δn₀, δn₁, δd]`, and `compose` applies
/// the shared S² retraction ([`s2::plus`], the tiny-solver backend's too) to
/// the normal and adds to the distance. `identity`, `inverse` and `log` act
/// on carriers; nothing in this backend applies them to a plane.
#[derive(Clone, Debug)]
pub(crate) struct PlaneVar<T: Numeric = dtype> {
    pub(super) normal: Vector3<T>,
    pub(super) distance: T,
}

impl PlaneVar {
    /// A plane from its IR blocks.
    pub(super) fn from_ir(normal: &DVector<f64>, distance: f64) -> Self {
        Self {
            normal: Vector3::new(normal[0], normal[1], normal[2]),
            distance,
        }
    }
}

impl<T: Numeric> fmt::Display for PlaneVar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PlaneVar(n: [{}, {}, {}], d: {})",
            self.normal[0], self.normal[1], self.normal[2], self.distance
        )
    }
}

impl<T: Numeric> Variable for PlaneVar<T> {
    type T = T;
    type Dim = Const<3>;
    type Alias<TT: Numeric> = PlaneVar<TT>;

    fn identity() -> Self {
        Self {
            normal: Vector3::zeros(),
            distance: T::from(0.0),
        }
    }

    fn inverse(&self) -> Self {
        Self {
            normal: -self.normal,
            distance: -self.distance,
        }
    }

    fn compose(&self, step: &Self) -> Self {
        let dn = Vector2::new(step.normal[0], step.normal[1]);
        Self {
            normal: s2::plus(&self.normal, &dn),
            distance: self.distance + step.distance,
        }
    }

    fn exp(delta: VectorViewX<T>) -> Self {
        Self {
            normal: Vector3::new(delta[0], delta[1], T::from(0.0)),
            distance: delta[2],
        }
    }

    fn log(&self) -> VectorX<T> {
        VectorX::from_column_slice(&[self.normal[0], self.normal[1], self.distance])
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        PlaneVar {
            normal: self.normal.cast(),
            distance: TT::from_subset(&self.distance),
        }
    }
}

/// An IR SE3 block `[qx, qy, qz, qw, tx, ty, tz]` as a factrs [`SE3`].
pub(super) fn se3_from_ir(v: &DVector<f64>) -> SE3 {
    SE3::from_rot_trans(
        SO3::from_xyzw(v[0], v[1], v[2], v[3]),
        Vector3::new(v[4], v[5], v[6]),
    )
}

/// A factrs [`SE3`] as an IR SE3 block `[qx, qy, qz, qw, tx, ty, tz]`.
pub(super) fn se3_to_ir<T: Numeric>(p: &SE3<T>) -> DVector<T> {
    let (r, t) = (p.rot(), p.xyz());
    DVector::from_column_slice(&[r.x(), r.y(), r.z(), r.w(), t[0], t[1], t[2]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::tiny_solver_manifolds::UnitVector3Manifold;
    use tiny_solver::manifold::Manifold;

    /// The plane variable retracts exactly as the tiny-solver backend's S²
    /// manifold, plus a Euclidean distance.
    #[test]
    fn plane_retraction_matches_the_tiny_solver_manifold() {
        let normals = [
            Vector3::new(0.09, 0.17, 1.0).normalize(),
            Vector3::new(0.95, -0.2, 0.1).normalize(),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        let steps = [[0.0, 0.0, 0.0], [0.01, -0.02, 0.003], [0.4, 0.3, -0.2]];
        for n in normals {
            for d in steps {
                let plane = PlaneVar::from_ir(&DVector::from_column_slice(n.as_slice()), -0.33);
                let delta = VectorX::from_column_slice(&d);
                let moved = plane.oplus(delta.as_view());
                let expected = UnitVector3Manifold.plus_f64(
                    DVector::from_column_slice(n.as_slice()).as_view(),
                    DVector::from_column_slice(&d[..2]).as_view(),
                );
                assert_eq!(moved.normal.as_slice(), expected.as_slice());
                assert_eq!(moved.distance, -0.33 + d[2]);
            }
        }
    }

    #[test]
    fn se3_round_trips_through_the_ir_layout() {
        let v = DVector::from_row_slice(&[0.1, -0.2, 0.05, 0.9722, 0.4, -0.5, 0.6]);
        assert_eq!(se3_to_ir(&se3_from_ir(&v)), v);
    }
}

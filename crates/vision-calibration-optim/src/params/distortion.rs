//! Distortion parameter packing for optimization.

use crate::Error;
use crate::ir::DistortionKind;
use nalgebra::{DVector, DVectorView};
use vision_calibration_core::{
    BrownConrady5, DistortionParams, RationalPolynomial, Real, ThinPrism,
};

/// Dimension of the Brown-Conrady distortion vector [k1, k2, k3, p1, p2].
pub const DISTORTION_DIM: usize = 5;

/// Pack distortion into a dense parameter vector `[k1, k2, k3, p1, p2]`.
pub fn pack_distortion(dist: &BrownConrady5<Real>) -> DVector<f64> {
    nalgebra::dvector![dist.k1, dist.k2, dist.k3, dist.p1, dist.p2]
}

/// Unpack distortion from a dense parameter vector `[k1, k2, k3, p1, p2]`.
///
/// The `iters` field is set to the default of 8.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if the vector length does not equal
/// [`DISTORTION_DIM`].
pub fn unpack_distortion(v: DVectorView<'_, f64>) -> Result<BrownConrady5<Real>, Error> {
    if v.len() != DISTORTION_DIM {
        return Err(Error::invalid_input(format!(
            "expected distortion vector of length {}, got {}",
            DISTORTION_DIM,
            v.len()
        )));
    }
    Ok(BrownConrady5 {
        k1: v[0],
        k2: v[1],
        k3: v[2],
        p1: v[3],
        p2: v[4],
        iters: 8,
    })
}

/// Pack a [`DistortionParams`] into the IR-ordered distortion vector for the
/// active variant.
///
/// - `None`           → empty `DVector` (length 0)
/// - `BrownConrady5`  → `[k1, k2, k3, p1, p2]`
/// - `Rational`       → `[k1, k2, k3, k4, k5, k6, p1, p2]`
/// - `ThinPrism`      → `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`
/// - `Division`       → `[lambda]`
pub fn pack_distortion_params(d: &DistortionParams) -> DVector<f64> {
    match d {
        DistortionParams::None => DVector::zeros(0),
        DistortionParams::BrownConrady5 { params: p } => {
            nalgebra::dvector![p.k1, p.k2, p.k3, p.p1, p.p2]
        }
        DistortionParams::Rational { params: p } => {
            nalgebra::dvector![p.k1, p.k2, p.k3, p.k4, p.k5, p.k6, p.p1, p.p2]
        }
        DistortionParams::ThinPrism { params: p } => {
            nalgebra::dvector![p.k1, p.k2, p.k3, p.p1, p.p2, p.s1, p.s2, p.s3, p.s4]
        }
        DistortionParams::Division { lambda } => {
            nalgebra::dvector![*lambda]
        }
    }
}

/// Unpack a [`DistortionParams`] from an IR-ordered distortion vector.
///
/// `kind` selects the expected length and layout; returns
/// [`Error::InvalidInput`] if `v.len() != kind.dim()`.
///
/// The `iters` field (where present on the underlying struct) is set to 8.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `v.len() != kind.dim()`.
pub fn unpack_distortion_params(
    kind: DistortionKind,
    v: DVectorView<'_, f64>,
) -> Result<DistortionParams, Error> {
    if v.len() != kind.dim() {
        return Err(Error::invalid_input(format!(
            "expected distortion vector of length {} for {:?}, got {}",
            kind.dim(),
            kind,
            v.len()
        )));
    }
    match kind {
        DistortionKind::None => Ok(DistortionParams::None),
        DistortionKind::BrownConrady5 => Ok(DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: v[0],
                k2: v[1],
                k3: v[2],
                p1: v[3],
                p2: v[4],
                iters: 8,
            },
        }),
        DistortionKind::Rational8 => Ok(DistortionParams::Rational {
            params: RationalPolynomial {
                k1: v[0],
                k2: v[1],
                k3: v[2],
                k4: v[3],
                k5: v[4],
                k6: v[5],
                p1: v[6],
                p2: v[7],
                iters: 8,
            },
        }),
        DistortionKind::ThinPrism9 => Ok(DistortionParams::ThinPrism {
            params: ThinPrism {
                k1: v[0],
                k2: v[1],
                k3: v[2],
                p1: v[3],
                p2: v[4],
                s1: v[5],
                s2: v[6],
                s3: v[7],
                s4: v[8],
                iters: 8,
            },
        }),
        DistortionKind::Division1 => Ok(DistortionParams::Division { lambda: v[0] }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let dist = BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.01,
            p1: 0.001,
            p2: -0.001,
            iters: 9,
        };
        let v = pack_distortion(&dist);
        let restored = unpack_distortion(v.as_view()).unwrap();
        assert_eq!(restored.k1, dist.k1);
        assert_eq!(restored.k2, dist.k2);
        assert_eq!(restored.k3, dist.k3);
        assert_eq!(restored.p1, dist.p1);
        assert_eq!(restored.p2, dist.p2);
        assert_eq!(restored.iters, 8);
    }

    // ── Model-aware pack/unpack roundtrips ──────────────────────────────────

    #[test]
    fn pack_unpack_params_none() {
        let d = DistortionParams::None;
        let v = pack_distortion_params(&d);
        assert_eq!(v.len(), 0);
        let restored = unpack_distortion_params(DistortionKind::None, v.as_view()).unwrap();
        assert!(matches!(restored, DistortionParams::None));
    }

    #[test]
    fn pack_unpack_params_brown_conrady5() {
        let d = DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.1,
                k2: 0.02,
                k3: 0.003,
                p1: 0.001,
                p2: -0.002,
                iters: 9,
            },
        };
        let v = pack_distortion_params(&d);
        assert_eq!(v.len(), 5);
        let restored =
            unpack_distortion_params(DistortionKind::BrownConrady5, v.as_view()).unwrap();
        let DistortionParams::BrownConrady5 { params: p } = restored else {
            panic!("wrong variant");
        };
        assert!((p.k1 - (-0.1)).abs() < 1e-15);
        assert!((p.k2 - 0.02).abs() < 1e-15);
        assert!((p.k3 - 0.003).abs() < 1e-15);
        assert!((p.p1 - 0.001).abs() < 1e-15);
        assert!((p.p2 - (-0.002)).abs() < 1e-15);
        assert_eq!(p.iters, 8);
    }

    #[test]
    fn pack_unpack_params_rational8() {
        use vision_calibration_core::RationalPolynomial;
        let d = DistortionParams::Rational {
            params: RationalPolynomial {
                k1: 0.1,
                k2: -0.05,
                k3: 0.001,
                k4: 0.01,
                k5: -0.005,
                k6: 0.002,
                p1: 0.003,
                p2: -0.001,
                iters: 10,
            },
        };
        let v = pack_distortion_params(&d);
        assert_eq!(v.len(), 8);
        let restored = unpack_distortion_params(DistortionKind::Rational8, v.as_view()).unwrap();
        let DistortionParams::Rational { params: p } = restored else {
            panic!("wrong variant");
        };
        assert!((p.k1 - 0.1).abs() < 1e-15);
        assert!((p.k4 - 0.01).abs() < 1e-15);
        assert!((p.p2 - (-0.001)).abs() < 1e-15);
        assert_eq!(p.iters, 8);
    }

    #[test]
    fn pack_unpack_params_thinprism9() {
        use vision_calibration_core::ThinPrism;
        let d = DistortionParams::ThinPrism {
            params: ThinPrism {
                k1: -0.2,
                k2: 0.05,
                k3: 0.0,
                p1: 0.001,
                p2: -0.001,
                s1: 0.002,
                s2: -0.001,
                s3: 0.003,
                s4: -0.002,
                iters: 10,
            },
        };
        let v = pack_distortion_params(&d);
        assert_eq!(v.len(), 9);
        let restored = unpack_distortion_params(DistortionKind::ThinPrism9, v.as_view()).unwrap();
        let DistortionParams::ThinPrism { params: p } = restored else {
            panic!("wrong variant");
        };
        assert!((p.s1 - 0.002).abs() < 1e-15);
        assert!((p.s4 - (-0.002)).abs() < 1e-15);
        assert_eq!(p.iters, 8);
    }

    #[test]
    fn pack_unpack_params_division1() {
        let d = DistortionParams::Division { lambda: -0.15 };
        let v = pack_distortion_params(&d);
        assert_eq!(v.len(), 1);
        let restored = unpack_distortion_params(DistortionKind::Division1, v.as_view()).unwrap();
        let DistortionParams::Division { lambda } = restored else {
            panic!("wrong variant");
        };
        assert!((lambda - (-0.15)).abs() < 1e-15);
    }

    #[test]
    fn unpack_params_rejects_wrong_length() {
        let v = nalgebra::dvector![1.0, 2.0]; // len 2
        let err = unpack_distortion_params(DistortionKind::BrownConrady5, v.as_view()).unwrap_err();
        assert!(
            err.to_string().contains("5"),
            "error should mention expected length 5: {err}"
        );
    }
}

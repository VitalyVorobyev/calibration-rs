//! Distortion parameter packing for optimization.

use crate::Error;
use crate::ir::DistortionKind;
use nalgebra::{DVector, DVectorView};
use vision_calibration_core::{
    BrownConrady5, DistortionFixMask, DistortionParams, RationalPolynomial, Real, ThinPrism,
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

/// Translate a Brown-Conrady-shaped [`DistortionFixMask`] onto the packed
/// coefficient layout of `kind`, returning the fixed indices into that layout.
///
/// This is the single name-based mechanism used for **both** user-supplied fix
/// masks and the pipeline's staging masks (A0/A1), so the extended models
/// honour the same "which coefficients are free" invariants as Brown-Conrady.
/// The packed orderings are the ones produced by [`pack_distortion_params`];
/// index accordingly.
///
/// The `DistortionFixMask` has five named bits `{k1, k2, k3, p1, p2}`. They map
/// by name onto every model's shared coefficients; each model's extra
/// coefficients follow the mask bit of the *family* they belong to:
///
/// - **BrownConrady5** `[k1, k2, k3, p1, p2]`: exactly [`DistortionFixMask::to_indices`]
///   (byte-identical to the pre-M-WIRE path).
/// - **Rational8** `[k1, k2, k3, k4, k5, k6, p1, p2]`: `k1,k2,k3,p1,p2` by name;
///   the higher-order radial block `k4,k5,k6` follows the `k3` bit (OpenCV-style
///   — it is the same radial family, promoted/demoted together).
/// - **ThinPrism9** `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`: `k1,k2,k3,p1,p2` by
///   name; the thin-prism block `s1..s4` follows the tangential pair — fixed iff
///   **both** `p1` and `p2` are fixed (prism is a tangential-family refinement).
/// - **Division1** `[lambda]`: the single barrel term follows the leading radial
///   `k1` bit, consistent with [`with_leading_radial`].
/// - **None**: no coefficients, so no indices.
pub fn fix_mask_indices(mask: &DistortionFixMask, kind: DistortionKind) -> Vec<usize> {
    // Fixed-flag per packed coefficient, in `pack_distortion_params` order.
    let flags: Vec<bool> = match kind {
        DistortionKind::None => Vec::new(),
        DistortionKind::BrownConrady5 => vec![mask.k1, mask.k2, mask.k3, mask.p1, mask.p2],
        DistortionKind::Rational8 => vec![
            mask.k1, mask.k2, mask.k3, mask.k3, mask.k3, mask.k3, mask.p1, mask.p2,
        ],
        DistortionKind::ThinPrism9 => {
            let prism = mask.p1 && mask.p2;
            vec![
                mask.k1, mask.k2, mask.k3, mask.p1, mask.p2, prism, prism, prism, prism,
            ]
        }
        DistortionKind::Division1 => vec![mask.k1],
    };
    flags
        .into_iter()
        .enumerate()
        .filter_map(|(i, fixed)| fixed.then_some(i))
        .collect()
}

/// Map a [`DistortionParams`] variant to its [`DistortionKind`] discriminant.
///
/// This is the single source of truth for the params → kind mapping shared by
/// the planar and Scheimpflug problem builders.
pub fn distortion_kind(d: &DistortionParams) -> DistortionKind {
    match d {
        DistortionParams::None => DistortionKind::None,
        DistortionParams::BrownConrady5 { .. } => DistortionKind::BrownConrady5,
        DistortionParams::Rational { .. } => DistortionKind::Rational8,
        DistortionParams::ThinPrism { .. } => DistortionKind::ThinPrism9,
        DistortionParams::Division { .. } => DistortionKind::Division1,
    }
}

/// Return a copy of `d` with its **leading radial coefficient** set to `value`,
/// preserving the model variant and all other coefficients.
///
/// The leading radial term is `k1` for [`DistortionParams::BrownConrady5`],
/// [`DistortionParams::Rational`], and [`DistortionParams::ThinPrism`], and
/// `lambda` for [`DistortionParams::Division`] (also the primary barrel term).
/// [`DistortionParams::None`] carries no coefficient, so it is returned
/// unchanged (a leading-radial multi-start collapses to a single start).
///
/// Used by the Scheimpflug warm-start to sweep the leading barrel coefficient
/// across a coarse grid without hard-coding the Brown-Conrady layout.
pub fn with_leading_radial(d: &DistortionParams, value: f64) -> DistortionParams {
    match d {
        DistortionParams::None => DistortionParams::None,
        DistortionParams::BrownConrady5 { params } => DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: value,
                ..*params
            },
        },
        DistortionParams::Rational { params } => DistortionParams::Rational {
            params: RationalPolynomial {
                k1: value,
                ..*params
            },
        },
        DistortionParams::ThinPrism { params } => DistortionParams::ThinPrism {
            params: ThinPrism {
                k1: value,
                ..*params
            },
        },
        DistortionParams::Division { .. } => DistortionParams::Division { lambda: value },
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

    // ── Name-based fix-mask translation ─────────────────────────────────────

    #[test]
    fn fix_mask_bc5_is_identity_with_to_indices() {
        // Every possible BC5 mask must translate to exactly `to_indices()`.
        for bits in 0u8..32 {
            let mask = DistortionFixMask {
                k1: bits & 1 != 0,
                k2: bits & 2 != 0,
                k3: bits & 4 != 0,
                p1: bits & 8 != 0,
                p2: bits & 16 != 0,
            };
            assert_eq!(
                fix_mask_indices(&mask, DistortionKind::BrownConrady5),
                mask.to_indices(),
                "BC5 translation diverged from to_indices() for {mask:?}"
            );
        }
    }

    #[test]
    fn fix_mask_all_fixed_covers_every_index() {
        let mask = DistortionFixMask::all_fixed();
        for kind in [
            DistortionKind::None,
            DistortionKind::BrownConrady5,
            DistortionKind::Rational8,
            DistortionKind::ThinPrism9,
            DistortionKind::Division1,
        ] {
            let expected: Vec<usize> = (0..kind.dim()).collect();
            assert_eq!(
                fix_mask_indices(&mask, kind),
                expected,
                "all-fixed mask must fix every packed index for {kind:?}"
            );
        }
    }

    #[test]
    fn fix_mask_a1_shape_frees_leading_radial_per_kind() {
        // The A1 staging mask: {k1, k2 free; k3, p1, p2 fixed}.
        let a1 = DistortionFixMask {
            k1: false,
            k2: false,
            k3: true,
            p1: true,
            p2: true,
        };
        // BC5 `[k1,k2,k3,p1,p2]`: fix k3,p1,p2.
        assert_eq!(
            fix_mask_indices(&a1, DistortionKind::BrownConrady5),
            vec![2, 3, 4]
        );
        // Rational8 `[k1,k2,k3,k4,k5,k6,p1,p2]`: k4,k5,k6 follow k3 → fixed.
        assert_eq!(
            fix_mask_indices(&a1, DistortionKind::Rational8),
            vec![2, 3, 4, 5, 6, 7]
        );
        // ThinPrism9 `[k1,k2,k3,p1,p2,s1,s2,s3,s4]`: s1..s4 follow (p1&&p2) → fixed.
        assert_eq!(
            fix_mask_indices(&a1, DistortionKind::ThinPrism9),
            vec![2, 3, 4, 5, 6, 7, 8]
        );
        // Division1 `[lambda]`: lambda follows k1 (free) → nothing fixed.
        assert_eq!(
            fix_mask_indices(&a1, DistortionKind::Division1),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn fix_mask_thinprism_prism_follows_tangential_pair() {
        // Only one of p1/p2 fixed ⇒ prism block stays free.
        let one_tangential = DistortionFixMask {
            k1: false,
            k2: false,
            k3: false,
            p1: true,
            p2: false,
        };
        assert_eq!(
            fix_mask_indices(&one_tangential, DistortionKind::ThinPrism9),
            vec![3], // p1 only; s1..s4 free because p2 is free
        );
    }
}

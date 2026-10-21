//! Intrinsics parameter packing for optimization.

use crate::Error;
use nalgebra::{DVector, DVectorView};
use vision_calibration_core::{FxFyCxCy, Real};

/// Dimension of the intrinsics parameter vector [fx, fy, cx, cy].
pub const INTRINSICS_DIM: usize = 4;

/// Pack intrinsics into a dense parameter vector `[fx, fy, cx, cy]`.
pub fn pack_intrinsics(k: &FxFyCxCy<Real>) -> DVector<f64> {
    nalgebra::dvector![k.fx, k.fy, k.cx, k.cy]
}

/// Unpack intrinsics from a dense parameter vector `[fx, fy, cx, cy]`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if the vector length does not equal
/// [`INTRINSICS_DIM`].
pub fn unpack_intrinsics(v: DVectorView<'_, f64>) -> Result<FxFyCxCy<Real>, Error> {
    if v.len() != INTRINSICS_DIM {
        return Err(Error::invalid_input(format!(
            "expected intrinsics vector of length {}, got {}",
            INTRINSICS_DIM,
            v.len()
        )));
    }
    Ok(FxFyCxCy {
        fx: v[0],
        fy: v[1],
        cx: v[2],
        cy: v[3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        };
        let v = pack_intrinsics(&k);
        let restored = unpack_intrinsics(v.as_view()).unwrap();
        assert_eq!(restored.fx, k.fx);
        assert_eq!(restored.fy, k.fy);
        assert_eq!(restored.cx, k.cx);
        assert_eq!(restored.cy, k.cy);
    }
}

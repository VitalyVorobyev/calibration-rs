//! Intrinsics parameter packing for optimization.

use anyhow::{ensure, Result};
use calib_core::{FxFyCxCySkew, Real};
use nalgebra::{DVector, DVectorView};

/// Dimension of the intrinsics parameter vector [fx, fy, cx, cy].
pub const INTRINSICS_DIM: usize = 4;

/// Pack intrinsics into a dense parameter vector `[fx, fy, cx, cy]`.
///
/// Skew is not optimized in the current problems and must be ~0.
pub fn pack_intrinsics(k: &FxFyCxCySkew<Real>) -> Result<DVector<f64>> {
    ensure!(
        k.skew.abs() <= 1e-12,
        "intrinsics skew must be ~0 for 4-parameter packing"
    );
    Ok(nalgebra::dvector![k.fx, k.fy, k.cx, k.cy])
}

/// Unpack intrinsics from a dense parameter vector `[fx, fy, cx, cy]`.
pub fn unpack_intrinsics(v: DVectorView<'_, f64>) -> Result<FxFyCxCySkew<Real>> {
    ensure!(
        v.len() == INTRINSICS_DIM,
        "expected intrinsics vector of length {}, got {}",
        INTRINSICS_DIM,
        v.len()
    );
    Ok(FxFyCxCySkew {
        fx: v[0],
        fy: v[1],
        cx: v[2],
        cy: v[3],
        skew: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let v = pack_intrinsics(&k).unwrap();
        let restored = unpack_intrinsics(v.as_view()).unwrap();
        assert_eq!(restored.fx, k.fx);
        assert_eq!(restored.fy, k.fy);
        assert_eq!(restored.cx, k.cx);
        assert_eq!(restored.cy, k.cy);
        assert_eq!(restored.skew, 0.0);
    }
}

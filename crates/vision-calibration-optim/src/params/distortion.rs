//! Distortion parameter packing for optimization.

use anyhow::{Result, ensure};
use nalgebra::{DVector, DVectorView};
use vision_calibration_core::{BrownConrady5, Real};

/// Dimension of the Brown-Conrady distortion vector [k1, k2, k3, p1, p2].
pub const DISTORTION_DIM: usize = 5;

/// Pack distortion into a dense parameter vector `[k1, k2, k3, p1, p2]`.
pub fn pack_distortion(dist: &BrownConrady5<Real>) -> DVector<f64> {
    nalgebra::dvector![dist.k1, dist.k2, dist.k3, dist.p1, dist.p2]
}

/// Unpack distortion from a dense parameter vector `[k1, k2, k3, p1, p2]`.
///
/// The `iters` field is set to the default of 8.
pub fn unpack_distortion(v: DVectorView<'_, f64>) -> Result<BrownConrady5<Real>> {
    ensure!(
        v.len() == DISTORTION_DIM,
        "expected distortion vector of length {}, got {}",
        DISTORTION_DIM,
        v.len()
    );
    Ok(BrownConrady5 {
        k1: v[0],
        k2: v[1],
        k3: v[2],
        p1: v[3],
        p2: v[4],
        iters: 8,
    })
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
}

//! Distortion parameter blocks.

use anyhow::{ensure, Result};
use calib_core::models::BrownConrady5;
use calib_core::Real;
use nalgebra::{DVector, DVectorView};

/// Brown-Conrady 5-parameter distortion block.
///
/// Represents radial distortion (k1, k2, k3) and tangential distortion (p1, p2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrownConrady5Params {
    /// First radial distortion coefficient.
    pub k1: f64,
    /// Second radial distortion coefficient.
    pub k2: f64,
    /// Third radial distortion coefficient.
    pub k3: f64,
    /// First tangential distortion coefficient.
    pub p1: f64,
    /// Second tangential distortion coefficient.
    pub p2: f64,
}

impl BrownConrady5Params {
    /// Dimension of the parameter vector.
    pub const DIM: usize = 5;

    /// Zero distortion (identity mapping).
    pub fn zeros() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Convert to a dense parameter vector `[k1, k2, k3, p1, p2]`.
    pub fn to_dvec(&self) -> DVector<f64> {
        nalgebra::dvector![self.k1, self.k2, self.k3, self.p1, self.p2]
    }

    /// Build from a dense parameter vector `[k1, k2, k3, p1, p2]`.
    pub fn from_dvec(v: DVectorView<'_, f64>) -> Result<Self> {
        ensure!(
            v.len() == Self::DIM,
            "expected distortion vector of length {}, got {}",
            Self::DIM,
            v.len()
        );
        Ok(Self {
            k1: v[0],
            k2: v[1],
            k3: v[2],
            p1: v[3],
            p2: v[4],
        })
    }

    /// Convert into the calib-core distortion type with default iteration count.
    pub fn to_core(self) -> BrownConrady5<Real> {
        BrownConrady5 {
            k1: self.k1,
            k2: self.k2,
            k3: self.k3,
            p1: self.p1,
            p2: self.p2,
            iters: 8,
        }
    }

    /// Build from calib-core distortion type.
    pub fn from_core(dist: &BrownConrady5<Real>) -> Self {
        Self {
            k1: dist.k1,
            k2: dist.k2,
            k3: dist.k3,
            p1: dist.p1,
            p2: dist.p2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_conversion() {
        let orig = BrownConrady5Params {
            k1: -0.2,
            k2: 0.05,
            k3: 0.01,
            p1: 0.001,
            p2: -0.001,
        };
        let v = orig.to_dvec();
        let restored = BrownConrady5Params::from_dvec(v.as_view()).unwrap();
        assert_eq!(orig.k1, restored.k1);
        assert_eq!(orig.k2, restored.k2);
        assert_eq!(orig.k3, restored.k3);
        assert_eq!(orig.p1, restored.p1);
        assert_eq!(orig.p2, restored.p2);
    }

    #[test]
    fn zeros_creates_identity() {
        let zero = BrownConrady5Params::zeros();
        assert_eq!(zero.k1, 0.0);
        assert_eq!(zero.k2, 0.0);
        assert_eq!(zero.k3, 0.0);
        assert_eq!(zero.p1, 0.0);
        assert_eq!(zero.p2, 0.0);
    }

    #[test]
    fn core_conversion() {
        let params = BrownConrady5Params {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
        };
        let core = params.to_core();
        let restored = BrownConrady5Params::from_core(&core);
        assert_eq!(params, restored);
    }
}

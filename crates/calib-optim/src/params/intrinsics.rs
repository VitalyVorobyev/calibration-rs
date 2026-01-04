//! Intrinsics parameter blocks.

use anyhow::{ensure, Result};
use calib_core::{FxFyCxCySkew, Real};
use nalgebra::{DVector, DVectorView};

/// Minimal 4-parameter intrinsics block.
///
/// This is the primary intrinsics representation for planar optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics4 {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
}

impl Intrinsics4 {
    /// Dimension of the parameter vector.
    pub const DIM: usize = 4;

    /// Convert to a dense parameter vector `[fx, fy, cx, cy]`.
    pub fn to_dvec(&self) -> DVector<f64> {
        nalgebra::dvector![self.fx, self.fy, self.cx, self.cy]
    }

    /// Build from a dense parameter vector `[fx, fy, cx, cy]`.
    pub fn from_dvec(v: DVectorView<'_, f64>) -> Result<Self> {
        ensure!(
            v.len() == Self::DIM,
            "expected intrinsics vector of length {}, got {}",
            Self::DIM,
            v.len()
        );
        Ok(Self {
            fx: v[0],
            fy: v[1],
            cx: v[2],
            cy: v[3],
        })
    }

    /// Convert into the calib-core intrinsics type with zero skew.
    pub fn to_core(self) -> FxFyCxCySkew<Real> {
        FxFyCxCySkew {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            skew: 0.0,
        }
    }
}

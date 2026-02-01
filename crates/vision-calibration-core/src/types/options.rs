//! Configuration options for calibration.
//!
//! This module provides structured types for configuring parameter fixing
//! during optimization, replacing scattered boolean fields with semantic masks.

use serde::{Deserialize, Serialize};

/// Mask for fixing intrinsics parameters during optimization.
///
/// Each field corresponds to a component of the camera matrix K:
/// ```text
/// K = | fx   skew  cx |
///     | 0    fy    cy |
///     | 0    0     1  |
/// ```
///
/// Set a field to `true` to keep that parameter fixed during optimization.
///
/// # Example
///
/// ```
/// use vision_calibration_core::IntrinsicsFixMask;
///
/// // Fix only the principal point
/// let mask = IntrinsicsFixMask {
///     cx: true,
///     cy: true,
///     ..Default::default()
/// };
///
/// assert!(!mask.fx); // fx will be optimized
/// assert!(mask.cx);  // cx is fixed
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IntrinsicsFixMask {
    /// Fix focal length x component.
    pub fx: bool,
    /// Fix focal length y component.
    pub fy: bool,
    /// Fix principal point x component.
    pub cx: bool,
    /// Fix principal point y component.
    pub cy: bool,
}

impl IntrinsicsFixMask {
    /// Create a mask that fixes all intrinsics parameters.
    pub fn all_fixed() -> Self {
        Self {
            fx: true,
            fy: true,
            cx: true,
            cy: true,
        }
    }

    /// Create a mask that allows all intrinsics parameters to be optimized.
    pub fn all_free() -> Self {
        Self::default()
    }

    /// Returns true if any parameter is fixed.
    pub fn any_fixed(&self) -> bool {
        self.fx || self.fy || self.cx || self.cy
    }

    /// Returns true if all parameters are fixed.
    pub fn all_are_fixed(&self) -> bool {
        self.fx && self.fy && self.cx && self.cy
    }

    /// Convert to a vector of fixed indices (for IR FixedMask).
    ///
    /// Index mapping: fx=0, fy=1, cx=2, cy=3
    pub fn to_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        if self.fx {
            indices.push(0);
        }
        if self.fy {
            indices.push(1);
        }
        if self.cx {
            indices.push(2);
        }
        if self.cy {
            indices.push(3);
        }
        indices
    }
}

/// Mask for fixing distortion parameters during optimization.
///
/// Supports Brown-Conrady distortion model with:
/// - Radial coefficients: k1, k2, k3
/// - Tangential coefficients: p1, p2
///
/// Set a field to `true` to keep that parameter fixed during optimization.
///
/// # Default
///
/// By default, `k3` is fixed because it often causes overfitting with
/// typical calibration data. Only enable k3 optimization for wide-angle
/// lenses or with high-quality calibration data.
///
/// # Example
///
/// ```
/// use vision_calibration_core::DistortionFixMask;
///
/// // Default: k3 fixed, others free
/// let mask = DistortionFixMask::default();
/// assert!(!mask.k1);
/// assert!(mask.k3); // k3 fixed by default
///
/// // Fix tangential distortion
/// let mask = DistortionFixMask {
///     p1: true,
///     p2: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistortionFixMask {
    /// Fix first radial distortion coefficient.
    pub k1: bool,
    /// Fix second radial distortion coefficient.
    pub k2: bool,
    /// Fix third radial distortion coefficient (fixed by default).
    pub k3: bool,
    /// Fix first tangential distortion coefficient.
    pub p1: bool,
    /// Fix second tangential distortion coefficient.
    pub p2: bool,
}

impl Default for DistortionFixMask {
    /// Default mask with k3 fixed to prevent overfitting.
    fn default() -> Self {
        Self {
            k1: false,
            k2: false,
            k3: true, // k3 often overfits, fix by default
            p1: false,
            p2: false,
        }
    }
}

impl DistortionFixMask {
    /// Create a mask that fixes all distortion parameters.
    pub fn all_fixed() -> Self {
        Self {
            k1: true,
            k2: true,
            k3: true,
            p1: true,
            p2: true,
        }
    }

    /// Create a mask that allows all distortion parameters to be optimized.
    pub fn all_free() -> Self {
        Self {
            k1: false,
            k2: false,
            k3: false,
            p1: false,
            p2: false,
        }
    }

    /// Create a mask that only allows radial distortion (k1, k2) to be optimized.
    ///
    /// Fixes k3, p1, p2 - useful for standard lenses with minimal tangential distortion.
    pub fn radial_only() -> Self {
        Self {
            k1: false,
            k2: false,
            k3: true,
            p1: true,
            p2: true,
        }
    }

    /// Returns true if any parameter is fixed.
    pub fn any_fixed(&self) -> bool {
        self.k1 || self.k2 || self.k3 || self.p1 || self.p2
    }

    /// Returns true if all parameters are fixed.
    pub fn all_are_fixed(&self) -> bool {
        self.k1 && self.k2 && self.k3 && self.p1 && self.p2
    }

    /// Convert to a vector of fixed indices (for IR FixedMask).
    ///
    /// Index mapping: k1=0, k2=1, k3=2, p1=3, p2=4
    pub fn to_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        if self.k1 {
            indices.push(0);
        }
        if self.k2 {
            indices.push(1);
        }
        if self.k3 {
            indices.push(2);
        }
        if self.p1 {
            indices.push(3);
        }
        if self.p2 {
            indices.push(4);
        }
        indices
    }
}

/// Combined mask for camera calibration parameters.
///
/// Groups intrinsics and distortion masks for convenient configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CameraFixMask {
    /// Mask for intrinsics (fx, fy, cx, cy).
    pub intrinsics: IntrinsicsFixMask,
    /// Mask for distortion (k1, k2, k3, p1, p2).
    pub distortion: DistortionFixMask,
}

impl CameraFixMask {
    /// Create a mask that fixes all camera parameters.
    pub fn all_fixed() -> Self {
        Self {
            intrinsics: IntrinsicsFixMask::all_fixed(),
            distortion: DistortionFixMask::all_fixed(),
        }
    }

    /// Create a mask that allows all camera parameters to be optimized,
    /// except k3 which is fixed by default.
    pub fn all_free() -> Self {
        Self {
            intrinsics: IntrinsicsFixMask::all_free(),
            distortion: DistortionFixMask::default(), // k3 still fixed
        }
    }

    /// Create a mask that allows all camera parameters to be optimized,
    /// including k3.
    pub fn truly_all_free() -> Self {
        Self {
            intrinsics: IntrinsicsFixMask::all_free(),
            distortion: DistortionFixMask::all_free(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_mask_default() {
        let mask = IntrinsicsFixMask::default();
        assert!(!mask.fx);
        assert!(!mask.fy);
        assert!(!mask.cx);
        assert!(!mask.cy);
        assert!(!mask.any_fixed());
    }

    #[test]
    fn intrinsics_mask_all_fixed() {
        let mask = IntrinsicsFixMask::all_fixed();
        assert!(mask.all_are_fixed());
        assert_eq!(mask.to_indices(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn distortion_mask_default() {
        let mask = DistortionFixMask::default();
        assert!(!mask.k1);
        assert!(!mask.k2);
        assert!(mask.k3); // k3 fixed by default
        assert!(!mask.p1);
        assert!(!mask.p2);
    }

    #[test]
    fn distortion_mask_radial_only() {
        let mask = DistortionFixMask::radial_only();
        assert!(!mask.k1);
        assert!(!mask.k2);
        assert!(mask.k3);
        assert!(mask.p1);
        assert!(mask.p2);
    }

    #[test]
    fn distortion_mask_to_indices() {
        let mask = DistortionFixMask {
            k1: true,
            k2: false,
            k3: true,
            p1: false,
            p2: true,
        };
        assert_eq!(mask.to_indices(), vec![0, 2, 4]);
    }

    #[test]
    fn camera_mask_serde_roundtrip() {
        let mask = CameraFixMask {
            intrinsics: IntrinsicsFixMask {
                fx: true,
                fy: false,
                cx: true,
                cy: false,
            },
            distortion: DistortionFixMask::default(),
        };

        let json = serde_json::to_string(&mask).unwrap();
        let restored: CameraFixMask = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, mask);
    }
}

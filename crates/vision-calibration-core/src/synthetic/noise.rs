//! Deterministic noise helpers for synthetic datasets.
//!
//! The functions here avoid `thread_rng` and do not depend on the internal
//! algorithm of `rand` RNGs. This keeps synthetic datasets stable across
//! versions and platforms.

use crate::{Real, Vec2};

/// Deterministic uniform pixel noise in `[-max_abs_px, +max_abs_px]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformPixelNoise {
    /// Base seed controlling the pseudo-random sequence.
    pub seed: u64,
    /// Maximum absolute per-axis noise (pixels).
    pub max_abs_px: Real,
}

impl Default for UniformPixelNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            max_abs_px: 0.0,
        }
    }
}

impl UniformPixelNoise {
    /// Sample a deterministic 2D noise vector (pixels) for a given `(view_idx, point_idx)` key.
    #[inline]
    pub fn sample(&self, view_idx: usize, point_idx: usize) -> Vec2 {
        let max_abs = self.max_abs_px.abs();
        if max_abs == 0.0 {
            return Vec2::zeros();
        }

        let key = mix_key(self.seed, view_idx, point_idx);
        let u = u64_to_unit_f64(splitmix64(key));
        let v = u64_to_unit_f64(splitmix64(key ^ 0x94D0_49BB_1331_11EB));

        // Map [0, 1) -> [-max_abs, +max_abs].
        let du = (u - 0.5) * 2.0 * max_abs;
        let dv = (v - 0.5) * 2.0 * max_abs;
        Vec2::new(du, dv)
    }

    /// Apply deterministic noise to a pixel observation.
    #[inline]
    pub fn apply(&self, view_idx: usize, point_idx: usize, uv: Vec2) -> Vec2 {
        uv + self.sample(view_idx, point_idx)
    }
}

#[inline]
fn mix_key(seed: u64, view_idx: usize, point_idx: usize) -> u64 {
    // SplitMix64 stream selection via a stable integer mix.
    seed ^ (view_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (point_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn u64_to_unit_f64(x: u64) -> Real {
    // Convert the top 53 bits to a double in [0, 1).
    // This is deterministic and platform-independent.
    let mantissa = x >> 11;
    (mantissa as Real) * (1.0 / ((1u64 << 53) as Real))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_pixel_noise_is_deterministic() {
        let noise = UniformPixelNoise {
            seed: 123,
            max_abs_px: 0.5,
        };

        let a = noise.sample(0, 0);
        let b = noise.sample(0, 0);
        let c = noise.sample(0, 1);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.x.abs() <= 0.5);
        assert!(a.y.abs() <= 0.5);
    }
}

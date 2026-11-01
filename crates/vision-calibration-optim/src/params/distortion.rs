//! Distortion helpers for the staged initializers.

use vision_calibration_core::DistortionParams;

/// Return a copy of `d` with its **leading radial coefficient**
/// ([`DistortionKind::leading_radial`](vision_calibration_core::DistortionKind::leading_radial):
/// `k1`, or the division model's `lambda`) set to `value`, keeping the model
/// and every other coefficient. A distortion without coefficients is
/// returned unchanged (a leading-radial multi-start collapses to a single
/// start).
///
/// Used by the Scheimpflug warm-start to sweep the leading barrel
/// coefficient across a coarse grid without hard-coding a model's layout.
pub fn with_leading_radial(d: &DistortionParams, value: f64) -> DistortionParams {
    let kind = d.kind();
    let Some(index) = kind
        .leading_radial()
        .and_then(|name| kind.param_names().iter().position(|n| *n == name))
    else {
        return d.clone();
    };
    let mut values = d.values();
    values[index] = value;
    d.with_values(&values)
        .expect("values() has the model's dimension, so with_values accepts it")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{BrownConrady5, DistortionKind};

    #[test]
    fn sets_the_leading_radial_coefficient_of_every_model() {
        for kind in [
            DistortionKind::BrownConrady5,
            DistortionKind::Rational8,
            DistortionKind::ThinPrism9,
            DistortionKind::Division1,
        ] {
            let values: Vec<f64> = (1..=kind.dim()).map(|i| 0.01 * i as f64).collect();
            let d = DistortionParams::from_values(kind, &values).unwrap();
            let out = with_leading_radial(&d, -0.5);
            let mut expected = values.clone();
            expected[0] = -0.5;
            assert_eq!(out.kind(), kind);
            assert_eq!(out.values(), expected, "{kind:?}");
        }
    }

    #[test]
    fn keeps_no_distortion_unchanged() {
        let out = with_leading_radial(&DistortionParams::None, -0.5);
        assert_eq!(out.kind(), DistortionKind::None);
    }

    #[test]
    fn keeps_the_inverse_iteration_count() {
        let d = DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: 0.1,
                iters: 20,
                ..Default::default()
            },
        };
        let DistortionParams::BrownConrady5 { params } = with_leading_radial(&d, -0.2) else {
            panic!("model changed");
        };
        assert_eq!(params.k1, -0.2);
        assert_eq!(params.iters, 20);
    }
}

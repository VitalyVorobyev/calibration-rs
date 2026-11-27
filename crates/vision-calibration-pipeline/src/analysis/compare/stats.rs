//! The tests of [`compare_runs`](super::compare_runs): which one a relation
//! supports, the variance of a difference, and the χ² tail.

use nalgebra::{DMatrix, DVector};
use vision_calibration_core::ParameterCovariance;

use super::{DataRelation, DifferenceTest, JointTest, NoTest, RunSide};

/// How the differences of one pair of runs are tested: the relation and
/// the two covariances.
pub(super) struct Scale<'p> {
    relation: DataRelation,
    reference: Option<&'p ParameterCovariance>,
    other: Option<&'p ParameterCovariance>,
}

/// The scales that put a run's stated covariance `Σ = k · c · A⁻¹` on its
/// information `A⁻¹`: the view-scatter factor `k` (1 when not scaled) and
/// the variance factor `c`.
#[derive(Clone, Copy)]
struct NoiseScale {
    view_scatter: f64,
    variance_factor: f64,
}

impl<'p> Scale<'p> {
    pub(super) fn new(
        relation: DataRelation,
        reference: Option<&'p ParameterCovariance>,
        other: Option<&'p ParameterCovariance>,
    ) -> Self {
        Self {
            relation,
            reference,
            other,
        }
    }

    /// Why no test applies, before any σ is read; `None` when the relation
    /// supports one.
    pub(super) fn relation_bar(&self) -> Option<NoTest> {
        match self.relation {
            DataRelation::Independent => {
                (self.reference.is_none() || self.other.is_none()).then_some(NoTest::NoCovariance)
            }
            DataRelation::Nested { .. } => {
                let (Some(a), Some(b)) = (self.reference, self.other) else {
                    return Some(NoTest::NoCovariance);
                };
                if mixed(a) || mixed(b) {
                    Some(NoTest::MixedResiduals)
                } else if a.variance_factor.is_none() || b.variance_factor.is_none() {
                    Some(NoTest::NoVarianceFactor)
                } else {
                    None
                }
            }
            DataRelation::Unknown => Some(NoTest::UnknownRelation),
            DataRelation::Identical => Some(NoTest::Identical),
            _ => Some(NoTest::SharedData),
        }
    }

    /// The nested runs' scales, smaller first; `None` unless nested with
    /// both variance factors.
    fn nested(&self) -> Option<(RunSide, NoiseScale, NoiseScale)> {
        let DataRelation::Nested { smaller } = self.relation else {
            return None;
        };
        let scale = |c: &ParameterCovariance| {
            Some(NoiseScale {
                view_scatter: c.view_scatter.unwrap_or(1.0),
                variance_factor: c.variance_factor?,
            })
        };
        let (a, b) = (scale(self.reference?)?, scale(self.other?)?);
        Some(match smaller {
            RunSide::Reference => (smaller, a, b),
            RunSide::Other => (smaller, b, a),
        })
    }

    /// The nested difference on one noise level, as the smaller run's
    /// variance factor `c_s` times the larger run's view-scatter factor
    /// `k_l` (the scale of `Σ_Δ`), and the per-unit-noise variances
    /// `A_s⁻¹` and `A_l⁻¹` (`A⁻¹ = Σ / (k c)`).
    fn nested_parts(
        &self,
        reference: &DMatrix<f64>,
        other: &DMatrix<f64>,
    ) -> Option<(f64, DMatrix<f64>, DMatrix<f64>)> {
        let (smaller, s, l) = self.nested()?;
        let (sigma_s, sigma_l) = match smaller {
            RunSide::Reference => (reference, other),
            RunSide::Other => (other, reference),
        };
        Some((
            s.variance_factor * l.view_scatter,
            sigma_s / (s.view_scatter * s.variance_factor),
            sigma_l / (l.view_scatter * l.variance_factor),
        ))
    }

    pub(super) fn scalar_test(
        &self,
        delta: f64,
        reference_var: Option<f64>,
        other_var: Option<f64>,
    ) -> DifferenceTest {
        if let Some(reason) = self.relation_bar() {
            return DifferenceTest::NotAvailable { reason };
        }
        let (Some(a), Some(b)) = (reference_var, other_var) else {
            return DifferenceTest::NotAvailable {
                reason: NoTest::NotEstimated,
            };
        };
        let nested = matches!(self.relation, DataRelation::Nested { .. });
        let variance = if nested {
            let scalar = |v: f64| DMatrix::from_element(1, 1, v);
            let Some((scale, info_s, info_l)) = self.nested_parts(&scalar(a), &scalar(b)) else {
                return DifferenceTest::NotAvailable {
                    reason: NoTest::NoVarianceFactor,
                };
            };
            let (i_s, i_l) = (info_s[(0, 0)], info_l[(0, 0)]);
            // The relative growth of the variance without the left-out data.
            let growth = i_s / i_l - 1.0;
            if !(growth > 0.0 && growth.is_finite()) {
                return DifferenceTest::NotAvailable {
                    reason: NoTest::NestedVarianceNotPositive,
                };
            }
            scale * (i_s - i_l)
        } else {
            a + b
        };
        if !(variance > 0.0 && variance.is_finite()) {
            return DifferenceTest::NotAvailable {
                reason: NoTest::SingularCovariance,
            };
        }
        let sigma = variance.sqrt();
        let z = delta / sigma;
        let p_value = chi2_survival(z * z, 1);
        if nested {
            DifferenceTest::Nested { sigma, z, p_value }
        } else {
            DifferenceTest::Independent { sigma, z, p_value }
        }
    }

    /// The joint test of a pose's tangent (or, with `nested_allowed`
    /// false, of a camera's parameters, whose nested joint form is not
    /// offered: [`NoTest::NestedJoint`]).
    pub(super) fn joint_test(
        &self,
        delta: &DVector<f64>,
        reference: Option<DMatrix<f64>>,
        other: Option<DMatrix<f64>>,
        nested_allowed: bool,
    ) -> JointTest {
        if let Some(reason) = self.relation_bar() {
            return JointTest::NotAvailable { reason };
        }
        if !nested_allowed && matches!(self.relation, DataRelation::Nested { .. }) {
            return JointTest::NotAvailable {
                reason: NoTest::NestedJoint,
            };
        }
        let (Some(a), Some(b)) = (reference, other) else {
            return JointTest::NotAvailable {
                reason: NoTest::NotEstimated,
            };
        };
        if delta.is_empty() {
            return JointTest::NotAvailable {
                reason: NoTest::NotEstimated,
            };
        }
        let nested = matches!(self.relation, DataRelation::Nested { .. });
        let found = if nested {
            let Some((scale, info_s, info_l)) = self.nested_parts(&a, &b) else {
                return JointTest::NotAvailable {
                    reason: NoTest::NoVarianceFactor,
                };
            };
            nested_chi2(scale, &info_s, &info_l, delta)
        } else {
            mahalanobis_squared(&(a + b), delta)
                .map(|chi2| (chi2, delta.len()))
                .ok_or(NoTest::SingularCovariance)
        };
        let (chi2, dof) = match found {
            Ok(found) => found,
            Err(reason) => return JointTest::NotAvailable { reason },
        };
        let p_value = chi2_survival(chi2, dof);
        if nested {
            JointTest::Nested { chi2, dof, p_value }
        } else {
            JointTest::Independent { chi2, dof, p_value }
        }
    }
}

/// The share of the largest relative variance growth below which a
/// direction of a nested pose difference counts as uninformed by the
/// left-out data (numerically zero) and leaves the generalized inverse.
const NULL_DIRECTION: f64 = 1e-9;

/// The nested `χ²` through a generalized inverse (Hausman 1978): with
/// `A_l⁻¹ = L Lᵀ` and the eigen-decomposition `W = L⁻¹ (A_s⁻¹ − A_l⁻¹) L⁻ᵀ
/// = Σ μₖ uₖ uₖᵀ` (each `μₖ` the relative growth of the variance along a
/// direction without the left-out data), `χ² = Σ (uₖᵀ L⁻¹ δ)² / (scale ·
/// μₖ)` over the positive `μₖ` (beyond [`NULL_DIRECTION`] of the largest),
/// with as many degrees of freedom as directions kept.
fn nested_chi2(
    scale: f64,
    info_s: &DMatrix<f64>,
    info_l: &DMatrix<f64>,
    delta: &DVector<f64>,
) -> Result<(f64, usize), NoTest> {
    let n = delta.len();
    let symmetric = |m: &DMatrix<f64>| 0.5 * (m + m.transpose());
    let l = symmetric(info_l)
        .cholesky()
        .ok_or(NoTest::SingularCovariance)?
        .l();
    let l_inverse = l.try_inverse().ok_or(NoTest::SingularCovariance)?;
    let w = symmetric(&(&l_inverse * (info_s - info_l) * l_inverse.transpose()));
    let y = &l_inverse * delta;
    let eigen = w.symmetric_eigen();
    let largest = eigen.eigenvalues.max();
    if !(largest > 0.0 && largest.is_finite()) {
        return Err(NoTest::NestedVarianceNotPositive);
    }
    let mut chi2 = 0.0;
    let mut dof = 0;
    for k in 0..n {
        let mu = eigen.eigenvalues[k];
        if mu > NULL_DIRECTION * largest {
            let projection = eigen.eigenvectors.column(k).dot(&y);
            chi2 += projection * projection / (scale * mu);
            dof += 1;
        }
    }
    if chi2.is_finite() {
        Ok((chi2, dof))
    } else {
        Err(NoTest::SingularCovariance)
    }
}

/// Whether a covariance combines residual families, priors or per-camera
/// noise levels: what the nested form has not been validated for.
fn mixed(c: &ParameterCovariance) -> bool {
    c.noise.len() > 1
        || c.noise.iter().any(|n| n.camera.is_some())
        || c.views
            .iter()
            .any(|v| v.labels.iter().any(|l| l.starts_with("robot_delta")))
}

/// `δᵀ Σ⁻¹ δ` through a Cholesky factor of the symmetrized `Σ`; `None`
/// when it is not positive definite.
pub(super) fn mahalanobis_squared(covariance: &DMatrix<f64>, delta: &DVector<f64>) -> Option<f64> {
    if covariance.nrows() != delta.len() || delta.is_empty() {
        return None;
    }
    let symmetric = 0.5 * (covariance + covariance.transpose());
    let cholesky = symmetric.cholesky()?;
    let solved = cholesky.solve(delta);
    let value = delta.dot(&solved);
    value.is_finite().then_some(value)
}

/// The upper tail `P(X > x)` of a χ² variable of `dof` degrees of freedom:
/// the regularized upper incomplete gamma `Q(dof/2, x/2)`.
pub(crate) fn chi2_survival(x: f64, dof: usize) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 0.0 } else { 1.0 };
    }
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(dof as f64 / 2.0, x / 2.0)
}

/// The regularized upper incomplete gamma function `Q(a, x)`, `a > 0`,
/// `x ≥ 0`: the series below `a + 1`, Lentz's continued fraction above
/// (Press et al., *Numerical Recipes*, §6.2), each to `1e-15`.
fn gamma_q(a: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-15;
    const MAX_TERMS: usize = 1000;
    const TINY: f64 = 1e-300;
    let log_prefix = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let (mut term, mut sum, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..MAX_TERMS {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * EPS {
                break;
            }
        }
        (1.0 - sum * log_prefix.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_TERMS {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let step = d * c;
            h *= step;
            if (step - 1.0).abs() < EPS {
                break;
            }
        }
        (log_prefix.exp() * h).clamp(0.0, 1.0)
    }
}

/// `ln Γ(x)`, `x > 0`, by the Lanczos approximation (`g = 7`, 9 terms;
/// relative error below `1e-15`).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x) Γ(1 − x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = C[0];
    for (k, c) in C.iter().enumerate().skip(1) {
        sum += c / (x + k as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

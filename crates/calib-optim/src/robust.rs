use calib_core::Real;

/// Robust loss kernels for iteratively re-weighted least squares (IRLS).
#[derive(Debug, Clone, Copy, Default)]
pub enum RobustKernel {
    /// No robustness, pure L2 (quadratic).
    #[default]
    None,
    /// Huber loss with a given threshold.
    Huber { delta: Real },
    /// Cauchy loss with a scale parameter.
    Cauchy { c: Real },
}

impl RobustKernel {
    /// Return the robust loss `rho(r^2)` and the IRLS weight `w(r)` for a squared residual.
    ///
    /// This is designed for the classic IRLS procedure:
    /// 1. evaluate residuals `r_i` for the current parameters,
    /// 2. compute weights `w_i` using this method,
    /// 3. scale both residuals and Jacobian rows by `sqrt(w_i)` before solving the linearised system.
    pub fn rho_and_weight(self, r2: Real) -> (Real, Real) {
        match self {
            RobustKernel::None => {
                // Standard least squares: rho = r^2, w = 1
                (r2, 1.0)
            }
            RobustKernel::Huber { delta } => {
                let r = r2.sqrt();
                if r <= delta {
                    // Quadratic region
                    (r2, 1.0)
                } else {
                    // Linear region
                    let loss = 2.0 * delta * r - delta * delta;
                    let w = delta / r;
                    (loss, w)
                }
            }
            RobustKernel::Cauchy { c } => {
                // R. Hartley & Z. Zisserman style Cauchy: rho = c^2 * log(1 + r^2 / c^2)
                let t = r2 / (c * c);
                let loss = c * c * (1.0 + t).ln();
                // w = derivative wrt r divided by r => w = 1 / (1 + t)
                let w = 1.0 / (1.0 + t);
                (loss, w)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Real, b: Real, tol: Real) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: {} vs {} (tol={})",
            a,
            b,
            tol
        );
    }

    #[test]
    fn huber_matches_l2_for_small_residuals() {
        let kernel = RobustKernel::Huber { delta: 1.0 };
        let r = 0.5;
        let r2 = r * r;
        let (rho, w) = kernel.rho_and_weight(r2);
        approx_eq(rho, r2, 1e-9);
        approx_eq(w, 1.0, 1e-9);
    }

    #[test]
    fn huber_linear_for_large_residuals() {
        let kernel = RobustKernel::Huber { delta: 1.0 };
        let r = 5.0;
        let r2 = r * r;
        let (rho, w) = kernel.rho_and_weight(r2);
        let expected_rho = 2.0 * 1.0 * r - 1.0;
        let expected_w = 1.0 / r;
        approx_eq(rho, expected_rho, 1e-9);
        approx_eq(w, expected_w, 1e-9);
    }

    #[test]
    fn cauchy_weight_decreases_with_r() {
        let kernel = RobustKernel::Cauchy { c: 1.0 };
        let (_, w_small) = kernel.rho_and_weight(0.1_f64.powi(2));
        let (_, w_large) = kernel.rho_and_weight(10.0_f64.powi(2));
        assert!(
            w_small > 0.9,
            "w_small should be close to 1, got {}",
            w_small
        );
        assert!(
            w_large < 0.02,
            "w_large should be small for large residuals, got {}",
            w_large
        );
        assert!(w_small > w_large);
    }

    fn irls_constant_fit(y: &[Real], kernel: RobustKernel, iters: usize) -> Real {
        let mut x = y.iter().copied().sum::<Real>() / (y.len() as Real);
        for _ in 0..iters {
            let mut num = 0.0;
            let mut den = 0.0;
            for &yi in y {
                let r = x - yi;
                let (_, w) = kernel.rho_and_weight(r * r);
                num += w * yi;
                den += w;
            }
            if den > 0.0 {
                x = num / den;
            }
        }
        x
    }

    #[test]
    fn huber_irls_resists_outliers() {
        let mut y = vec![0.9, 1.0, 1.1, 0.95, 1.05];
        y.push(5.0); // outlier
        y.push(-4.0); // another outlier

        let mean_all = y.iter().copied().sum::<Real>() / (y.len() as Real);
        let x_none = irls_constant_fit(&y, RobustKernel::None, 10);
        let x_huber = irls_constant_fit(&y, RobustKernel::Huber { delta: 0.2 }, 10);

        approx_eq(x_none, mean_all, 1e-9);
        let inlier_mean = y[..5].iter().copied().sum::<Real>() / 5.0;
        let err_none = (x_none - inlier_mean).abs();
        let err_huber = (x_huber - inlier_mean).abs();
        assert!(
            err_huber < err_none,
            "Huber should reduce error vs inliers: err_none {}, err_huber {}",
            err_none,
            err_huber
        );
    }
}

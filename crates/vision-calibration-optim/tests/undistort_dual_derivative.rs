//! The Newton undistortion evaluated under dual numbers carries the
//! derivative of the exact inverse, also with non-zero distortion.
//!
//! Reference: the implicit-function derivative of `distort(x_u; k) = x_d`,
//! `∂x_u/∂k1 = −J⁻¹ · ∂distort/∂k1`, evaluated in `f64` at the converged
//! point, with `∂distort/∂k1 = (x_u·r², y_u·r²)` and the analytic Jacobian `J`
//! of the Brown-Conrady forward map.

use nalgebra::Point2;
use num_dual::Dual64;
use vision_calibration_core::{BrownConrady5, DistortionModel};

/// Wide-angle barrel with tangential terms (folds at a distorted radius of
/// ~0.945).
fn wide_angle() -> BrownConrady5<f64> {
    BrownConrady5 {
        k1: -0.35,
        k2: 0.15,
        k3: -0.03,
        p1: 5e-4,
        p2: -3e-4,
        iters: 0,
    }
}

/// `J = ∂distort/∂(x, y)` of Brown-Conrady at `(x, y)`.
fn forward_jacobian(m: &BrownConrady5<f64>, x: f64, y: f64) -> [[f64; 2]; 2] {
    let r2 = x * x + y * y;
    let radial = 1.0 + m.k1 * r2 + m.k2 * r2 * r2 + m.k3 * r2 * r2 * r2;
    let d_radial = m.k1 + 2.0 * m.k2 * r2 + 3.0 * m.k3 * r2 * r2;
    let cross = 2.0 * x * y * d_radial + 2.0 * m.p1 * x + 2.0 * m.p2 * y;
    [
        [
            radial + 2.0 * x * x * d_radial + 2.0 * m.p1 * y + 6.0 * m.p2 * x,
            cross,
        ],
        [
            cross,
            radial + 2.0 * y * y * d_radial + 6.0 * m.p1 * y + 2.0 * m.p2 * x,
        ],
    ]
}

#[test]
fn undistort_dual_derivative_wrt_k1_matches_the_implicit_function_derivative() {
    let m = wide_angle();
    let m_dual = BrownConrady5 {
        k1: Dual64::new(m.k1, 1.0),
        k2: Dual64::from(m.k2),
        k3: Dual64::from(m.k3),
        p1: Dual64::from(m.p1),
        p2: Dual64::from(m.p2),
        iters: m.iters,
    };
    let n = 41;
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            // The distorted field of a 2048×1536, f = 1800 px image, extended
            // as for a tilted sensor; inside the fold.
            let xd = -0.65 + 1.3 * i as f64 / (n - 1) as f64;
            let yd = -0.5 + 1.0 * j as f64 / (n - 1) as f64;
            let u = m.undistort(&Point2::new(xd, yd));
            let ud = m_dual.undistort(&Point2::new(Dual64::from(xd), Dual64::from(yd)));
            // The value is the plain f64 solution (dual-number division rounds
            // differently, so the two agree to rounding, not bit for bit).
            assert!(
                (ud.x.re - u.x).abs() < 1e-15 && (ud.y.re - u.y).abs() < 1e-15,
                "dual real part differs at ({xd}, {yd})"
            );

            let r2 = u.x * u.x + u.y * u.y;
            let [[a, b], [c, d]] = forward_jacobian(&m, u.x, u.y);
            let det = a * d - b * c;
            let (gx, gy) = (u.x * r2, u.y * r2);
            let ref_x = -(d * gx - b * gy) / det;
            let ref_y = -(a * gy - c * gx) / det;
            let err = (ud.x.eps - ref_x).abs().max((ud.y.eps - ref_y).abs());
            worst = worst.max(err);
        }
    }
    eprintln!("WORST {worst:e}");
    assert!(worst < 1e-14, "worst ∂x_u/∂k1 error {worst:e}");
}

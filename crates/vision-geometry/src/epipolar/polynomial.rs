//! Polynomial constraint system for 5-point essential matrix solver.
//!
//! Implements symbolic polynomial manipulation in three variables
//! (x, y, z) up to degree 3, used to encode the essential matrix constraints
//! in Nistér's 5-point algorithm.

use vision_calibration_core::{Mat3, Real};

/// Monomial ordering for degree-3 polynomials in three variables.
const MONOMIALS: [(u8, u8, u8); 20] = [
    (3, 0, 0), // x^3
    (2, 1, 0), // x^2 y
    (2, 0, 1), // x^2 z
    (1, 2, 0), // x y^2
    (1, 1, 1), // x y z
    (1, 0, 2), // x z^2
    (0, 3, 0), // y^3
    (0, 2, 1), // y^2 z
    (0, 1, 2), // y z^2
    (0, 0, 3), // z^3
    (2, 0, 0), // x^2
    (1, 1, 0), // x y
    (1, 0, 1), // x z
    (0, 2, 0), // y^2
    (0, 1, 1), // y z
    (0, 0, 2), // z^2
    (1, 0, 0), // x
    (0, 1, 0), // y
    (0, 0, 1), // z
    (0, 0, 0), // 1
];

/// Polynomial in three variables (x, y, z) with degree ≤ 3.
#[derive(Clone, Copy)]
pub(crate) struct Poly3 {
    pub coeffs: [Real; 20],
}

impl Poly3 {
    pub fn zero() -> Self {
        Self { coeffs: [0.0; 20] }
    }

    pub fn linear(c0: Real, cx: Real, cy: Real, cz: Real) -> Self {
        let mut p = Self::zero();
        p.coeffs[19] = c0;
        p.coeffs[16] = cx;
        p.coeffs[17] = cy;
        p.coeffs[18] = cz;
        p
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        out
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] - other.coeffs[i];
        }
        out
    }

    pub fn scale(&self, s: Real) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] * s;
        }
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for (i, &ai) in self.coeffs.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            let (ix, iy, iz) = MONOMIALS[i];
            for (j, &bj) in other.coeffs.iter().enumerate() {
                if bj == 0.0 {
                    continue;
                }
                let (jx, jy, jz) = MONOMIALS[j];
                let dx = ix + jx;
                let dy = iy + jy;
                let dz = iz + jz;
                if dx + dy + dz > 3 {
                    continue;
                }
                if let Some(idx) = monomial_index(dx, dy, dz) {
                    out.coeffs[idx] += ai * bj;
                }
            }
        }
        out
    }
}

fn monomial_index(x: u8, y: u8, z: u8) -> Option<usize> {
    MONOMIALS.iter().enumerate().find_map(|(i, &(mx, my, mz))| {
        if mx == x && my == y && mz == z {
            Some(i)
        } else {
            None
        }
    })
}

pub(crate) fn poly_mat_mul(a: &[[Poly3; 3]; 3], b: &[[Poly3; 3]; 3]) -> [[Poly3; 3]; 3] {
    let mut out = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut sum = Poly3::zero();
            for k in 0..3 {
                sum = sum.add(&a[r][k].mul(&b[k][c]));
            }
            out[r][c] = sum;
        }
    }
    out
}

pub(crate) fn poly_transpose(a: &[[Poly3; 3]; 3]) -> [[Poly3; 3]; 3] {
    let mut out = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[c][r];
        }
    }
    out
}

pub(crate) fn poly_det3(a: &[[Poly3; 3]; 3]) -> Poly3 {
    let term1 = a[0][0].mul(&a[1][1].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][1])));
    let term2 = a[0][1].mul(&a[1][0].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][0])));
    let term3 = a[0][2].mul(&a[1][0].mul(&a[2][1]).sub(&a[1][1].mul(&a[2][0])));

    term1.sub(&term2).add(&term3)
}

/// Build the polynomial constraint system for the 5-point algorithm.
///
/// Given four basis essential matrices, constructs the 10 equations that
/// encode det(E) = 0 and trace(E E^T E) - 0.5 * trace(E E^T) E = 0.
pub(crate) fn build_polynomial_system(
    e1: &Mat3,
    e2: &Mat3,
    e3: &Mat3,
    e4: &Mat3,
) -> [[Real; 20]; 10] {
    let mut e = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            e[r][c] = Poly3::linear(e4[(r, c)], e1[(r, c)], e2[(r, c)], e3[(r, c)]);
        }
    }

    let det = poly_det3(&e);

    let e_t = poly_transpose(&e);
    let eet = poly_mat_mul(&e, &e_t);
    let eet_e = poly_mat_mul(&eet, &e);

    let trace = eet[0][0].add(&eet[1][1]).add(&eet[2][2]);

    let mut eqs = [[0.0; 20]; 10];
    eqs[0] = det.coeffs;

    let mut row = 1;
    for r in 0..3 {
        for c in 0..3 {
            let term = eet_e[r][c].scale(2.0).sub(&trace.mul(&e[r][c]));
            eqs[row] = term.coeffs;
            row += 1;
        }
    }

    eqs
}

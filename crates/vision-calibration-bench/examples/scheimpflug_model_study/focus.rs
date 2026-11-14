//! Tilt from focus: the plane of sharp focus (PoF) from per-marker blur, and
//! the lens tilt it implies.
//!
//! **Model.** For a lens imaging through an entrance pupil of diameter `D`,
//! the blur spot of an object point `X` on a sensor conjugate to the plane
//! `π·X = 1` is (geometric optics, small pupil, see the dev note
//! `tilt-from-focus.md`)
//!
//! ```text
//! σ_defocus ≈ κ · |1/z − π·(x, y, 1)| = κ·|1 − π·X| / z,   κ = D·f_px / 4
//! σ_measured² = σ0² + σ_defocus²
//! ```
//!
//! with `X = z·(x, y, 1)` in the frame of the plain pinhole `K'` (the frame
//! whose `z` axis is the sensor normal) and `σ` the per-axis standard
//! deviation of a uniformly filled disc (`D·f_px·|Δ(1/z)|/4`). Written as
//! `g = β·r`, `r = (1/z, x, y, 1)`, `β = κ·(1, −πx, −πy, −πz)`, the model is
//! `σ² = σ0² + (β·r)²`: linear in nothing, but its square is a rank-one
//! quadratic form in `r` plus a constant, which seeds the robust fit.
//!
//! **Inversion (thin lens).** The PoF, the sensor plane `z = −v0` and the lens
//! plane meet in one line, so `π = a/F − e_z/v0` with `a` the lens axis:
//! `a ∝ π + e_z/v0`. The lens tilt therefore needs one metric datum, `v0 =
//! f_px × pixel pitch`; its direction (the azimuth of `π_xy`) needs none.
//!
//! [`Lens`] is the exact paraxial thick lens (focal length, pupil
//! magnification, tilted sensor) the synthetic study draws blur from.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

/// The library's tilt rotation `R = Ry(τy)·Rx(τx)` (OpenCV convention):
/// a point of the routed (lens) frame `X` is `R·X` in the frame of the
/// plain pinhole `K'`.
pub fn tilt_rotation(tau: [f64; 2]) -> Matrix3<f64> {
    let (sx, cx) = tau[0].sin_cos();
    let (sy, cy) = tau[1].sin_cos();
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
    let ry = Matrix3::new(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    ry * rx
}

/// The tilt angles of a rotation of the form [`tilt_rotation`]: read from
/// its entries, exact for such a rotation.
#[cfg(test)]
pub fn tilt_of_rotation(r: &Matrix3<f64>) -> [f64; 2] {
    [r[(1, 2)].atan2(r[(1, 1)]), r[(2, 0)].atan2(r[(0, 0)])]
}

/// The lens axis in the `K'` frame for a tilt (third column of
/// [`tilt_rotation`]).
pub fn axis_of_tilt(tau: [f64; 2]) -> Vector3<f64> {
    tilt_rotation(tau).column(2).into_owned()
}

/// The tilt whose lens axis is `a` (any length, `a_z > 0`), in the
/// library's convention.
pub fn tilt_of_axis(a: &Vector3<f64>) -> [f64; 2] {
    let a = a.normalize();
    [a.y.clamp(-1.0, 1.0).asin(), (-a.x).atan2(a.z)]
}

/// Thin-lens inversion: the tilt implied by the PoF `pi` (`K'` frame,
/// 1/m) for a projection-centre-to-sensor distance `v0` (m).
pub fn tilt_from_pof(pi: &Vector3<f64>, v0: f64) -> [f64; 2] {
    tilt_of_axis(&(pi + Vector3::new(0.0, 0.0, 1.0 / v0)))
}

/// The thin-lens focal length (m) implied by the PoF and `v0`.
pub fn focal_from_pof(pi: &Vector3<f64>, v0: f64) -> f64 {
    1.0 / (pi + Vector3::new(0.0, 0.0, 1.0 / v0)).norm()
}

/// The `1/v0` (1/m) at which the thin-lens inversion of `pi` reproduces the
/// tilt `tau` (least squares along the tilt's direction): `π_xy ≈ (πz +
/// 1/v0)·t`, `t = a_xy/a_z`.
pub fn implied_inv_v0(pi: &Vector3<f64>, tau: [f64; 2]) -> f64 {
    let a = axis_of_tilt(tau);
    let t = [a.x / a.z, a.y / a.z];
    let tt = t[0] * t[0] + t[1] * t[1];
    (pi.x * t[0] + pi.y * t[1]) / tt - pi.z
}

/// Regressors `r = (1/z, x, y, 1)` of a `K'`-frame point.
pub fn regressors(x: &Vector3<f64>) -> [f64; 4] {
    [1.0 / x.z, x.x / x.z, x.y / x.z, 1.0]
}

/// One blur observation: the marker's `K'`-frame position (m) and its
/// measured blur σ (px).
#[derive(Debug, Clone, Copy)]
pub struct BlurObs {
    pub x: Vector3<f64>,
    pub blur: f64,
}

/// A robust fit of the focus model.
#[derive(Debug, Clone)]
pub struct PofFit {
    /// `κ·(1, −πx, −πy, −πz)` and, with a curvature centre, the
    /// coefficient of `|(x, y) − c|²` in `g`.
    pub beta: Vec<f64>,
    /// Centre of the curvature term (`K'` normalized coordinates).
    pub centre: Option<[f64; 2]>,
    /// In-focus blur, px.
    pub s0: f64,
    /// The plane of sharp focus, `π·X = 1`, 1/m.
    pub pi: Vector3<f64>,
    /// `D·f_px/4`, px·m.
    pub kappa: f64,
    /// Covariance of `pi` (from the robust normal equations).
    pub cov_pi: Matrix3<f64>,
    /// Covariance of all parameters (`beta` then `s0`).
    pub cov: Vec<Vec<f64>>,
    /// Robust residual scale (1.4826·MAD), px.
    pub scale: f64,
}

impl PofFit {
    /// The defocus term `g` at a point (signed, px).
    pub fn g(&self, x: &Vector3<f64>) -> f64 {
        g_of(&self.beta, self.centre, x)
    }

    /// The predicted blur σ at a point, px.
    pub fn predict(&self, x: &Vector3<f64>) -> f64 {
        (self.s0 * self.s0 + self.g(x).powi(2)).sqrt()
    }

    /// The field curvature `c` of `1/z* = π·(x, y, 1) + c·|(x, y) − centre|²`,
    /// 1/m, and its σ.
    pub fn curvature(&self) -> Option<(f64, f64)> {
        self.centre?;
        let c = -self.beta[4] / self.kappa;
        // Delta method on −β4/β0.
        let (b0, b4) = (self.beta[0], self.beta[4]);
        let j = [b4 / (b0 * b0), -1.0 / b0];
        let v = j[0] * j[0] * self.cov[0][0]
            + 2.0 * j[0] * j[1] * self.cov[0][4]
            + j[1] * j[1] * self.cov[4][4];
        Some((c, v.sqrt()))
    }
}

fn curv_reg(centre: [f64; 2], r: &[f64; 4]) -> f64 {
    (r[1] - centre[0]).powi(2) + (r[2] - centre[1]).powi(2)
}

fn g_of(beta: &[f64], centre: Option<[f64; 2]>, x: &Vector3<f64>) -> f64 {
    let r = regressors(x);
    let mut g = (0..4).map(|k| beta[k] * r[k]).sum::<f64>();
    if let Some(c) = centre {
        g += beta[4] * curv_reg(c, &r);
    }
    g
}

/// The full quadratic of `σ²` in `r` (10 coefficients, plain least
/// squares): the unconstrained model the physical one is a rank-one case
/// of.
pub fn quad_fit(obs: &[BlurObs]) -> Option<[f64; 10]> {
    if obs.len() < 20 {
        return None;
    }
    let mut ata = SMatrix::<f64, 10, 10>::zeros();
    let mut atb = SVector::<f64, 10>::zeros();
    for o in obs {
        let q = quad_row(&regressors(&o.x));
        let q = SVector::<f64, 10>::from_row_slice(&q);
        ata += q * q.transpose();
        atb += q * o.blur * o.blur;
    }
    let c = ata.cholesky()?.solve(&atb);
    Some(std::array::from_fn(|k| c[k]))
}

/// The same free quadratic fitted like the focus model: robust (Huber) on
/// `σ = sqrt(max(q·ρ, 0))`, seeded by [`quad_fit`].
pub fn quad_fit_robust(obs: &[BlurObs]) -> Option<[f64; 10]> {
    let init = quad_fit(obs)?;
    let model = |th: &[f64], o: &BlurObs, j: &mut [f64]| -> f64 {
        let q = quad_row(&regressors(&o.x));
        let v: f64 = (0..10).map(|k| th[k] * q[k]).sum();
        let m = v.max(1e-4).sqrt();
        for k in 0..10 {
            j[k] = if v > 1e-4 { q[k] / (2.0 * m) } else { 0.0 };
        }
        m
    };
    let (th, _, _) = robust_lm(obs, init.to_vec(), &model)?;
    Some(std::array::from_fn(|k| th[k]))
}

fn quad_row(r: &[f64; 4]) -> [f64; 10] {
    let mut out = [0.0; 10];
    let mut k = 0;
    for i in 0..4 {
        for j in i..4 {
            out[k] = r[i] * r[j];
            k += 1;
        }
    }
    out
}

/// The blur σ predicted by [`quad_fit`] coefficients.
pub fn quad_predict(c: &[f64; 10], x: &Vector3<f64>) -> f64 {
    let q = quad_row(&regressors(x));
    (0..10).map(|k| c[k] * q[k]).sum::<f64>().max(0.0).sqrt()
}

/// Seed of the robust fit: the rank-one part of the full quadratic (in
/// regressors scaled to unit RMS), `σ0²` the rest of its constant.
fn seed(obs: &[BlurObs]) -> Option<([f64; 4], f64)> {
    let c = quad_fit(obs)?;
    let mut m = SMatrix::<f64, 4, 4>::zeros();
    let mut k = 0;
    for i in 0..4 {
        for j in i..4 {
            let v = if i == j { c[k] } else { c[k] / 2.0 };
            m[(i, j)] = v;
            m[(j, i)] = v;
            k += 1;
        }
    }
    let mut s = [0.0; 4];
    for o in obs {
        let r = regressors(&o.x);
        for i in 0..4 {
            s[i] += r[i] * r[i];
        }
    }
    let n = obs.len() as f64;
    let s: [f64; 4] = std::array::from_fn(|i| (s[i] / n).sqrt().max(1e-12));
    let ms = SMatrix::<f64, 4, 4>::from_fn(|i, j| m[(i, j)] * s[i] * s[j]);
    let e = ms.symmetric_eigen();
    let (imax, lmax) = e
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, l)| (i, *l))?;
    if lmax <= 0.0 {
        return None;
    }
    let v = e.eigenvectors.column(imax);
    let mut beta: [f64; 4] = std::array::from_fn(|i| lmax.sqrt() * v[i] / s[i]);
    if beta[0] < 0.0 {
        beta.iter_mut().for_each(|b| *b = -*b);
    }
    let s0sq = (m[(3, 3)] - beta[3] * beta[3]).max(0.04);
    Some((beta, s0sq.sqrt()))
}

type Model<'a> = dyn Fn(&[f64], &BlurObs, &mut [f64]) -> f64 + 'a;

/// Robust (Huber at `1.345·scale`, the scale re-estimated as 1.4826·MAD
/// until it settles) Levenberg-Marquardt on `blur − model`. Returns the
/// parameters, their covariance (robust sandwich-free approximation
/// `s²·(JᵀWJ)⁻¹`) and the residual scale.
fn robust_lm(
    obs: &[BlurObs],
    init: Vec<f64>,
    model: &Model<'_>,
) -> Option<(Vec<f64>, Vec<Vec<f64>>, f64)> {
    let np = init.len();
    if obs.len() <= np + 5 {
        return None;
    }
    let mut th = init;
    let mut jbuf = vec![0.0; np];
    let mut resid = |th: &[f64]| -> Vec<f64> {
        obs.iter()
            .map(|o| o.blur - model(th, o, &mut jbuf))
            .collect()
    };
    let mad_scale = |e: &[f64]| {
        let mut a: Vec<f64> = e.iter().map(|v| v.abs()).collect();
        a.sort_by(f64::total_cmp);
        (1.4826 * a[a.len() / 2]).max(1e-6)
    };
    let huber = |e: &[f64], k: f64| -> f64 {
        e.iter()
            .map(|v| {
                let a = v.abs();
                if a <= k {
                    0.5 * a * a
                } else {
                    k * (a - 0.5 * k)
                }
            })
            .sum()
    };
    let normal = |th: &[f64], k: f64| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut jtj = vec![vec![0.0; np]; np];
        let mut jte = vec![0.0; np];
        let mut j = vec![0.0; np];
        for o in obs {
            let e = o.blur - model(th, o, &mut j);
            let w = if e.abs() <= k { 1.0 } else { k / e.abs() };
            for a in 0..np {
                jte[a] += w * j[a] * e;
                for b in 0..np {
                    jtj[a][b] += w * j[a] * j[b];
                }
            }
        }
        (jtj, jte)
    };
    let solve = |a: &[Vec<f64>], b: &[f64]| -> Option<Vec<f64>> {
        let n = b.len();
        let m = nalgebra::DMatrix::from_fn(n, n, |i, j| a[i][j]);
        let v = nalgebra::DVector::from_column_slice(b);
        m.cholesky().map(|c| c.solve(&v).iter().copied().collect())
    };
    let mut k = 1.345 * mad_scale(&resid(&th));
    for _outer in 0..8 {
        let mut lambda = 1e-3;
        let mut cost = huber(&resid(&th), k);
        for _ in 0..100 {
            let (jtj, jte) = normal(&th, k);
            let mut improved = false;
            for _ in 0..12 {
                let mut a = jtj.clone();
                for (i, row) in a.iter_mut().enumerate() {
                    row[i] += lambda * jtj[i][i].max(1e-12);
                }
                let Some(step) = solve(&a, &jte) else {
                    lambda *= 10.0;
                    continue;
                };
                let cand: Vec<f64> = th.iter().zip(&step).map(|(t, s)| t + s).collect();
                let c = huber(&resid(&cand), k);
                if c < cost {
                    let rel = (cost - c) / cost.max(1e-300);
                    th = cand;
                    cost = c;
                    lambda = (lambda * 0.3).max(1e-12);
                    improved = rel > 1e-13;
                    break;
                }
                lambda *= 10.0;
            }
            if !improved {
                break;
            }
        }
        let k_new = 1.345 * mad_scale(&resid(&th));
        if (k_new - k).abs() < 1e-6 * k {
            break;
        }
        k = k_new;
    }
    let e = resid(&th);
    let scale = mad_scale(&e);
    let k = 1.345 * scale;
    let (jtj, _) = normal(&th, k);
    let wsum: f64 = e
        .iter()
        .map(|v| if v.abs() <= k { v * v } else { k * v.abs() })
        .sum();
    let s2 = wsum / (obs.len() as f64 - np as f64);
    let inv = nalgebra::DMatrix::from_fn(np, np, |i, j| jtj[i][j]).try_inverse()?;
    let cov = (0..np)
        .map(|i| (0..np).map(|j| s2 * inv[(i, j)]).collect())
        .collect();
    Some((th, cov, scale))
}

/// Robust fit of `σ = sqrt(σ0² + g²)`, `g = β·r`, plus `β4·|(x, y) − c|²`
/// when a curvature centre `c` is given. `None` when the data does not
/// determine it (too few points, singular normal equations, or `κ ≤ 0`).
pub fn fit_pof(obs: &[BlurObs], centre: Option<[f64; 2]>) -> Option<PofFit> {
    let (b0, s0) = seed(obs)?;
    let nb = if centre.is_some() { 5 } else { 4 };
    let mut init: Vec<f64> = b0.to_vec();
    if centre.is_some() {
        init.push(0.0);
    }
    init.push(s0);
    let model = |th: &[f64], o: &BlurObs, j: &mut [f64]| -> f64 {
        let r = regressors(&o.x);
        let s0 = th[nb];
        let g = g_of(&th[..nb], centre, &o.x);
        let m = (s0 * s0 + g * g).sqrt().max(1e-9);
        for q in 0..4 {
            j[q] = g * r[q] / m;
        }
        if let Some(c) = centre {
            j[4] = g * curv_reg(c, &r) / m;
        }
        j[nb] = s0 / m;
        m
    };
    let (mut th, cov, scale) = robust_lm(obs, init, &model)?;
    if th[0] < 0.0 {
        for t in th.iter_mut().take(nb) {
            *t = -*t;
        }
    }
    let kappa = th[0];
    if kappa <= 0.0 {
        return None;
    }
    let pi = Vector3::new(-th[1] / kappa, -th[2] / kappa, -th[3] / kappa);
    // dπ_i/dβ: −β_{i+1}/κ, with ∂/∂κ = β_{i+1}/κ² (the sign flip of β
    // leaves the covariance unchanged).
    let mut jp = SMatrix::<f64, 3, 4>::zeros();
    for i in 0..3 {
        jp[(i, 0)] = th[i + 1] / (kappa * kappa);
        jp[(i, i + 1)] = -1.0 / kappa;
    }
    let cb = SMatrix::<f64, 4, 4>::from_fn(|i, j| cov[i][j]);
    let cov_pi = jp * cb * jp.transpose();
    Some(PofFit {
        beta: th[..nb].to_vec(),
        centre,
        s0: th[nb].abs(),
        pi,
        kappa,
        cov_pi,
        cov,
        scale,
    })
}

/// The 2×2 covariance of [`tilt_from_pof`] at `pi` (numerical Jacobian).
pub fn tilt_cov(pi: &Vector3<f64>, cov_pi: &Matrix3<f64>, v0: f64) -> [[f64; 2]; 2] {
    let h = 1e-6 * pi.norm().max(1.0);
    let mut j = SMatrix::<f64, 2, 3>::zeros();
    for k in 0..3 {
        let mut a = *pi;
        let mut b = *pi;
        a[k] += h;
        b[k] -= h;
        let ta = tilt_from_pof(&a, v0);
        let tb = tilt_from_pof(&b, v0);
        j[(0, k)] = (ta[0] - tb[0]) / (2.0 * h);
        j[(1, k)] = (ta[1] - tb[1]) / (2.0 * h);
    }
    let c = j * cov_pi * j.transpose();
    [[c[(0, 0)], c[(0, 1)]], [c[(1, 0)], c[(1, 1)]]]
}

/// RQ decomposition of a 3×3 matrix with `det > 0`: `m = k·r`, `k` upper
/// triangular with a positive diagonal, `r` a rotation (Gram-Schmidt on the
/// rows from the bottom).
pub fn rq3(m: &Matrix3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let m3: Vector3<f64> = m.row(2).transpose();
    let m2: Vector3<f64> = m.row(1).transpose();
    let m1: Vector3<f64> = m.row(0).transpose();
    let k33 = m3.norm();
    let r3 = m3 / k33;
    let k23 = m2.dot(&r3);
    let u2 = m2 - k23 * r3;
    let k22 = u2.norm();
    let r2 = u2 / k22;
    let k13 = m1.dot(&r3);
    let k12 = m1.dot(&r2);
    let u1 = m1 - k13 * r3 - k12 * r2;
    let k11 = u1.norm();
    let r1 = u1 / k11;
    let k = Matrix3::new(k11, k12, k13, 0.0, k22, k23, 0.0, 0.0, k33);
    let r = Matrix3::from_rows(&[r1.transpose(), r2.transpose(), r3.transpose()]);
    (k, r)
}

/// A paraxial thick lens with a tilted sensor, in the lens frame (object
/// space, origin at the entrance pupil, `z` along the axis towards the
/// object). Image space is measured from the rear principal plane `H'`,
/// `w` positive behind it; the front principal plane `H` is at `z =
/// F(1 − m_p)/m_p` and the exit pupil at `w = F(1 − m_p)` (pupil
/// magnification `m_p`). The sensor is the plane `w = s'0 + gx·x + gy·y`,
/// its pixel axes the rows of `R(τ_s)` ([`tilt_rotation`], `z` mirrored
/// into image space), so a thin lens is exactly the library's tilted
/// pinhole with tilt `τ_s`.
#[derive(Debug, Clone)]
pub struct Lens {
    /// Focal length, m.
    pub f: f64,
    /// Pupil magnification `D_exit / D_entrance`.
    pub mp: f64,
    /// Axial image distance of the sensor from `H'`, m.
    pub s_img: f64,
    /// Sensor orientation: the physical tilt in the library's convention.
    pub tau_s: [f64; 2],
    /// Pixel pitch, m.
    pub pitch: f64,
    /// f-number (entrance pupil diameter `F/N`).
    pub fnum: f64,
    /// Pixel where the lens axis meets the sensor (the library's
    /// principal point).
    pub centre: [f64; 2],
    /// Field curvature: `1/z*` along a ray gains `fc·ρ²`, `ρ` the ray's
    /// normalized radius about the lens axis (1/m).
    pub fc: f64,
}

impl Lens {
    /// A lens focused (on axis) at object distance `z_focus` from the
    /// entrance pupil, with a sensor tilted by `tau` (library convention:
    /// the sensor normal is the lens-frame image of the third row of
    /// [`tilt_rotation`]; for `m_p = 1` the library's tilt is `tau`).
    pub fn new(
        f: f64,
        mp: f64,
        z_focus: f64,
        tau: [f64; 2],
        pitch: f64,
        fnum: f64,
        centre: [f64; 2],
    ) -> Self {
        let zh = f * (1.0 - mp) / mp;
        let s0 = z_focus - zh;
        let s_img = 1.0 / (1.0 / f - 1.0 / s0);
        Self {
            f,
            mp,
            s_img,
            tau_s: tau,
            pitch,
            fnum,
            centre,
            fc: 0.0,
        }
    }

    /// Entrance pupil diameter, m.
    pub fn pupil(&self) -> f64 {
        self.f / self.fnum
    }

    /// Axial magnification `s'/s` at the on-axis focus.
    pub fn magnification(&self) -> f64 {
        let s = 1.0 / (1.0 / self.f - 1.0 / self.s_img);
        self.s_img / s
    }

    /// Sensor slope `(gx, gy)`.
    pub fn slope(&self) -> [f64; 2] {
        let r3 = tilt_rotation(self.tau_s).row(2).transpose();
        [r3.x / r3.z, r3.y / r3.z]
    }

    /// The object point conjugate to the image-space point `(x, y, w)`.
    fn conj(&self, y: [f64; 3]) -> Vector3<f64> {
        let s = 1.0 / (1.0 / self.f - 1.0 / y[2]);
        let m = -y[2] / s;
        let zh = self.f * (1.0 - self.mp) / self.mp;
        Vector3::new(y[0] / m, y[1] / m, s + zh)
    }

    /// The plane of sharp focus in the lens frame (`π·X = 1`), the
    /// conjugate of the sensor plane.
    pub fn pof_lens(&self) -> Vector3<f64> {
        let h = 1e-3;
        let g = self.slope();
        let p = |x: f64, y: f64| self.conj([x, y, self.s_img + g[0] * x + g[1] * y]);
        let (a, b, c) = (p(0.0, 0.0), p(h, 0.0), p(0.0, h));
        let n = (b - a).cross(&(c - a));
        n / n.dot(&a)
    }

    /// The chief-ray map: lens-frame direction `(x, y, 1)` (or any point)
    /// to homogeneous pixel. Exit-pupil chief rays have their lateral slope
    /// divided by `m_p`; the pixel axes undo the image inversion.
    pub fn chief(&self) -> Matrix3<f64> {
        let we = self.f * (1.0 - self.mp);
        let q0 = self.s_img - we;
        let [gx, gy] = self.slope();
        let r = tilt_rotation(self.tau_s);
        let e1 = Vector3::new(r[(0, 0)], r[(0, 1)], -r[(0, 2)]);
        let e2 = Vector3::new(r[(1, 0)], r[(1, 1)], -r[(1, 2)]);
        // In-plane offset from the axial sensor point: −(q/m_p)·(x, y, g·x).
        let l = |e: &Vector3<f64>| [e.x + gx * e.z, e.y + gy * e.z];
        let (l1, l2) = (l(&e1), l(&e2));
        let s = q0 / self.pitch;
        let [cx, cy] = self.centre;
        Matrix3::new(
            s * l1[0] + cx * gx,
            s * l1[1] + cx * gy,
            cx * self.mp,
            s * l2[0] + cy * gx,
            s * l2[1] + cy * gy,
            cy * self.mp,
            gx,
            gy,
            self.mp,
        )
    }

    /// The calibrated pinhole: `K'` and the tilt rotation `R` with
    /// `chief = K'·R` (what a distortion-based calibration recovers).
    pub fn pinhole(&self) -> (Matrix3<f64>, Matrix3<f64>) {
        let m = self.chief();
        let m = if m.determinant() < 0.0 { -m } else { m };
        let (k, r) = rq3(&m);
        (k / k[(2, 2)], r)
    }

    /// The tilt a distortion-based calibration reports: the direction of
    /// the lens axis relative to `K'`'s axis (the third row of the RQ
    /// rotation), in the library's convention. A two-axis tilt under
    /// `m_p ≠ 1` adds a small roll to the RQ rotation, which the library's
    /// zero-skew model absorbs in the poses; the tilt is the roll-free part.
    pub fn tilt_d(&self) -> [f64; 2] {
        let (_, r) = self.pinhole();
        let r3 = r.row(2);
        [(-r3[1]).atan2(r3[2]), r3[0].clamp(-1.0, 1.0).asin()]
    }

    /// Pixel of a lens-frame point.
    pub fn project(&self, x: &Vector3<f64>) -> [f64; 2] {
        let p = self.chief() * x;
        [p.x / p.z, p.y / p.z]
    }

    /// Per-axis standard deviation (px) of the geometric blur spot of the
    /// lens-frame point `x`: the entrance pupil (uniformly filled, `n`
    /// sunflower samples) cast through `x` onto the plane of sharp focus
    /// (the tangent plane at the ray's in-focus point when `fc ≠ 0`), and
    /// imaged by the chief-ray map.
    pub fn spot_sigma(&self, x: &Vector3<f64>, n: usize) -> f64 {
        let m = self.chief();
        let pi0 = self.pof_lens();
        let w = x / x.z;
        let rho2 = w.x * w.x + w.y * w.y;
        let inv = pi0.dot(&w) + self.fc * rho2;
        let xs = w / inv;
        let pi = pi0 / pi0.dot(&xs);
        let r = self.pupil() / 2.0;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let (mut su, mut sv, mut suu, mut svv) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..n {
            let rr = r * ((k as f64 + 0.5) / n as f64).sqrt();
            let (s, c) = (k as f64 * golden).sin_cos();
            let e = Vector3::new(rr * c, rr * s, 0.0);
            let d = e - x;
            let t = (1.0 - pi.dot(x)) / pi.dot(&d);
            let p = m * (x + t * d);
            let (u, v) = (p.x / p.z, p.y / p.z);
            su += u;
            sv += v;
            suu += u * u;
            svv += v * v;
        }
        let nn = n as f64;
        let var = (suu / nn - (su / nn).powi(2)) + (svv / nn - (sv / nn).powi(2));
        (var.max(0.0) / 2.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(mp: f64, tau: [f64; 2]) -> Lens {
        Lens::new(0.008, mp, 0.25, tau, 7e-6, 2.0, [364.0, 271.0])
    }

    #[test]
    fn tilt_rotation_round_trips() {
        for tau in [[-0.09, 0.0], [0.05, -0.12], [0.2, 0.15]] {
            let t = tilt_of_rotation(&tilt_rotation(tau));
            assert!((t[0] - tau[0]).abs() < 1e-14 && (t[1] - tau[1]).abs() < 1e-14);
            let t = tilt_of_axis(&axis_of_tilt(tau));
            assert!((t[0] - tau[0]).abs() < 1e-14 && (t[1] - tau[1]).abs() < 1e-14);
        }
    }

    /// The chief-ray map of a thin lens is the library's tilted pinhole: its
    /// RQ rotation is `R(τ)`, its principal point is where the axis meets
    /// the sensor, and the library's `K·H(τ)` reproduces it.
    #[test]
    fn thin_lens_pinhole_is_the_library_tilt() {
        let tau = [-0.09, 0.03];
        let l = lens(1.0, tau);
        let (k, r) = l.pinhole();
        let t = tilt_of_rotation(&r);
        assert!((t[0] - tau[0]).abs() < 1e-12 && (t[1] - tau[1]).abs() < 1e-12);
        let t = l.tilt_d();
        assert!((t[0] - tau[0]).abs() < 1e-12 && (t[1] - tau[1]).abs() < 1e-12);
        assert!(k[(0, 1)].abs() < 1e-9 * k[(0, 0)], "skew {}", k[(0, 1)]);
        let c = l.project(&Vector3::new(0.0, 0.0, 1.0));
        assert!((c[0] - 364.0).abs() < 1e-9 && (c[1] - 271.0).abs() < 1e-9);
    }

    /// Pupil magnification: the distortion-defined tilt is the physical
    /// sensor tilt with its tangent divided by `m_p`.
    #[test]
    fn calibrated_tilt_is_physical_over_pupil_magnification() {
        for mp in [0.7, 1.0, 1.4] {
            let l = lens(mp, [-0.09, 0.0]);
            let t = l.tilt_d();
            let want = (0.09f64.tan() / mp).atan();
            assert!(
                (t[0] + want).abs() < 1e-12,
                "mp {mp}: {} vs {}",
                t[0],
                -want
            );
        }
    }

    /// Thin lens: the inversion of the conjugate PoF with the true `v0`
    /// returns the tilt exactly; with `m_p ≠ 1` it returns
    /// `tan τ_f = tan τ_d·(m_p + μ)/(1 + μ + (1 − m_p)·tan²τ_d)` (single
    /// axis, exact in paraxial optics).
    #[test]
    fn inversion_thin_exact_thick_scaled() {
        for mp in [1.0, 0.7, 1.4] {
            let l = lens(mp, [-0.09, 0.0]);
            let (k, _) = l.pinhole();
            let td = l.tilt_d();
            let pi = tilt_rotation(td) * l.pof_lens();
            let v0 = k[(0, 0)] * l.pitch;
            let tf = tilt_from_pof(&pi, v0);
            let mu = l.magnification();
            let t2 = td[0].tan().powi(2);
            let want = td[0].tan() * (mp + mu) / (1.0 + mu + (1.0 - mp) * t2);
            assert!(
                (tf[0].tan() - want).abs() < 1e-9,
                "mp {mp}: {} vs {want}",
                tf[0].tan()
            );
            assert!(tf[1].abs() < 1e-12);
            // The implied 1/v0 reproduces the distortion tilt.
            let w = implied_inv_v0(&pi, td);
            let back = tilt_from_pof(&pi, 1.0 / w);
            assert!((back[0] - td[0]).abs() < 1e-12);
            if mp == 1.0 {
                assert!((w * v0 - 1.0).abs() < 1e-9);
            }
        }
    }

    /// The exact spot of a thin lens follows `κ·|1/z − π·(x, y, 1)|` in the
    /// `K'` frame to within the small-pupil terms.
    #[test]
    fn spot_follows_the_defocus_model() {
        let l = lens(1.0, [-0.09, 0.02]);
        let (k, r) = l.pinhole();
        let pi = r * l.pof_lens();
        let kappa = l.pupil() * k[(0, 0)] / 4.0;
        let mut worst: f64 = 0.0;
        for (x, y, z) in [(0.02, 0.03, 0.2), (-0.05, -0.04, 0.32), (0.0, 0.05, 0.15)] {
            let xl = Vector3::new(x, y, z);
            let s = l.spot_sigma(&xl, 4000);
            let xk = r * xl;
            let rr = regressors(&xk);
            let model = kappa * (rr[0] - pi.x * rr[1] - pi.y * rr[2] - pi.z).abs();
            worst = worst.max((s - model).abs() / model);
        }
        assert!(worst < 0.03, "relative spot error {worst}");
    }

    /// Noise-free blur from the exact spot: the fit recovers the PoF and,
    /// through the inversion, the tilt.
    #[test]
    fn fit_recovers_the_pof() {
        let tau = [-0.09, 0.0];
        let l = lens(1.0, tau);
        let (k, r) = l.pinhole();
        let mut obs = Vec::new();
        for i in 0..30 {
            for j in 0..20 {
                let z = 0.15 + 0.2 * f64::from(i) / 29.0;
                let xn = -0.25 + 0.5 * f64::from(j) / 19.0;
                let yn = -0.2 + 0.4 * ((i * 7 + j * 3) % 20) as f64 / 19.0;
                let xl = Vector3::new(xn * z, yn * z, z);
                let sp = l.spot_sigma(&xl, 400);
                obs.push(BlurObs {
                    x: r * xl,
                    blur: (0.5f64.powi(2) + sp * sp).sqrt(),
                });
            }
        }
        let fit = fit_pof(&obs, None).expect("fit");
        let pi = r * l.pof_lens();
        assert!(
            (fit.pi - pi).norm() < 0.03 * pi.norm(),
            "{} vs {}",
            fit.pi,
            pi
        );
        let tf = tilt_from_pof(&fit.pi, k[(0, 0)] * l.pitch);
        assert!((tf[0] - tau[0]).abs() < 0.003, "{tf:?}");
    }
}

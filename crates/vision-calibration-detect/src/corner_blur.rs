//! The per-corner blur `calib-targets` measures on a chessboard-like
//! detection, as [`FeatureBlur`]s.

use calib_targets::blur::{BlurDecline, CornerBlur};

use crate::FeatureBlur;

/// Each corner's estimate as a [`FeatureBlur`], in the order given; `None`
/// for a corner `calib-targets` declined.
///
/// `calib-targets` reports the kernel `Σ` in px², image axes, as the total
/// blur including the pixel aperture (a perfectly focused ideal camera has
/// `Σ = I/12`), and the 3×3 covariance of `(Σxx, Σxy, Σyy)` in px⁴: the
/// conventions of [`FeatureBlur`], so the values carry over unchanged.
pub(crate) fn feature_blurs(
    estimates: Vec<Result<CornerBlur, BlurDecline>>,
) -> Vec<Option<FeatureBlur>> {
    estimates
        .into_iter()
        .map(|estimate| estimate.ok().as_ref().and_then(feature_blur))
        .collect()
}

/// One estimate as a [`FeatureBlur`], or `None` when it is not finite, `Σ`
/// is not positive definite or its covariance has a negative variance. A
/// delivered estimate is none of these (`calib-targets` declines a kernel
/// below `I/12`); the check keeps a corner from carrying one that is.
fn feature_blur(blur: &CornerBlur) -> Option<FeatureBlur> {
    let k = blur.kernel;
    let psf_px = [f64::from(k.xx), f64::from(k.xy), f64::from(k.yy)];
    let cov: [[f64; 3]; 3] = blur.kernel_covariance.map(|row| row.map(f64::from));
    let [xx, xy, yy] = psf_px;
    let finite = psf_px
        .iter()
        .chain(cov.iter().flatten())
        .all(|v| v.is_finite());
    let positive = xx > 0.0 && yy > 0.0 && xx * yy - xy * xy > 0.0;
    let variances = (0..3).all(|i| cov[i][i] >= 0.0);
    (finite && positive && variances).then(|| FeatureBlur::new(psf_px, cov))
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_targets::blur::{BlurFitQuality, BlurKernel};

    fn estimate(xx: f32, xy: f32, yy: f32, var: f32) -> CornerBlur {
        let cov = [[var, 0.0, 0.0], [0.0, var, 0.0], [0.0, 0.0, var]];
        CornerBlur::new(
            BlurKernel::new(xx, xy, yy),
            cov,
            BlurFitQuality::new(0.5, 100, 200.0, 1.0, 0.0),
        )
    }

    #[test]
    fn carries_the_kernel_and_its_covariance() {
        let blurs = feature_blurs(vec![
            Ok(estimate(1.5, 0.25, 2.0, 1e-3)),
            Err(BlurDecline::TooBlurred),
        ]);
        let b = blurs[0].expect("a delivered estimate");
        assert_eq!(b.psf_px, [1.5, 0.25, 2.0]);
        assert_eq!(b.cov[1][1], f64::from(1e-3_f32));
        assert_eq!(b.cov[0][1], 0.0);
        assert!(blurs[1].is_none());
    }

    #[test]
    fn drops_an_estimate_that_is_not_a_covariance() {
        let bad = [
            estimate(f32::NAN, 0.0, 1.0, 1e-3),
            estimate(1.0, 0.0, 1.0, f32::INFINITY),
            estimate(1.0, 1.0, 1.0, 1e-3),
            estimate(-1.0, 0.0, 1.0, 1e-3),
            estimate(1.0, 0.0, 1.0, -1e-3),
        ];
        let blurs = feature_blurs(bad.into_iter().map(Ok).collect());
        assert!(blurs.iter().all(Option::is_none), "{blurs:?}");
    }
}

/// Renders of a board through a camera of known blur, and the blur measured
/// on them.
#[cfg(test)]
pub(crate) mod known_blur {
    use calib_targets::printable::{PageSize, PageSpec, PrintableTargetDocument};

    use std::f64::consts::TAU;

    use crate::Feature;

    /// Gray levels of the print's black and white in [`camera_image`].
    const BLACK: f64 = 30.0;
    const WHITE: f64 = 220.0;

    /// Sub-pixels per pixel and axis the print is rasterized at in
    /// [`camera_image`]; odd, so a pixel's centre is a sub-pixel's centre.
    const N: u32 = 5;

    /// The blur [`camera_image`] renders for a lens of `sigma` px, as the
    /// half-trace `tr Σ / 2`, px²: the lens's `σ²` plus the unit pixel
    /// box's `1/12`.
    fn truth(sigma: f64) -> f64 {
        sigma * sigma + 1.0 / 12.0
    }

    /// `document`'s board as a camera with a Gaussian lens blur of `sigma`
    /// px and square pixels sees it, with `cell_px` pixels per cell, on a
    /// `page_mm` square page, with seeded Gaussian noise of `noise` gray
    /// levels.
    ///
    /// The print is rasterized with cells of 25.4 mm at `cell_px · N` dots
    /// per inch: `cell_px · N` sub-pixels per cell, each edge on a sub-pixel
    /// boundary, so the print is constant on every sub-pixel (the circles of
    /// a PuzzleBoard as the rasterizer draws them). Each pixel is that print
    /// integrated against the lens Gaussian convolved with the pixel's box,
    /// which is exact sub-pixel by sub-pixel: the weight of the sub-pixel
    /// `j` sub-pixels from the pixel's centre is, per axis, the integral of
    /// `Φ((t + N/2)/(Nσ)) − Φ((t − N/2)/(Nσ))` over `t ∈ [j − ½, j + ½]`,
    /// normalized to `N`. The pixel's gray level maps black to 30 and white
    /// to 220, so the noise is never clipped (clipping flattens the plateaus
    /// and reads as a sharper edge).
    pub(crate) fn camera_image(
        mut document: PrintableTargetDocument,
        cell_px: u32,
        page_mm: f64,
        sigma: f64,
        noise: f64,
        seed: u64,
    ) -> image::GrayImage {
        document.page = PageSpec::default().with_size(PageSize::Custom {
            width_mm: page_mm,
            height_mm: page_mm,
        });
        document.render.png_dpi = cell_px * N;
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        let fine = image::load_from_memory(&bundle.png_bytes)
            .expect("decode rendered PNG")
            .to_luma8();
        let (w, h) = (fine.width() as usize, fine.height() as usize);
        let n = N as usize;
        let (cw, ch) = (w / n, h / n);

        let weights = pixel_weights(sigma);
        let r = (weights.len() / 2) as isize;
        // Sub-pixel → [0, 1] reflectance, the page extended by its border
        // (paper).
        let at = |x: isize, y: isize| {
            let x = x.clamp(0, w as isize - 1) as u32;
            let y = y.clamp(0, h as isize - 1) as u32;
            f64::from(fine.get_pixel(x, y).0[0]) / 255.0
        };
        let centre = |i: u32| (i as usize * n + n / 2) as isize;
        // Along x at each pixel's centre column, for every sub-pixel row.
        let rows: Vec<f64> = (0..h as isize)
            .flat_map(|y| (0..cw as u32).map(move |x| (x, y)))
            .map(|(x, y)| {
                weights
                    .iter()
                    .enumerate()
                    .map(|(i, k)| k * at(centre(x) + i as isize - r, y))
                    .sum()
            })
            .collect();
        let mut rng = SplitMix64(seed);
        image::GrayImage::from_fn(cw as u32, ch as u32, |x, y| {
            let reflectance: f64 = weights
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    let row = (centre(y) + i as isize - r).clamp(0, h as isize - 1) as usize;
                    k * rows[row * cw + x as usize]
                })
                .sum();
            let value = BLACK + (WHITE - BLACK) * reflectance + noise * rng.normal();
            image::Luma([value.round().clamp(0.0, 255.0) as u8])
        })
    }

    /// The per-axis weights of the sub-pixels around a pixel's centre (see
    /// [`camera_image`]), out to 6σ beyond the pixel's half-width, summing
    /// to 1.
    fn pixel_weights(sigma: f64) -> Vec<f64> {
        let (s, half) = (sigma * f64::from(N), 0.5 * f64::from(N));
        let r = (6.0 * s + half).ceil() as isize;
        // ∫ Φ(u) du = u Φ(u) + φ(u).
        let antiderivative = |u: f64| u * phi_cdf(u) + (-0.5 * u * u).exp() / TAU.sqrt();
        // ∫_a^b Φ((t + c)/s) dt.
        let edge = |a: f64, b: f64, c: f64| {
            s * (antiderivative((b + c) / s) - antiderivative((a + c) / s))
        };
        let raw: Vec<f64> = (-r..=r)
            .map(|j| {
                let (a, b) = (j as f64 - 0.5, j as f64 + 0.5);
                edge(a, b, half) - edge(a, b, -half)
            })
            .collect();
        let total: f64 = raw.iter().sum();
        raw.into_iter().map(|v| v / total).collect()
    }

    /// The standard normal distribution function, from `erfc` by
    /// Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7).
    fn phi_cdf(u: f64) -> f64 {
        let x = u.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erfc = poly * (-x * x).exp();
        if u >= 0.0 {
            1.0 - 0.5 * erfc
        } else {
            0.5 * erfc
        }
    }

    /// Check the blur measured on [`camera_image`] at lens blur `sigma`
    /// against the rendered one.
    ///
    /// Each feature's `tr Σ / 2` has the standard error
    /// `½ √(c_xx,xx + 2 c_xx,yy + c_yy,yy)` from its covariance. The stated
    /// standard errors must be honest: the RMS of the features' errors over
    /// their standard errors is at most 1.5 and none exceeds 5 (measured on
    /// six noise seeds: RMS 0.8–1.2, largest 3.8). The mean must be within
    /// 0.5 % of the truth (measured within ±0.2 %). Nine in ten features
    /// must carry a blur; at these blurs, well under the estimator's limit
    /// of 0.3 of a cell, none is declined.
    pub(crate) fn assert_matches_render(features: &[Feature], sigma: f64) {
        let truth = truth(sigma);
        let measured: Vec<(f64, f64)> = features
            .iter()
            .filter_map(|f| f.blur)
            .map(|b| {
                let [xx, _, yy] = b.psf_px;
                let var = 0.25 * (b.cov[0][0] + 2.0 * b.cov[0][2] + b.cov[2][2]);
                (0.5 * (xx + yy), var.sqrt())
            })
            .collect();
        assert!(
            measured.len() * 10 >= features.len() * 9,
            "{} of {} features carry a blur",
            measured.len(),
            features.len()
        );
        let n = measured.len() as f64;
        let z: Vec<f64> = measured.iter().map(|(v, se)| (v - truth) / se).collect();
        let rms_z = (z.iter().map(|z| z * z).sum::<f64>() / n).sqrt();
        let max_z = z.iter().fold(0.0_f64, |m, z| m.max(z.abs()));
        assert!(
            rms_z <= 1.5 && max_z < 5.0,
            "σ {sigma}: errors over stated standard errors: RMS {rms_z}, max {max_z}"
        );
        let mean = measured.iter().map(|m| m.0).sum::<f64>() / n;
        assert!(
            (mean / truth - 1.0).abs() < 0.005,
            "σ {sigma}: mean tr Σ / 2 {mean} px² vs {truth} px²"
        );
    }

    /// A seeded generator of standard normal samples (SplitMix64 and
    /// Box–Muller).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // In (0, 1]: 53 random bits.
            ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }
}

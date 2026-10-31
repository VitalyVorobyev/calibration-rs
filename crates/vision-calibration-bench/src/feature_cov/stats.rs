//! Seeded randomness, medians and the bootstrap intervals of the study.
//!
//! Every interval is a percentile bootstrap: the statistic is recomputed on
//! resampled data and the 2.5 % and 97.5 % quantiles bound it. Pooled
//! statistics are hierarchical: the items (views or splits) are resampled
//! within each dataset and the statistic pooled by the median over datasets.

use serde::Serialize;

/// Deterministic splitmix64 generator.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Uniform index in `0..n` (`n > 0`).
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.index(i + 1));
        }
    }
}

/// FNV-1a hash of a label: a stable seed for one dataset and metric.
pub fn seed_of(label: &str) -> u64 {
    label.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Median of the finite values; NaN when there are none.
pub fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        0.5 * (v[m / 2 - 1] + v[m / 2])
    }
}

/// A point estimate with its 95 % bootstrap interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Interval {
    /// The statistic on the data.
    pub point: f64,
    /// 2.5 % bootstrap quantile.
    pub lo: f64,
    /// 97.5 % bootstrap quantile.
    pub hi: f64,
}

impl Interval {
    /// The interval lies above 0.
    pub fn above_zero(&self) -> bool {
        self.lo > 0.0
    }

    /// The interval lies below 0.
    pub fn below_zero(&self) -> bool {
        self.hi < 0.0
    }

    /// The interval excludes 0.
    pub fn excludes_zero(&self) -> bool {
        self.above_zero() || self.below_zero()
    }
}

/// What an effect (arm − `uniform`, lower is better) amounts to once its
/// practical margin is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// The interval lies below 0 and the effect reaches −margin.
    Better,
    /// The interval lies above 0 and the effect reaches +margin.
    Worse,
    /// Anything else: an interval that includes 0, or an effect smaller than
    /// the margin whatever its interval.
    Neutral,
}

impl Class {
    /// Classify `effect` against `margin` (> 0).
    pub fn of(effect: Interval, margin: f64) -> Self {
        if effect.below_zero() && effect.point <= -margin {
            Class::Better
        } else if effect.above_zero() && effect.point >= margin {
            Class::Worse
        } else {
            Class::Neutral
        }
    }
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64) {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    v.sort_by(f64::total_cmp);
    let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
    (q(0.025), q(0.975))
}

/// `stat` of each dataset's items, pooled by the median over datasets, with
/// a hierarchical bootstrap: each dataset's items are resampled with
/// replacement, `resamples` times. A single dataset gives the plain
/// bootstrap of `stat`.
pub fn pooled<T: Copy>(
    datasets: &[&[T]],
    stat: impl Fn(&[T]) -> f64,
    resamples: usize,
    seed: u64,
) -> Interval {
    let point = median(datasets.iter().map(|d| stat(d)));
    let mut rng = Rng::new(seed);
    let mut buffer: Vec<T> = Vec::new();
    let boots = (0..resamples)
        .map(|_| {
            median(datasets.iter().map(|d| {
                if d.is_empty() {
                    return f64::NAN;
                }
                buffer.clear();
                buffer.extend((0..d.len()).map(|_| d[rng.index(d.len())]));
                stat(&buffer)
            }))
        })
        .collect();
    let (lo, hi) = quantiles(boots);
    Interval { point, lo, hi }
}

/// One view's paired squared-error sums: `(Σe²_reference, Σe²_arm, n)`.
pub type PairedSums = (f64, f64, usize);

/// Pooled RMS of the arm minus pooled RMS of the reference.
pub fn rms_difference(items: &[PairedSums]) -> f64 {
    let (a, x, n) = items
        .iter()
        .fold((0.0, 0.0, 0usize), |t, v| (t.0 + v.0, t.1 + v.1, t.2 + v.2));
    if n == 0 {
        return f64::NAN;
    }
    (x / n as f64).sqrt() - (a / n as f64).sqrt()
}

/// Pooled RMS of one side (`arm = false`: the reference) of paired sums.
pub fn rms_of(items: &[PairedSums], arm: bool) -> f64 {
    let (s, n) = items.iter().fold((0.0, 0usize), |t, v| {
        (t.0 + if arm { v.1 } else { v.0 }, t.1 + v.2)
    });
    if n == 0 {
        f64::NAN
    } else {
        (s / n as f64).sqrt()
    }
}

/// Median of a slice (for [`pooled`]).
pub fn median_of(items: &[f64]) -> f64 {
    median(items.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_skips_non_finite_values() {
        assert_eq!(median([3.0, f64::NAN, 1.0, 2.0]), 2.0);
        assert_eq!(median([4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median([f64::NAN]).is_nan());
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_point() {
        let items: Vec<f64> = (0..40).map(|i| f64::from(i % 7) - 3.0).collect();
        let a = pooled(&[&items], median_of, 2000, 7);
        let b = pooled(&[&items], median_of, 2000, 7);
        assert_eq!(a, b);
        assert!(a.lo <= a.point && a.point <= a.hi, "{a:?}");
    }

    #[test]
    fn rms_difference_is_arm_minus_reference() {
        // Reference RMS 1, arm RMS 2 over 4 points.
        let items = [(2.0, 8.0, 2), (2.0, 8.0, 2)];
        assert!((rms_difference(&items) - 1.0).abs() < 1e-12);
        assert!((rms_of(&items, false) - 1.0).abs() < 1e-12);
        assert!((rms_of(&items, true) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn an_effect_counts_only_past_its_margin() {
        let at = |point, lo, hi| Interval { point, lo, hi };
        assert_eq!(
            Class::of(at(-0.0001, -0.0002, -0.00005), 0.005),
            Class::Neutral
        );
        assert_eq!(Class::of(at(-0.01, -0.02, -0.005), 0.005), Class::Better);
        assert_eq!(Class::of(at(0.01, 0.002, 0.02), 0.005), Class::Worse);
        assert_eq!(Class::of(at(0.01, -0.002, 0.02), 0.005), Class::Neutral);
    }

    #[test]
    fn a_shifted_sample_has_an_interval_that_excludes_zero() {
        let mut rng = Rng::new(1);
        let items: Vec<f64> = (0..200).map(|_| 1.0 + 0.1 * rng.normal()).collect();
        let i = pooled(&[&items], median_of, 1000, 3);
        assert!(i.above_zero(), "{i:?}");
    }
}

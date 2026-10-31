//! The pooled statistics and the pre-registered decision rule, as Amendment
//! 1 of `docs/notes/feature-covariance-study.md` states it.
//!
//! Every effect is classed **better**, **worse** or **neutral**: better or
//! worse only when its 95 % interval excludes 0 *and* its size reaches the
//! practical margin (E1: `max(0.005 px, 1 %` of uniform's held-out RMS`)`;
//! E2: 2 % of uniform's split RMS). Pooling is over the decisive cameras, by
//! the median of the per-camera statistics with a hierarchical bootstrap.

use std::collections::BTreeMap;

use serde::Serialize;

use super::arms::Arm;
use super::geometry::ZSums;
use super::planar::{
    UnitRaw, UnitSummary, coverage_68_of, coverage_95_of, e1_margin, e2_margin, paired, ratio_of,
    split_differences, split_z,
};
use super::stats::{Class, Interval, median, median_of, pooled, rms_difference, seed_of};
use super::timing::Timing;

/// Nominal χ²(2) coverage at `z² = 2.30` and `6.18`.
pub const NOMINAL: [f64; 2] = [0.683, 0.954];
/// An arm is materially more honest when its `|r − 1|` is at most this
/// fraction of uniform's…
const HONESTY_RATIO: f64 = 0.8;
/// …and each coverage is at least this much closer to nominal.
const COVERAGE_GAIN: f64 = 0.05;
/// The synthetic gate's tolerance on `r`, reported as "honest".
pub const HONEST_RATIO: [f64; 2] = [0.85, 1.15];
/// The synthetic gate's tolerance on each coverage, reported as "honest".
pub const HONEST_COVERAGE: f64 = 0.05;
/// The largest detection-time increase that counts as cheap.
const MAX_ADDED_TIME: f64 = 0.15;

/// A pooled effect with its margin and class.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PooledEffect {
    /// Median over cameras of the per-camera effect, with its interval.
    pub interval: Interval,
    /// The pooled margin, px.
    pub margin: f64,
    /// Better, worse or neutral.
    pub class: Class,
}

impl PooledEffect {
    fn new(interval: Interval, margin: f64) -> Self {
        Self {
            interval,
            margin,
            class: Class::of(interval, margin),
        }
    }
}

/// Pooled E3 of one arm.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PooledHonesty {
    /// Median over cameras of `r`.
    pub ratio: Interval,
    /// Median over cameras of `|r − 1|`.
    pub gap: Interval,
    /// Median over cameras of the coverage below `z² = 2.30`.
    pub coverage_68: Interval,
    /// Median over cameras of the coverage below `z² = 6.18`.
    pub coverage_95: Interval,
    /// Within the synthetic gate's tolerance: `r` in 0.85–1.15 and both
    /// coverages within ±5 points of nominal.
    pub honest: bool,
}

impl PooledHonesty {
    fn of(cameras: &[Vec<ZSums>], resamples: usize, seed: u64) -> Option<Self> {
        let cameras: Vec<&[ZSums]> = cameras
            .iter()
            .filter(|c| !c.is_empty())
            .map(Vec::as_slice)
            .collect();
        if cameras.is_empty() {
            return None;
        }
        let ratio = pooled(&cameras, ratio_of, resamples, seed);
        let coverage_68 = pooled(&cameras, coverage_68_of, resamples, seed ^ 1);
        let coverage_95 = pooled(&cameras, coverage_95_of, resamples, seed ^ 2);
        Some(Self {
            ratio,
            gap: pooled(&cameras, |s| (ratio_of(s) - 1.0).abs(), resamples, seed ^ 3),
            coverage_68,
            coverage_95,
            honest: (HONEST_RATIO[0]..=HONEST_RATIO[1]).contains(&ratio.point)
                && (coverage_68.point - NOMINAL[0]).abs() <= HONEST_COVERAGE
                && (coverage_95.point - NOMINAL[1]).abs() <= HONEST_COVERAGE,
        })
    }

    /// `self` is materially more honest than `other`: `|r − 1|` at most 0.8 ×
    /// `other`'s, and each coverage at least 5 points closer to nominal.
    pub fn materially_more_honest_than(&self, other: &Self) -> bool {
        let closer = |mine: f64, theirs: f64, nominal: f64| {
            (mine - nominal).abs() <= (theirs - nominal).abs() - COVERAGE_GAIN
        };
        self.gap.point <= HONESTY_RATIO * other.gap.point
            && closer(self.coverage_68.point, other.coverage_68.point, NOMINAL[0])
            && closer(self.coverage_95.point, other.coverage_95.point, NOMINAL[1])
    }
}

/// One arm's pooled figures over a set of cameras.
#[derive(Debug, Clone, Serialize)]
pub struct Pooled {
    /// Cameras pooled.
    pub units: Vec<String>,
    /// E1 held-out RMS effect.
    pub heldout: PooledEffect,
    /// E1 homography RMS effect.
    pub homography: PooledEffect,
    /// E2 split-half effect (decisive grid).
    pub split: PooledEffect,
    /// E3 of this arm (decisive grid).
    pub honesty: Option<PooledHonesty>,
    /// E3 of `uniform` on the same cameras.
    pub uniform_honesty: Option<PooledHonesty>,
}

impl Pooled {
    /// The pooled effects.
    pub fn effects(&self) -> [PooledEffect; 3] {
        [self.heldout, self.homography, self.split]
    }
}

/// Pool `arm` over the cameras `units` that have it.
pub fn pool(arm: Arm, units: &[(&UnitSummary, &UnitRaw)], resamples: usize) -> Option<Pooled> {
    let units: Vec<_> = units
        .iter()
        .filter(|(s, _)| s.arms.contains_key(&arm))
        .collect();
    if units.is_empty() {
        return None;
    }
    let seed = |what: &str| seed_of(&format!("pooled/{what}/{arm}"));
    let held: Vec<_> = units.iter().map(|(_, r)| paired(&r.heldout, arm)).collect();
    let homo: Vec<_> = units
        .iter()
        .map(|(_, r)| paired(&r.homography, arm))
        .collect();
    let split: Vec<_> = units
        .iter()
        .map(|(_, r)| split_differences(r, arm))
        .collect();
    let z = |a: Arm| -> Vec<Vec<ZSums>> { units.iter().map(|(_, r)| split_z(r, a)).collect() };
    let uniform = |pick: fn(&super::planar::ArmSummary) -> f64| {
        median(units.iter().map(|(s, _)| pick(&s.arms[&Arm::Uniform])))
    };
    let m1 = e1_margin(uniform(|a| a.heldout_rms_px));
    let m2 = e2_margin(uniform(|a| a.split_rms_px));
    Some(Pooled {
        units: units.iter().map(|(s, _)| s.unit.clone()).collect(),
        heldout: PooledEffect::new(
            pooled(&slices(&held), rms_difference, resamples, seed("heldout")),
            m1,
        ),
        homography: PooledEffect::new(
            pooled(
                &slices(&homo),
                rms_difference,
                resamples,
                seed("homography"),
            ),
            m1,
        ),
        split: PooledEffect::new(
            pooled(&slices(&split), median_of, resamples, seed("split")),
            m2,
        ),
        honesty: PooledHonesty::of(&z(arm), resamples, seed("z2")),
        uniform_honesty: PooledHonesty::of(&z(Arm::Uniform), resamples, seed("z2-uniform")),
    })
}

fn slices<T>(v: &[Vec<T>]) -> Vec<&[T]> {
    v.iter().map(Vec::as_slice).collect()
}

/// What the rule concludes for one weighting arm.
#[derive(Debug, Clone, Serialize)]
pub struct ArmVerdict {
    /// Pooled over the decisive cameras that have the arm.
    pub pooled: Option<Pooled>,
    /// Decisive cameras where E1 (either measure) or E2 is **worse**.
    pub worse_on: Vec<String>,
    /// Decisive cameras where E1 (either measure) or E2 is not neutral.
    pub not_neutral_on: Vec<String>,
    /// Decisive cameras without the arm.
    pub absent_on: Vec<String>,
    /// Both pooled E1 measures are **better**.
    pub pooled_e1_better: bool,
    /// A pooled effect is **worse**.
    pub pooled_worse: bool,
    /// E3 is materially more honest than `uniform`.
    pub more_honest: bool,
    /// `uniform` is materially more honest than the arm.
    pub less_honest: bool,
    /// E5: the detection-time increase, when timed.
    pub added_time: Option<f64>,
    /// Every flip condition holds.
    pub flips: bool,
    /// Every effect is neutral (E1, E2 per decisive camera and pooled; E3
    /// neither materially more nor less honest).
    pub neutral: bool,
}

/// The rule's outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "outcome", content = "arm")]
pub enum Outcome {
    /// Flip the default to this arm.
    Flip(Arm),
    /// Keep it opt-in, documented as a reporting and weighting option.
    KeepOptIn,
    /// Recommend against weighting by it.
    RecommendAgainst,
    /// The default stays `uniform`, but no clause of the rule describes the
    /// result (e.g. better but too slow, or no decisive camera).
    DefaultStays,
}

/// The rule applied to every weighting arm.
#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    /// Cameras that pass the power check.
    pub decisive: Vec<String>,
    /// Per weighting arm.
    pub arms: BTreeMap<Arm, ArmVerdict>,
    /// The conclusion.
    pub outcome: Outcome,
}

/// Apply the decision rule.
pub fn decide(
    units: &[(&UnitSummary, &UnitRaw)],
    timing: Option<&Timing>,
    resamples: usize,
) -> Decision {
    let decisive: Vec<(&UnitSummary, &UnitRaw)> =
        units.iter().copied().filter(|(s, _)| s.decisive).collect();
    let arms: BTreeMap<Arm, ArmVerdict> = Arm::WEIGHTED
        .iter()
        .map(|&arm| (arm, verdict(arm, &decisive, timing, resamples)))
        .collect();
    let outcome = if decisive.is_empty() {
        Outcome::DefaultStays
    } else if let Some(arm) = Arm::WEIGHTED.iter().find(|a| arms[a].flips) {
        Outcome::Flip(*arm)
    } else if arms.values().any(|v| !v.worse_on.is_empty()) {
        Outcome::RecommendAgainst
    } else if arms.values().all(|v| v.neutral) {
        Outcome::KeepOptIn
    } else {
        Outcome::DefaultStays
    };
    Decision {
        decisive: decisive.iter().map(|(s, _)| s.unit.clone()).collect(),
        arms,
        outcome,
    }
}

fn verdict(
    arm: Arm,
    decisive: &[(&UnitSummary, &UnitRaw)],
    timing: Option<&Timing>,
    resamples: usize,
) -> ArmVerdict {
    let classes = |s: &UnitSummary| -> Vec<Class> {
        s.arms
            .get(&arm)
            .map(|a| {
                [a.heldout_effect, a.homography_effect, a.split_effect]
                    .into_iter()
                    .flatten()
                    .map(|e| e.class)
                    .collect()
            })
            .unwrap_or_default()
    };
    let on = |test: fn(&Class) -> bool| -> Vec<String> {
        decisive
            .iter()
            .filter(|(s, _)| classes(s).iter().any(test))
            .map(|(s, _)| s.unit.clone())
            .collect()
    };
    let worse_on = on(|c| *c == Class::Worse);
    let not_neutral_on = on(|c| *c != Class::Neutral);
    let absent_on: Vec<String> = decisive
        .iter()
        .filter(|(s, _)| !s.arms.contains_key(&arm))
        .map(|(s, _)| s.unit.clone())
        .collect();
    let pooled = pool(arm, decisive, resamples);
    let pooled_e1_better = pooled
        .as_ref()
        .is_some_and(|p| p.heldout.class == Class::Better && p.homography.class == Class::Better);
    let pooled_worse = pooled
        .as_ref()
        .is_some_and(|p| p.effects().iter().any(|e| e.class == Class::Worse));
    let honesty = pooled
        .as_ref()
        .and_then(|p| Some((p.honesty?, p.uniform_honesty?)));
    let more_honest = honesty.is_some_and(|(a, u)| a.materially_more_honest_than(&u));
    let less_honest = honesty.is_some_and(|(a, u)| u.materially_more_honest_than(&a));
    let added_time = timing.and_then(|t| t.pooled.added.get(&arm).copied());
    let flips = pooled.is_some()
        && absent_on.is_empty()
        && worse_on.is_empty()
        && !pooled_worse
        && (pooled_e1_better || more_honest)
        && added_time.is_some_and(|t| t <= MAX_ADDED_TIME);
    let neutral = pooled.as_ref().is_none_or(|p| {
        not_neutral_on.is_empty()
            && p.effects().iter().all(|e| e.class == Class::Neutral)
            && !more_honest
            && !less_honest
    });
    ArmVerdict {
        pooled,
        worse_on,
        not_neutral_on,
        absent_on,
        pooled_e1_better,
        pooled_worse,
        more_honest,
        less_honest,
        added_time,
        flips,
        neutral,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn honesty(gap: f64, c68: f64, c95: f64) -> PooledHonesty {
        let at = |v| Interval {
            point: v,
            lo: v,
            hi: v,
        };
        PooledHonesty {
            ratio: at(1.0 + gap),
            gap: at(gap),
            coverage_68: at(c68),
            coverage_95: at(c95),
            honest: false,
        }
    }

    #[test]
    fn material_honesty_needs_the_ratio_and_both_coverages() {
        let uniform = honesty(3.0, 0.20, 0.40);
        assert!(honesty(2.0, 0.30, 0.50).materially_more_honest_than(&uniform));
        // Ratio not 20 % closer.
        assert!(!honesty(2.5, 0.30, 0.50).materially_more_honest_than(&uniform));
        // One coverage not 5 points closer.
        assert!(!honesty(2.0, 0.24, 0.50).materially_more_honest_than(&uniform));
        assert!(uniform.materially_more_honest_than(&honesty(4.0, 0.1, 0.3)));
    }
}

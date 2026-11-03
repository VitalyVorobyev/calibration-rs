//! A plain chessboard's labels, matched across the cameras of a rig view.
//!
//! The detector labels each image on its own: `+u` is the board axis that
//! runs closest to the image's `+x`, `+v` the one closest to `+y`, and
//! `(0, 0)` the corner they start from. Two cameras of one view therefore
//! give the same physical corner the same label only while they see the
//! board turned alike. A whole-board detection is the board's own lattice
//! up to a turn in its plane (two candidates for a rectangular board, four
//! for a square one), and this module picks, per view, the candidate of
//! every camera that agrees with the view's anchor camera.
//!
//! # The rule
//!
//! Each whole-board detection gets an affine fit `pixel ≈ A · label + b`;
//! the direction of `A`'s `+u` column is the board's direction `θ` in that
//! image, and turning the labels by a quarter turns `θ` with them.
//!
//! 1. **The cameras' relative turn.** For two cameras of a rigid rig the
//!    difference of their `θ` in one view is close to a constant `φ` (their
//!    relative roll, as the board shows it), known from `θ` only modulo
//!    the board's symmetry: 180° for a rectangular board, 90° for a square
//!    one. Its value modulo the symmetry is the consensus over the views
//!    the cameras share (the largest set within a quarter of the symmetry
//!    of each other, which must be a strict majority). The multiple of the
//!    symmetry comes from how the board moves through the two images: its
//!    centre moves through both alike, turned by `φ`, so a 2D rotation fit
//!    between the centres' tracks gives `φ` whole. When that motion is too
//!    small or too incoherent to tell, the cameras are taken to be mounted
//!    the same way up (the multiple nearest 0); when it is coherent but
//!    disagrees with the board's direction, the turn is unknown. Camera 0
//!    is the reference, and every other camera is related to the
//!    lower-numbered camera it shares the most views with.
//! 2. **The anchor of a view.** With one target pose per view (rig
//!    extrinsics), the anchor is the view's first camera, its labels turned
//!    onto the declared extent. With a target frame fixed through the
//!    views (hand-eye, the laser rigs), the frame is the one camera 0
//!    labels: camera 0 anchors every view it sees, and must see the board
//!    as declared; in a view it misses, the first camera that predicts
//!    camera 0's labels (the board within 45° of camera 0's `+x`) anchors
//!    it.
//! 3. **The other cameras.** Each takes the candidate whose direction is
//!    nearest the anchor's turned by their relative turn, when it is within
//!    a quarter of the symmetry (45° or 22.5°) and every other candidate
//!    is at least twice as far. Anything else is
//!    [`RejectReason::AmbiguousChessboardTurn`], or
//!    [`RejectReason::TurnedChessboard`] for a fixed frame no camera of the
//!    view places.
//!
//! A detection the rule keeps unturned keeps its labels bit for bit, so
//! data whose cameras already agree calibrates exactly as before.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use vision_calibration_detect::Feature;

use super::view_filter::{BoardExtent, ChessboardGrid, LabelScope, RejectReason};

/// Fewest views two cameras share for the board's motion through them to
/// tell their relative turn.
const MIN_MOTION_VIEWS: usize = 3;

/// Least coherence of that motion (the length of the summed rotation over
/// the summed magnitudes: 1 when every displacement pair is turned by the
/// same angle, about `1/√n` for unrelated ones).
const MIN_MOTION_COHERENCE: f64 = 0.7;

/// Match every rig view's whole-board chessboard detections, in place:
/// `views[view][camera]` is `None` where the camera has no kept detection.
///
/// Detections are turned onto the agreed labels; the ones that cannot be
/// matched are set to `None` and returned as `(view, camera, reason)`, in
/// view order.
pub(super) fn match_rig_labels(
    grid: &ChessboardGrid,
    scope: LabelScope,
    views: &mut [Vec<Option<Vec<Feature>>>],
) -> Vec<(usize, usize, RejectReason)> {
    let mut rejected = Vec::new();
    // A board one corner wide, or with pixels that are not finite, has no
    // direction to match.
    let boards: Vec<Vec<Option<Fit>>> = views
        .iter_mut()
        .enumerate()
        .map(|(index, cameras)| {
            cameras
                .iter_mut()
                .enumerate()
                .map(|(c, slot)| {
                    let fit = slot.as_deref().and_then(|f| Fit::new(grid, f));
                    if slot.is_some() && fit.is_none() {
                        *slot = None;
                        rejected.push((index, c, RejectReason::AmbiguousChessboardTurn));
                    }
                    fit
                })
                .collect()
        })
        .collect();
    let symmetry = if grid.rows == grid.cols {
        FRAC_PI_2
    } else {
        PI
    };
    let tolerance = symmetry / 4.0;
    let turns = relative_turns(&boards, symmetry, tolerance);

    for (index, (cameras, boards)) in views.iter_mut().zip(&boards).enumerate() {
        let kept: Vec<usize> = (0..boards.len()).filter(|&c| boards[c].is_some()).collect();
        let Some(&first) = kept.first() else {
            continue;
        };
        let board = |c: usize| boards[c].as_ref().expect("a kept camera has a fit");
        let anchor = match scope {
            // One target pose per view: the view's first camera with a
            // known turn (camera 0's always is), else its first camera, its
            // labels turned onto the declared extent.
            LabelScope::Image | LabelScope::View => {
                let c = kept
                    .iter()
                    .copied()
                    .find(|&c| turns[c].is_some())
                    .unwrap_or(first);
                Some((c, board(c).free_turn()))
            }
            LabelScope::Fixed => fixed_anchor(&kept, &turns, board),
        };
        let mut reject = |camera: usize, reason| {
            cameras[camera] = None;
            rejected.push((index, camera, reason));
        };
        let Some((anchor, anchor_turn)) = anchor else {
            // No camera places the fixed frame: the board is turned as
            // camera 0 sees it, or would.
            for &c in &kept {
                reject(
                    c,
                    if turns[c].is_some() {
                        RejectReason::TurnedChessboard
                    } else {
                        RejectReason::AmbiguousChessboardTurn
                    },
                );
            }
            continue;
        };
        // The board's direction as camera 0 sees it, or would.
        let reference = turns[anchor].map(|phi| board(anchor).direction(anchor_turn) - phi);
        let mut chosen = vec![None; boards.len()];
        chosen[anchor] = Some(anchor_turn);
        for &c in kept.iter().filter(|&&c| c != anchor) {
            match reference
                .zip(turns[c])
                .and_then(|(r, phi)| board(c).nearest_turn(r + phi, tolerance))
            {
                Some(turn) => chosen[c] = Some(turn),
                None => reject(c, RejectReason::AmbiguousChessboardTurn),
            }
        }
        for (c, turn) in chosen.into_iter().enumerate() {
            if let (Some(turn), Some(features), Some(fit)) = (turn, cameras[c].as_mut(), &boards[c])
            {
                turn_labels(grid, fit, features, turn);
            }
        }
    }
    rejected.sort_by_key(|&(view, camera, _)| (view, camera));
    rejected
}

/// The anchor of a view whose target frame is fixed through the views:
/// camera 0 when it sees the board (and only when it sees it as declared),
/// otherwise the first camera with a known turn whose labels, turned by its
/// relative turn, put the board within 45° of camera 0's `+x`.
fn fixed_anchor<'a>(
    kept: &[usize],
    turns: &[Option<f64>],
    board: impl Fn(usize) -> &'a Fit,
) -> Option<(usize, u8)> {
    if kept.first() == Some(&0) {
        return (board(0).extent == BoardExtent::Declared).then_some((0, 0));
    }
    kept.iter().find_map(|&c| {
        let (b, phi) = (board(c), turns[c]?);
        let (turn, off) = b
            .candidates()
            .map(|k| (k, wrap(b.direction(k) - phi).abs()))
            .min_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)))?;
        (off <= FRAC_PI_4).then_some((c, turn))
    })
}

/// Every camera's turn relative to camera 0, radians; `None` where it
/// cannot be told.
fn relative_turns(boards: &[Vec<Option<Fit>>], symmetry: f64, tolerance: f64) -> Vec<Option<f64>> {
    let n = boards.first().map_or(0, Vec::len);
    let shared = |a: usize, c: usize| -> Vec<(&Fit, &Fit)> {
        boards
            .iter()
            .filter_map(|view| view[a].as_ref().zip(view[c].as_ref()))
            .collect()
    };
    let mut turns: Vec<Option<f64>> = vec![None; n];
    if n > 0 {
        turns[0] = Some(0.0);
    }
    for c in 1..n {
        // Relate camera c to the lower-numbered camera with a known turn it
        // shares the most views with (the lowest on a tie).
        let mut best: Option<(f64, Vec<(&Fit, &Fit)>)> = None;
        for (a, phi_a) in turns[..c].iter().enumerate() {
            let Some(phi_a) = *phi_a else { continue };
            let pairs = shared(a, c);
            if !pairs.is_empty() && best.as_ref().is_none_or(|(_, p)| pairs.len() > p.len()) {
                best = Some((phi_a, pairs));
            }
        }
        turns[c] = best.and_then(|(phi_a, pairs)| {
            turn_between(&pairs, symmetry, tolerance).map(|phi| wrap(phi_a + phi))
        });
    }
    turns
}

/// The turn of camera `c` relative to camera `a` from the views they share
/// (`(a's board, c's board)` per view), radians.
fn turn_between(pairs: &[(&Fit, &Fit)], symmetry: f64, tolerance: f64) -> Option<f64> {
    // The board's direction in c less its direction in a, each from the
    // detection's own labels turned onto the declared extent, modulo the
    // board's symmetry.
    let offsets: Vec<f64> = pairs
        .iter()
        .map(|(a, c)| {
            let (da, dc) = (a.direction(a.free_turn()), c.direction(c.free_turn()));
            wrap_modulo(dc - da, symmetry)
        })
        .collect();
    // The largest set of offsets within the tolerance of one of them (the
    // first on a tie) must be a strict majority; its mean is the turn
    // modulo the symmetry.
    let near = |x: f64, y: f64| wrap_modulo(x - y, symmetry).abs() <= tolerance;
    let (seed, count) = offsets
        .iter()
        .map(|&o| (o, offsets.iter().filter(|&&p| near(p, o)).count()))
        .fold(
            (0.0, 0),
            |best, cand| if cand.1 > best.1 { cand } else { best },
        );
    if 2 * count <= offsets.len() {
        return None;
    }
    let inliers: Vec<f64> = offsets
        .iter()
        .filter(|&&p| near(p, seed))
        .map(|&p| wrap_modulo(p - seed, symmetry))
        .collect();
    let folded = seed + inliers.iter().sum::<f64>() / inliers.len() as f64;

    // The multiple of the symmetry: from the board's motion through the two
    // images when it tells, else the one nearest 0 (mounted the same way up).
    let multiples = (TAU / symmetry).round() as usize;
    let pick = |target: f64| {
        (0..multiples)
            .map(|m| wrap(folded + m as f64 * symmetry))
            .min_by(|x, y| wrap(x - target).abs().total_cmp(&wrap(y - target).abs()))
            .expect("a board's symmetry divides a full turn at least twice")
    };
    match motion_turn(pairs) {
        Some(motion) => {
            let turn = pick(motion);
            (wrap(turn - motion).abs() <= symmetry / 4.0).then_some(turn)
        }
        None => Some(pick(0.0)),
    }
}

/// The turn of the board's motion from camera `a`'s image to camera `c`'s:
/// the rotation of a 2D fit between the tracks of the board's centre,
/// radians. `None` when the views are too few, the board moves by less than
/// a square in either image, or the motion is incoherent.
fn motion_turn(pairs: &[(&Fit, &Fit)]) -> Option<f64> {
    if pairs.len() < MIN_MOTION_VIEWS {
        return None;
    }
    let n = pairs.len() as f64;
    let mean = |centres: &mut dyn Iterator<Item = [f64; 2]>| {
        let s = centres.fold([0.0, 0.0], |s, c| [s[0] + c[0], s[1] + c[1]]);
        [s[0] / n, s[1] / n]
    };
    let ma = mean(&mut pairs.iter().map(|(a, _)| a.centre));
    let mc = mean(&mut pairs.iter().map(|(_, c)| c.centre));
    let (mut dot, mut cross, mut magnitude) = (0.0, 0.0, 0.0);
    let (mut spread_a, mut spread_c, mut square_a, mut square_c) = (0.0, 0.0, 0.0, 0.0);
    for (a, c) in pairs {
        let p = [a.centre[0] - ma[0], a.centre[1] - ma[1]];
        let q = [c.centre[0] - mc[0], c.centre[1] - mc[1]];
        dot += p[0] * q[0] + p[1] * q[1];
        cross += p[0] * q[1] - p[1] * q[0];
        let (np, nq) = (p[0].hypot(p[1]), q[0].hypot(q[1]));
        magnitude += np * nq;
        spread_a += np * np;
        spread_c += nq * nq;
        square_a += a.square_px();
        square_c += c.square_px();
    }
    // The centre's RMS spread must reach one square of the board.
    let moves = |spread: f64, square: f64| (spread / n).sqrt() >= square / n;
    let coherent = magnitude > 0.0 && dot.hypot(cross) >= MIN_MOTION_COHERENCE * magnitude;
    (moves(spread_a, square_a) && moves(spread_c, square_c) && coherent).then(|| cross.atan2(dot))
}

/// The affine fit of one whole-board detection: `pixel ≈ step_u · u +
/// step_v · v + offset` over its labels.
#[derive(Debug, Clone, Copy)]
struct Fit {
    extent: BoardExtent,
    /// Labels across (`u`) and down (`v`).
    size: (i64, i64),
    /// Image step of one label along `u`, and along `v`, pixels.
    step_u: [f64; 2],
    step_v: [f64; 2],
    /// Image position of the board's centre, pixels: the same point under
    /// every turn of the labels.
    centre: [f64; 2],
}

impl Fit {
    /// Least-squares fit of the features' pixels on their labels; `None`
    /// when the labels are collinear or the detection is not whole.
    fn new(grid: &ChessboardGrid, features: &[Feature]) -> Option<Self> {
        let extent = grid.extent(features).ok()?;
        let n = features.len() as f64;
        let label = |f: &Feature| {
            let (u, v) = grid.label(f);
            (u as f64, v as f64)
        };
        let (mut mu, mut mv, mut mx, mut my) = (0.0, 0.0, 0.0, 0.0);
        for f in features {
            let (u, v) = label(f);
            mu += u / n;
            mv += v / n;
            mx += f.image_xy[0] / n;
            my += f.image_xy[1] / n;
        }
        let (mut suu, mut suv, mut svv) = (0.0, 0.0, 0.0);
        let (mut sxu, mut sxv, mut syu, mut syv) = (0.0, 0.0, 0.0, 0.0);
        for f in features {
            let (u, v) = label(f);
            let (u, v) = (u - mu, v - mv);
            let (x, y) = (f.image_xy[0] - mx, f.image_xy[1] - my);
            suu += u * u;
            suv += u * v;
            svv += v * v;
            sxu += x * u;
            sxv += x * v;
            syu += y * u;
            syv += y * v;
        }
        let det = suu * svv - suv * suv;
        // Collinear labels.
        if det <= 1e-9 * suu * svv {
            return None;
        }
        let step_u = [(sxu * svv - sxv * suv) / det, (syu * svv - syv * suv) / det];
        let step_v = [(sxv * suu - sxu * suv) / det, (syv * suu - syu * suv) / det];
        // Pixels that are not finite.
        if !step_u.iter().chain(&step_v).all(|x| x.is_finite()) {
            return None;
        }
        let size = match extent {
            BoardExtent::Declared => (i64::from(grid.cols), i64::from(grid.rows)),
            BoardExtent::Turned => (i64::from(grid.rows), i64::from(grid.cols)),
        };
        let (cu, cv) = (
            (size.0 - 1) as f64 / 2.0 - mu,
            (size.1 - 1) as f64 / 2.0 - mv,
        );
        let centre = [
            mx + step_u[0] * cu + step_v[0] * cv,
            my + step_u[1] * cu + step_v[1] * cv,
        ];
        Some(Self {
            extent,
            size,
            step_u,
            step_v,
            centre,
        })
    }

    /// The quarter turns that take the labels onto the declared extent:
    /// even ones for a board seen as declared, odd ones for a turned one,
    /// all four for a square board.
    fn candidates(&self) -> impl Iterator<Item = u8> + '_ {
        let square = self.size.0 == self.size.1;
        let declared = self.extent == BoardExtent::Declared;
        (0..4u8).filter(move |k| square || (k % 2 == 0) == declared)
    }

    /// The image direction of `+u` once the labels are turned by `turn`
    /// quarter turns, radians.
    fn direction(&self, turn: u8) -> f64 {
        let [x, y] = match turn % 4 {
            0 => self.step_u,
            1 => [-self.step_v[0], -self.step_v[1]],
            2 => [-self.step_u[0], -self.step_u[1]],
            _ => self.step_v,
        };
        y.atan2(x)
    }

    /// The turn an anchor with a free frame takes: none for a board seen as
    /// declared, else the candidate whose `+u` runs nearest the image's
    /// `+x` (the first on a tie).
    fn free_turn(&self) -> u8 {
        if self.extent == BoardExtent::Declared {
            return 0;
        }
        self.candidates()
            .map(|k| (k, wrap(self.direction(k)).abs()))
            .min_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)))
            .map_or(0, |(k, _)| k)
    }

    /// The candidate whose direction is nearest `target`, when it is within
    /// `tolerance` and every other candidate is at least twice as far.
    fn nearest_turn(&self, target: f64, tolerance: f64) -> Option<u8> {
        let mut offs: Vec<(u8, f64)> = self
            .candidates()
            .map(|k| (k, wrap(self.direction(k) - target).abs()))
            .collect();
        offs.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        let &(turn, best) = offs.first()?;
        let decisive = offs.get(1).is_none_or(|second| second.1 >= 2.0 * tolerance);
        (best <= tolerance && decisive).then_some(turn)
    }

    /// The mean image length of one square, pixels.
    fn square_px(&self) -> f64 {
        (self.step_u[0].hypot(self.step_u[1]) + self.step_v[0].hypot(self.step_v[1])) / 2.0
    }
}

/// Turn a whole-board detection's labels by `turn` quarter turns onto the
/// declared extent. No turn leaves the features untouched.
fn turn_labels(grid: &ChessboardGrid, fit: &Fit, features: &mut [Feature], turn: u8) {
    if turn == 0 {
        return;
    }
    for f in features.iter_mut() {
        let (mut u, mut v) = grid.label(f);
        let (mut w, mut h) = fit.size;
        for _ in 0..turn {
            // One quarter turn: the new +u runs along the old -v.
            (u, v) = (h - 1 - v, u);
            (w, h) = (h, w);
        }
        debug_assert!((0..w).contains(&u) && (0..h).contains(&v));
        f.world_xyz = [
            u as f64 * grid.square_size_m,
            v as f64 * grid.square_size_m,
            0.0,
        ];
    }
}

/// `angle` wrapped to `(-π, π]`.
fn wrap(angle: f64) -> f64 {
    wrap_modulo(angle, TAU)
}

/// `angle` wrapped to `(-period/2, period/2]`.
fn wrap_modulo(angle: f64, period: f64) -> f64 {
    let r = angle.rem_euclid(period);
    if r > period / 2.0 { r - period } else { r }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    const SQUARE: f64 = 0.03;

    fn grid(rows: u32, cols: u32) -> ChessboardGrid {
        ChessboardGrid {
            rows,
            cols,
            square_size_m: SQUARE,
        }
    }

    /// A camera's image of the board, with each feature's physical corner
    /// `(i, j)` (`i < cols`): the board turned by `angle` in the image about
    /// its centre `centre`, `scale` pixels per square, labelled as the
    /// detector does (`+u` the board axis nearest `+x`, `+v` the one nearest
    /// `+y`, rebased to start at 0).
    fn seen(
        grid: &ChessboardGrid,
        centre: [f64; 2],
        angle: f64,
    ) -> (Vec<Feature>, Vec<(i64, i64)>) {
        let (rows, cols) = (i64::from(grid.rows), i64::from(grid.cols));
        let scale = 30.0;
        let (c, s) = (angle.cos(), angle.sin());
        let step_i = [scale * c, scale * s];
        let step_j = [-scale * s, scale * c];
        let swap = step_i[0].abs() < step_j[0].abs();
        let (step_a, step_b) = if swap {
            (step_j, step_i)
        } else {
            (step_i, step_j)
        };
        let (flip_a, flip_b) = (step_a[0] < 0.0, step_b[1] < 0.0);
        let mut features = Vec::new();
        let mut ids = Vec::new();
        for j in 0..rows {
            for i in 0..cols {
                let (x, y) = (
                    i as f64 - (cols - 1) as f64 / 2.0,
                    j as f64 - (rows - 1) as f64 / 2.0,
                );
                let px = [
                    centre[0] + step_i[0] * x + step_j[0] * y,
                    centre[1] + step_i[1] * x + step_j[1] * y,
                ];
                let (mut a, mut b, amax, bmax) = if swap {
                    (j, i, rows - 1, cols - 1)
                } else {
                    (i, j, cols - 1, rows - 1)
                };
                if flip_a {
                    a = amax - a;
                }
                if flip_b {
                    b = bmax - b;
                }
                features.push(Feature {
                    cov_px: None,
                    image_xy: px,
                    world_xyz: [a as f64 * SQUARE, b as f64 * SQUARE, 0.0],
                });
                ids.push((i, j));
            }
        }
        (features, ids)
    }

    /// A rig of cameras rolled by `rolls` (camera 0 first), seeing the board
    /// turned by `turns[view]` with its centre at `centres[view]` in camera
    /// 0 (camera c sees the centre's motion turned by its roll).
    struct Rig {
        views: Vec<Vec<Option<Vec<Feature>>>>,
        ids: Vec<Vec<Vec<(i64, i64)>>>,
    }

    fn rig(grid: &ChessboardGrid, rolls: &[f64], turns: &[f64], centres: &[[f64; 2]]) -> Rig {
        let mut views = Vec::new();
        let mut ids = Vec::new();
        for (&turn, m) in turns.iter().zip(centres) {
            let (mut cams, mut cam_ids) = (Vec::new(), Vec::new());
            for &roll in rolls {
                let (c, s) = (roll.cos(), roll.sin());
                let centre = [320.0 + c * m[0] - s * m[1], 240.0 + s * m[0] + c * m[1]];
                let (f, i) = seen(grid, centre, turn + roll);
                cams.push(Some(f));
                cam_ids.push(i);
            }
            views.push(cams);
            ids.push(cam_ids);
        }
        Rig { views, ids }
    }

    /// Each physical corner's label, as bits.
    fn corner_labels(features: &[Feature], ids: &[(i64, i64)]) -> BTreeMap<(i64, i64), [u64; 2]> {
        features
            .iter()
            .zip(ids)
            .map(|(f, id)| (*id, [f.world_xyz[0].to_bits(), f.world_xyz[1].to_bits()]))
            .collect()
    }

    /// Whether every kept camera of every view labels each corner as the
    /// view's first kept camera does.
    fn labels_agree(rig: &Rig) -> bool {
        rig.views.iter().zip(&rig.ids).all(|(cams, ids)| {
            let kept: Vec<_> = cams
                .iter()
                .zip(ids)
                .filter_map(|(f, ids)| f.as_deref().map(|f| corner_labels(f, ids)))
                .collect();
            kept.windows(2).all(|w| w[0] == w[1])
        })
    }

    const TURNS: [f64; 8] = [0.1, 1.7, 3.0, -1.4, 0.6, 2.2, -2.6, -0.5];
    const CENTRES: [[f64; 2]; 8] = [
        [0.0, 0.0],
        [60.0, 20.0],
        [-50.0, 40.0],
        [30.0, -60.0],
        [-70.0, -20.0],
        [40.0, 50.0],
        [-20.0, -40.0],
        [70.0, -10.0],
    ];

    #[test]
    fn a_turn_moves_the_direction_with_the_labels() {
        let g = grid(6, 9);
        let (mut features, _) = seen(&g, [320.0, 240.0], 0.4);
        let fit = Fit::new(&g, &features).unwrap();
        for turn in fit.candidates().collect::<Vec<_>>() {
            let mut turned = features.clone();
            turn_labels(&g, &fit, &mut turned, turn);
            let refit = Fit::new(&g, &turned).unwrap();
            assert_eq!(refit.extent, BoardExtent::Declared);
            assert!(wrap(refit.direction(0) - fit.direction(turn)).abs() < 1e-9);
            assert!((0..2).all(|i| (refit.centre[i] - fit.centre[i]).abs() < 1e-9));
        }
        // No turn: the features are untouched.
        let before = features.clone();
        turn_labels(&g, &fit, &mut features, 0);
        assert_eq!(before, features);
    }

    #[test]
    fn cameras_mounted_alike_agree_through_every_turn() {
        let g = grid(6, 9);
        let mut rig = rig(&g, &[0.0, 0.05], &TURNS, &CENTRES);
        let rejected = match_rig_labels(&g, LabelScope::View, &mut rig.views);
        assert!(rejected.is_empty(), "{rejected:?}");
        assert!(labels_agree(&rig));
    }

    #[test]
    fn a_camera_rolled_by_any_quarter_is_matched_from_the_boards_motion() {
        for (rows, cols) in [(6, 9), (7, 7)] {
            let g = grid(rows, cols);
            for roll in [FRAC_PI_2 + 0.1, PI - 0.05, -FRAC_PI_2] {
                let mut rig = rig(&g, &[0.0, roll, 0.02], &TURNS, &CENTRES);
                let rejected = match_rig_labels(&g, LabelScope::View, &mut rig.views);
                assert!(
                    rejected.is_empty(),
                    "{rows}x{cols}, roll {roll}: {rejected:?}"
                );
                assert!(labels_agree(&rig), "{rows}x{cols}, roll {roll}");
            }
        }
    }

    #[test]
    fn without_motion_the_cameras_are_taken_to_be_mounted_alike() {
        let g = grid(6, 9);
        let still = [[0.0, 0.0]; 8];
        let mut alike = rig(&g, &[0.0, 0.1], &TURNS, &still);
        assert!(match_rig_labels(&g, LabelScope::View, &mut alike.views).is_empty());
        assert!(labels_agree(&alike));
        // The assumption is all there is to go on: a camera mounted upside
        // down is then labelled half a turn off.
        let mut upside_down = rig(&g, &[0.0, PI], &TURNS, &still);
        assert!(match_rig_labels(&g, LabelScope::View, &mut upside_down.views).is_empty());
        assert!(!labels_agree(&upside_down));
    }

    #[test]
    fn a_view_whose_turn_disagrees_with_the_others_is_ambiguous() {
        let g = grid(6, 9);
        let mut rig = rig(&g, &[0.0, 0.0], &TURNS, &CENTRES);
        // In view 3, camera 1 sees the board turned by a further 90°, as no
        // rigid rig can.
        let (features, ids) = seen(&g, [350.0, 180.0], TURNS[3] + FRAC_PI_2);
        rig.views[3][1] = Some(features);
        rig.ids[3][1] = ids;
        let rejected = match_rig_labels(&g, LabelScope::View, &mut rig.views);
        assert_eq!(
            rejected,
            vec![(3, 1, RejectReason::AmbiguousChessboardTurn)]
        );
        assert!(rig.views[3][1].is_none());
        assert!(labels_agree(&rig));
    }

    #[test]
    fn a_fixed_frame_is_placed_by_camera_0_or_a_camera_that_stands_in() {
        let g = grid(6, 9);
        // Camera 1 is mounted upside down; the board turns by less than 45°
        // in camera 0, except in view 2, where camera 0 sees it turned.
        let turns = [0.1, -0.3, 1.6, 0.4, -0.2, 0.25, -0.4, 0.0];
        let mut rig = rig(&g, &[0.0, PI], &turns, &CENTRES);
        // Camera 0 misses view 5: camera 1 places the frame there, turned by
        // half a turn into camera 0's labels.
        rig.views[5][0] = None;
        let camera_0_view_0 = corner_labels(rig.views[0][0].as_ref().unwrap(), &rig.ids[0][0]);
        let rejected = match_rig_labels(&g, LabelScope::Fixed, &mut rig.views);
        assert_eq!(
            rejected,
            vec![
                (2, 0, RejectReason::TurnedChessboard),
                (2, 1, RejectReason::TurnedChessboard),
            ]
        );
        assert!(labels_agree(&rig));
        // The board turns by 0.25 rad in view 5 and 0.1 rad in view 0, both
        // within 45° of camera 0's +x: camera 1 gives each corner the label
        // camera 0 gave it in view 0.
        let stood_in = corner_labels(rig.views[5][1].as_ref().unwrap(), &rig.ids[5][1]);
        assert_eq!(stood_in, camera_0_view_0);
    }

    #[test]
    fn a_detection_without_a_direction_is_ambiguous() {
        let g = grid(6, 9);
        let mut rig = rig(&g, &[0.0, 0.0], &TURNS, &CENTRES);
        if let Some(f) = rig.views[1][1].as_mut() {
            f[7].image_xy = [f64::NAN, 0.0];
        }
        let rejected = match_rig_labels(&g, LabelScope::View, &mut rig.views);
        assert_eq!(
            rejected,
            vec![(1, 1, RejectReason::AmbiguousChessboardTurn)]
        );
        // A board one corner wide: its labels are collinear.
        let line = grid(1, 9);
        let (features, _) = seen(&line, [320.0, 240.0], 0.2);
        assert!(Fit::new(&line, &features).is_none());
    }

    #[test]
    fn matching_is_deterministic() {
        let g = grid(6, 9);
        let run = || {
            let mut rig = rig(&g, &[0.0, 2.0, -1.0], &TURNS, &CENTRES);
            let rejected = match_rig_labels(&g, LabelScope::View, &mut rig.views);
            (rig.views, rejected)
        };
        assert_eq!(run(), run());
    }
}

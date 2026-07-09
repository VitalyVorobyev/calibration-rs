# P6-PERCAM-CONVERGENCE — root-cause diagnosis (2026-06-16)

## Context

P6 was scoped from the [profiling report](2026-06-16-perf-from-scratch-rig-profiling.md)
as "from-scratch per-camera Scheimpflug init diverges on the 2 harder
`rtv3d_ref` cameras (cam 3 ~248 px, cam 4 ~52 px)", framed as a seeding problem
for two specific cameras. Its stated purpose is to **unblock measuring the joint
rig/hand-eye BA stages** (P2/P3), which the Phase-3 divergence guard
(`rig_family.rs`, 50 px) stops short of.

Investigation on the private `rtv3d_ref` dataset (6 Scheimpflug cameras, ~−5°
tilt, strong barrel k1≈−0.43, 20 views, ~3,400–4,000 corners/cam) **reframes the
problem** and shows it is **not fixable as a batch tweak**. No solver change was
committed; this report records the diagnosis so the real fix can be scoped.

## What's actually wrong

The oracle is nearly identical across all 6 cameras: fx≈1150–1165, **tauX≈−5°**
(= −0.087 rad, *exactly* a tilt-sweep seed), k1≈−0.43, reproj 0.25–0.30 px. So
cams 3/4 are **not geometrically special**. Yet from a cold start the per-camera
staged Scheimpflug solve lands like this (full run, guard relaxed for
observation):

| cam | recovered fx (oracle) | recovered tauX° (oracle) | recovered k1 (oracle) | reproj px |
|-----|----------------------|--------------------------|-----------------------|-----------|
| 0 | 1107 (1153) | +0.3° (−5.2°) | −0.16 (−0.44) | 1.74 |
| 1 | 1146 (1153) | −9.0° (−4.8°) | −0.24 (−0.43) | 8.15 |
| 2 | 1117 (1160) | −0.7° (−4.9°) | −0.20 (−0.44) | 1.56 |
| 3 | **0.2** (1158) | **−270°** (−5.1°) | **1e14** (−0.43) | **248** |
| 4 | 1987 (1151) | +81° (−5.0°) | +5.5 (−0.43) | 52 |
| 5 | 1165 (1166) | −2.6° (−4.9°) | −0.32 (−0.42) | 1.47 |

**No camera recovers the oracle tilt (−5°).** The "passing" cams (0/2/5) sit in
a *wrong* basin (tilt ≈ 0, weak distortion) that reaches a low-ish 1.5 px and
stays under the guard; cams 3/4 run away (`fx→0` / `fx→1987`, `k1→1e14`).

### Root cause: a degenerate seed the LM can't escape

1. **The per-camera linear init is a *pinhole* fit** (`estimate_intrinsics_iterative`)
   that ignores tilt. On a tilted sensor the tilt signal is mis-attributed to
   distortion/focal, so the seed has an **underestimated focal** (fx≈944 vs 1153,
   a ~1.22× gap) and weak distortion (cam 3 additionally falls back to the
   distortion-free iteration-0 estimate).
2. **The tilt↔distortion↔focal degeneracy** then creates many local minima. From
   the low-focal seed the LM settles into a wrong-tilt valley (1.5 px) far from
   the oracle (0.27 px), or runs to the `fx→0` degenerate limit.

## Why the quick fixes don't work (all tried, all reverted)

- **The tilt sweep is a no-op.** All 5 tilt seeds (−10°…+10°) converge to the
  *same* result per camera — the seed tilt washes out in the first iterations.
  The basin is set by the (fx, k1) seed, not the tilt seed.
- **Box bounds on the sweep / stage-2** (carrying stage-3's bounds upstream)
  prevent the numeric runaway but pin the bad cameras at the **bound edge**
  (tilt = ±17°, fx = 0.5× seed) with huge reproj, and the bounded full solve then
  fails to converge — trading a guard-catchable divergence for a hard solver
  error.
- **A focal-scale sweep** (seed fx ×[1.0, 1.15, 1.3], the natural fix for the
  underestimate) brings cam 4 under the 50 px guard — **but with garbage params**
  (fx 399, tilt at the +17° bound): an *illusory* pass. It does not move any
  camera toward the oracle, makes cam 1 worse, and leaves cam 3 at 2750 px.

All three just relocate the wrong local minimum. The synthetic
`staged_init_recovers_large_tilt_from_cold_start` test passes throughout — it
has **no distortion**, so it never exercises the tilt↔distortion degeneracy that
defeats the real data.

## The real fix (deferred — research-grade)

Reaching the oracle basin from scratch needs a **tilt-aware initialization**, not
a better local search from the pinhole seed. Candidate directions:

- Estimate sensor tilt up front from the per-view homography structure (the tilt
  induces a view-consistent projective signature distinct from radial
  distortion), then seed the focal/distortion with the tilt removed.
- Joint focal + tilt + distortion coarse search (a real 2-D+ basin search, not
  the current no-op tilt sweep), ranked by **full-convergence** reproj, not a
  short subsampled solve.
- A global/multi-start optimizer for the per-camera Scheimpflug block, or a
  continuation that anneals distortion in while holding focal/tilt.

This is a focused numerical-research effort, out of scope for the Track P batch.

## Corroborating evidence: a synthetic regression from the staged init

The full-workspace test gate surfaced a **pre-existing** failure (independent of
this batch — it fails at the branch base `d7139c7`):
`vision-calibration` → `tests/scheimpflug_intrinsics.rs::public_api_converges_on_synthetic_scheimpflug_dataset`
asserts `|tilt_x − 0.01| < 0.01` on a **small-tilt synthetic** dataset and now
fails. `optimize_scheimpflug_intrinsics_staged` is **absent on `main`** (it was
added by `a9a97ee` on this validate branch), so the staged init that was meant to
*fix* Scheimpflug convergence in fact **regressed** an easy small-tilt synthetic
case — not just the hard real rtv3d data. This is the same root cause: the
tilt-sweep + reproj-argmin staging does not reliably land the true tilt basin.
It reinforces that the staged-init approach needs the rethink described above,
and should be addressed together with this item before the validate branch
merges. (Not caused by this batch; flagged for the owner.)

## Status

**P6 stays OPEN, re-characterized.** It is not a cam-3/4 seeding bug but a
from-scratch initialization problem affecting *all* cameras on strong-distortion
+ tilt data; the guard correctly rejects the worst offenders. P2/P3 (joint-BA
cost) can still be measured independently of P6 by seeding good per-camera
intrinsics (e.g. the validated frozen path the V5 bench already uses) rather than
solving from scratch — the from-scratch convergence is a separate robustness
goal. No production code changed; the validated staged solver is untouched.

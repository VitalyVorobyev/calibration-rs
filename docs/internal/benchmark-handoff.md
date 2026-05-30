# Benchmark work — RESUME HERE (handoff)

_Written 2026-05-30 at ~97% context. Companion to the approved plan
`this-workspace-has-significan-luminous-orbit.md` (same dir)._

## One-line state
Building `vision-calibration-bench` (two-tier accuracy/stability benchmark over the
6 datasets). **Phase 0 done & verified. Phase 1: one slice verified, four runners
ON DISK but UNVERIFIED** (an over-running subagent wrote them, then was killed before
I confirmed the build/numbers). Phase 3 not started.

## ✅ CONFIRMED good (verified via returned results this session)
- New crate `crates/vision-calibration-bench` (workspace member, `publish=false`).
  Features: `tier-a` (default, math-only, NO image/detection deps) / `tier-b`
  (calib-targets+image+glob, optional) / `laser` (stub). Workspace builds, clippy +
  `cargo test --workspace` were GREEN after Phase 0b.
- **Custom subagents exist on disk** → real agent types in any NEW session:
  `.claude/agents/quick-implementer.md` (sonnet, mechanical) and
  `.claude/agents/deep-implementer.md` (opus, algorithmic). USE THESE next time.
  (This session they weren't in the loaded registry, so I faked them with
  general-purpose + model override — that's why the agents lacked the built-in
  "stop early" guardrails. Don't repeat that; the real agent files have the rules.)
- Phase 0b schemas (`src/{record,registry,fixtures,compare}.rs`): `BenchRecord`
  (+Convergence/Fit/Generalization/Stability/Detection/LaserMetrics/DeltaToPrior/
  Timing), `BenchRegistry`/`BenchEntry`/`SpecRef` (wraps `DatasetSpec`),
  `FrozenFixture`/`FrozenPayload` (7 variants), `Tolerances`. Reuses core
  `ReprojectionStats`/`FeatureResidualHistogram`/`SolveReport`. Poses=`[f64;7]`
  `[qx,qy,qz,qw,tx,ty,tz]`. Records lack `PartialEq` (upstream types do too) →
  roundtrip tests assert JSON-equality. `BENCH_SCHEMA_VERSION=1`.
- Registry is **JSON** (`registry/public.json`), NOT TOML (no `toml` crate locked).
  Loader `registry::load_registry` + `BenchRegistry::find`.
- **Phase 1 vertical slice VERIFIED**: `calib-bench run --dataset stereo_left`
  reproduces `planar_real.rs` bit-exactly → **0.23388 px**, 20/20 imgs, converged.
  Path detect→Input→run_calibration→BenchRecord is trustworthy.

## ⚠️ ON DISK BUT UNVERIFIED — verify FIRST next session
The killed agent wrote registry entries + runners for `stereo_right` (Planar),
`stereo_rig` + `stereo_charuco` (RigExtrinsics), `kuka_1` (SingleCamHandeye).
`run.rs` should contain `run_planar_intrinsics`/`run_rig_extrinsics`/
`run_single_cam_handeye` + a `detect_charuco_view`. The agent self-reported matching
numbers (stereo_right 0.21947, stereo_rig 0.20490 / [0.197,0.213], kuka_1 0.04162,
stereo_charuco 0.12642 / [0.123,0.130]) BUT I never confirmed a clean build, and the
~957-tool-call thrash may have left duplicate/half-written helpers. **Trust nothing
here until the commands below pass.**

### FIRST STEPS on resume (cheap — do before anything else)
```bash
cd /Users/vitalyvorobyev/vision/calibration-rs
cargo build -p vision-calibration-bench --features tier-b --offline           # must be clean
cargo clippy -p vision-calibration-bench --all-targets --all-features --offline -- -D warnings
# reproduce each example number (the acceptance test) — ONE run each, parse JSON:
for d in stereo_left stereo_right stereo_rig kuka_1 stereo_charuco; do
  cargo run -p vision-calibration-bench --features tier-b --offline --bin calib-bench -- run --dataset $d 2>/dev/null \
   | python3 -c "import json,sys;r=json.load(sys.stdin);print('$d',round(r['fit']['reported_mean_reproj_px'],5),[round(x,5) for x in (r['fit'].get('reported_per_cam_px') or [])],r['convergence']['converged'])"
done
# reference oracles to match within ~1% (only if a bench number looks off):
cargo run --example stereo_session --offline 2>/dev/null | tail -15
cargo run --example handeye_session --offline 2>/dev/null | tail -15
cargo run --example stereo_charuco_session --offline 2>/dev/null | tail -15
```
Match → keep. Mismatch → real finding OR thrash bug; inspect that runner in `run.rs`.
If `run.rs` is a mess, revert just that file and re-wire the 4 with a TIGHTLY-SCOPED
quick-implementer (one job, build once, ≤40 tool calls, report partial if exceeded).

## 🔬 Findings so far (preserve — these ARE the deliverable)
- **F1 (low):** stereo board declared 7×11 in `data/stereo/dataset_*.toml` but
  `calib-targets` auto-detects ~10×11; ~30% of corners dropped by the residual
  filter. Reproj grid-invariant for pure intrinsics → headline unaffected, but the
  manifest/detector contract is loose + coverage accounting muddied.
- **F2 (medium, MOST INTERESTING):** `kuka_1` reproj suspiciously tight (~0.04 px,
  5× better than planar) because the 30 robot poses are near-pure-translation
  (rotations ~1e-5 rad) → hand-eye ROTATION barely constrained; great reproj hides an
  ill-determined extrinsic. Exactly the "low error masks instability" case the
  benchmark exists to catch. Phase-3 stability should flag `he.rot_deg`.
- **F3 (blocks DS8):** `data/DS8/calibration_object.txt` self-inconsistent
  (`rows 6 cols 7 dimension 0.1` but 90+ coplanar points). Heterogeneous rig (2× JAI
  BB-500GE + Creative Senz3D + Denso arm), 14 chessboard views/cam under
  `images/cameraN/calib/` + `laser/`. No reference example → must reverse-engineer.

## ⏭️ Next phases
- **Phase 2** = mostly verifying the 4 above. DS8 PARKED (see open decisions).
  Private datasets (3536119669, 130x130_puzzle) need laser extractor +
  Scheimpflug/manual-seeds — real new infra, later.
- **Phase 3 (the core, your real worry) — NOT on disk yet.** Was about to be built by
  2 agents when interrupted. Order:
  1. Split detect-once / solve-many in `run.rs`: `detect_*_all(entry)->Detected*`
     (once) + `solve_*(det, view_indices, cfg)->Export` (pure, subset, errors below
     problem minimums) + `build_record_*`. Behavior-preserving (numbers must not move;
     add a regression test asserting stereo_left ≈0.23388).
  2. `src/params.rs`: flatten Export→`BTreeMap<String,f64>` incl. **rig
     baseline/extrinsics (camK.baseline_m, tx/ty/tz, rot_deg, axis-angle) and hand-eye
     axis-angle** — you emphasized the rig/baseline is the key output.
  3. `src/stability.rs` + `calib-bench stability --dataset <id>`: seed=base_seed+i,
     sample floor(subset_frac*n) views w/o replacement, re-solve, per-param
     {mean,std,min,max,cv}, multimodality flag (largest gap > 0.5*range w/ ≥2 each
     side & range/|mean|>0.02) = local-minima detector. Interpret kuka_1 he.rot_deg,
     stereo_rig baseline/rot stability.
  4. `src/crossval.rs`: held-out pose-holdout CV.
- Phases 4 (diagnose & fix), 5 (freeze Tier-A fixtures + `tests/tier_a.rs` CI gate +
  tolerances + `bench_tier_b.sh`), 6 (report + ADR 0015 + docs) unchanged.

## ❓ OPEN DECISIONS (I asked; you hadn't answered — decide on resume)
1. **Next focus:** (a) Depth — stability+CV on the 5 working datasets now
   (RECOMMENDED; attacks core worry, no new infra) · (b) Breadth — wire hard/private
   datasets first · (c) interleave.
2. **DS8:** (a) you supply real board geometry/camera roles · (b) I reverse-engineer
   w/ "no oracle" caveat · (c) defer · (d) drop from benchmark.
3. **Fix mode:** (a) catalogue all problems, fix in a batch (RECOMMENDED) · (b) fix
   as found.

## 🧰 Guardrails (why the over-run happened — don't repeat)
- One agent hit ~957 tool calls (≈940k tokens) on a 4-dataset wiring job. Likely:
  repeated full `cargo run --example`/rebuilds under `--offline` + per-step opus
  reasoning, no early-stop cap (because it was general-purpose, not the real
  deep-implementer with the rules).
- NEXT TIME: real agent types now exist → use `quick-implementer`/`deep-implementer`.
  Give each ONE tight job; "build ONCE, verify via the python one-liner not repeated
  cargo run; if you exceed ~40 tool calls STOP and report partial." NEVER batch a long
  agent with dependent agents or an approval question in one message (killing one
  killed the batch this session).
- ENV: cargo offline → always `--offline`. Available: anyhow, calib-targets, image,
  glob, nalgebra, serde, serde_json, clap, rand. NOT available: `toml`, `chrono`
  (registry=JSON; timestamps=unix-epoch String). Never read/commit `privatedata/`.

## Phase map
0a scaffold ✓ · 0b schemas ✓ · 1 slice ✓ + 4 runners UNVERIFIED ·
2 baseline partial (DS8 parked) · 3 stability+CV NOT STARTED · 4 fix · 5 gate · 6 docs.

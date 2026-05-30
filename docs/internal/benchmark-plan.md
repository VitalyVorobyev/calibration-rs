# Data-Driven Performance & Stability Benchmark for calibration-rs

## Context

The workspace holds ~24k LoC of advanced calibration algorithms (7 problem types,
8 with Scheimpflug variants), but there is **no data-based confidence in its
accuracy or numerical stability**. Quality today is "the real-data examples print
a plausible reprojection error to stdout" — ad-hoc, not machine-readable, not
gated, and silent about local minima or overfitting. The user wants to:

1. Build a reproducible **performance/accuracy benchmark** over all 6 available
   datasets (4 public in `data/`, 2 private/challenging in `privatedata/`).
2. **Find the problems it exposes** (local minima, excessive reprojection error,
   init failures, detector bugs) and fix them — algorithms, APIs, pipelines, and
   even the sibling crates are all in scope.
3. End in a state of **earned confidence**: known accuracy per dataset, a
   catalogued list of problems + mitigations, and the benchmark wired in as a
   **quality gate** so future changes can't silently regress.

### What exploration established (decisive constraints)

- **Only `data/kuka_1` has images in git.** `data/stereo` commits only its two
  `dataset_*.toml`; `data/DS8` + `data/stereo_charuco` are gitignored; both
  `privatedata/` sets are sealed by `privatedata/.gitignore` (`*`). A CI gate that
  needs images can see exactly one dataset → **two-tier design is forced.**
- **Detection is fragmented.** `vision-calibration-detect` exposes a *sealed*
  `Detector` trait with **only `ChessboardDetector`**; `dataset_runner::pick_detector`
  (`dataset_runner.rs:312`) knows only `"chessboard"` and explicitly rejects
  ChArUco/Puzzleboard. ChArUco/laser/puzzle detection lives in the sibling crates
  `calib-targets` (`/Users/vitalyvorobyev/vision/calib-targets-rs`) and
  `vision-metrology` (`/Users/vitalyvorobyev/vision/vision-metrology`), called
  directly by the examples.
- **No ground truth** — only prior computed exports (`data/stereo/viewer_export.json`,
  `data/stereo_charuco/calibration.json`, private `export.json`). Reprojection
  error alone hides overfitting and local minima → honest metrics need held-out
  cross-validation + cross-seed stability.
- **Metrics infra already exists and must be reused** (not reinvented):
  `ReprojectionStats` (`core/src/types/observation.rs:190`, `from_errors` at `:203`),
  `TargetFeatureResidual`/`LaserFeatureResidual`/`FeatureResidualHistogram`
  (`core/src/types/residual.rs:44-100`), `SolveReport`, and every `*Export` carries
  `mean_reproj_error` + `per_cam_reproj_errors` + `per_feature_residuals` +
  `image_manifest`. The committed `linear/tests/data/stereo_linear.json` is the
  precedent for "freeze correspondences to JSON".
- **The real-data examples are the de-facto benchmark** and their detection I/O is
  **duplicated** across `vision-calibration/examples/support/{stereo,handeye,
  stereo_charuco,rig_handeye}_io.rs` (with drift — `rig_handeye_io.rs` uses identity
  poses + has an `EyeToHand` typo path, i.e. it is *not* working DS8 wiring). The
  harness **refactors these into a reusable lib**, it does not duplicate them.

## Locked decisions (from the interview)

| # | Decision | Choice |
|---|----------|--------|
| 1 | **Gate model** | **Two-tier.** Tier A = small committed frozen detection+pose fixtures → init+optimize math regression-gated in CI (no images). Tier B = full image→detection→calibration, local/self-hosted, rich report. **Private fixtures stay gitignored.** |
| 2 | **Correctness (no GT)** | **Honest suite:** reprojection (mean/RMS/max + histogram) + held-out cross-validation + cross-seed/subset **parameter stability w/ multi-modality detection** + delta-to-prior-export where one exists. |
| 3 | **Detection** | **Full pipeline, in scope.** Targets **always via `calib-targets`**; laser via **`vision-metrology`**. Bugs found in those sibling repos are in scope (separate fix PRs). Track detection count/coverage. |
| 4 | **Speed** | **Quality gated; runtime recorded, not gated** (CI timing is noisy). |

Pre-1.0: breaking API changes are acceptable when a benchmark finding justifies them.

## Architecture — `vision-calibration-bench`

A new **unpublished** workspace crate (`publish = false`) = library + one CLI binary
`calib-bench` (`run | report | compare | freeze-fixtures | list`). Rationale: Tier A
must run under `cargo test --workspace` (the existing CI gate, `ci.yml:34`), which a
library crate + integration test gives for free; an xtask cannot. Private wiring
never enters the public crate.

**Feature gating** keeps CI image-free and private-free:
- `tier-a` (default): serde record/registry/compare/fixtures; **math only**, reads
  committed fixtures, no images, no detection deps → CI-safe on all 3 OSes.
- `tier-b`: detection from images via `calib-targets` + `image` (public datasets).
- `laser`: laser extraction; the **public crate never names `vision-metrology`** —
  it defines a `LaserExtractor` trait whose impl is supplied by a **gitignored
  sibling crate** (`vision-calibration-bench-private`, own workspace like
  examples-private) linked only locally. This keeps the `*` path-dep and the
  private detection out of CI's `--all-features` clippy step.

**Layout:**
```
crates/vision-calibration-bench/
  src/{lib,registry,record,run,detect,stability,crossval,params,compare,fixtures}.rs
  src/bin/calib_bench.rs
  registry/public.toml                  # committed
  registry/tolerances.toml              # committed
  registry/golden/<id>__tier{A,B}.json  # committed (public)
  fixtures/<id>__<problem>.json         # committed Tier-A frozen IR (public)
  tests/tier_a.rs                       # CI #[test] per public dataset
privatedata/bench/{registry.overlay.toml,golden/,fixtures/}   # gitignored (private)
```

**Key schema rules** (from grounding):
- `DatasetSpec` is `#[serde(deny_unknown_fields)]` (`dataset/src/spec.rs:23,80,212,288`)
  → the registry **wraps** a spec (`BenchEntry { … spec: SpecRef::{Inline|Path} … }`),
  it does not add keys inside one. `data/stereo/dataset_left.toml` is already a valid
  spec the registry can point at.
- `BenchRecord` **reuses** `ReprojectionStats`, `FeatureResidualHistogram`, `SolveReport`
  verbatim; `fit.overall` is built by feeding `export.per_feature_residuals` `error_px`
  into `ReprojectionStats::from_errors` (same path the diagnose UI uses), and keeps
  `export.mean_reproj_error` alongside as a cross-check.
- `ident.{git_sha,timestamp}` are **injected externally** so a record is otherwise a
  pure function of (fixture, config, code) → byte-stable golden comparison.
- **Detector path:** the bench calls `calib-targets`/`vision-metrology` **directly**
  in `src/detect.rs` (as the examples do) — do **not** block on extending the sealed
  pipeline `Detector`. Extending `vision-calibration-detect` with a `CharucoDetector`
  (so the Tauri `run.rs` gains ChArUco too) is a **non-blocking follow-up**.

## Metric taxonomy (the "honest suite")

Per `(dataset, tier, run)` → one `BenchRecord` JSON:
- **Convergence:** `init_ok`, `converged`, `SolveReport{iterations, final_cost}`.
- **Fit:** overall + per-camera `ReprojectionStats` (mean/RMS/max px) + per-camera
  `FeatureResidualHistogram` (≤1/≤2/≤5/≤10/>10 px).
- **Generalization:** held-out cross-validation — train on a view subset, measure
  reprojection on held-out views (k folds). *Pose-holdout, not parameter-holdout for
  hand-eye/laser* (held-out view still shares global params) — documented as such.
- **Stability:** N runs over random view subsets/seeds → per-parameter spread
  `{mean,std,min,max,cv}` for focal/principal-point/distortion/rig-baseline/extrinsics/
  handeye/laser-plane; **multi-modality flag** = local-minimum detector (the user's
  core worry). A *new* multimodal flag vs golden is a hard regression.
- **Detection:** per-camera detected vs expected feature count, coverage %, detect ms.
- **Laser:** point-to-plane residual (m + px) + inlier ratio (so "great RMS on 3
  inliers" is visible).
- **Delta-to-prior:** per-parameter abs/rel delta vs prior export where present.
- **Timing:** init/optimize/total/detection ms — **recorded, never gated.**

**Gate:** `compare` flags a regression only when **both** abs and rel tolerances are
exceeded in the *worse* direction (`tolerances.toml`, per-class + per-dataset
overrides). Reproj/CV/laser tight; stability loose (guards against new local minima,
not exact spread); timing never compared. Exit 1 on regression; `--update-golden` is
manual + human-reviewed, never in CI.

## Dataset → problem → detector wiring

| id | vis | ProblemKind | detector (calib-targets) | laser (vision-metrology) | poses | tiers | prior |
|---|---|---|---|---|---|---|---|
| stereo | public | RigExtrinsics (+2× Planar) | chessboard 7×11 @30mm | — | — | A,B | viewer_export.json |
| kuka_1 | public (imgs in git) | SingleCamHandeye | chessboard @ squaresize.txt | — | RobotPosesVec.txt (30) | A,B | — |
| **DS8** | public (local) | RigExtrinsics→RigHandeye→RigLaserlineDevice | chessboard 10×14 @52mm ×3 **heterogeneous** cams | RANSAC plane ×cams | robot_cali.txt (42, Denso) | B | — |
| stereo_charuco | public (local) | RigExtrinsics | charuco 22×22 5.2mm 4x4_1000 | — | — | B | calibration.json |
| 3536119669 | private | LaserlineDevice (+handeye) | charuco 22×22 + laser | RANSAC (config.json) | poses.json (Tsai) | B | config.json |
| **130x130_puzzle** | private | RigHandeye(Scheimpflug)→RigLaserlineDevice | puzzleboard 1.014mm ×6 tiles | 6 planes RANSAC | poses.json | B | export.json |

Hardest two (bold) are private/local-only and never become committed Tier-A fixtures.

## Subagents to create (Phase 0, step 1)

**`.claude/agents/quick-implementer.md`** — model `sonnet`. For mechanical, fully
specified work: crate scaffolding, serde structs, CLI/registry/TOML plumbing, moving
code between modules, fixtures, doc edits, applying a precise diff. System prompt:
follow `CLAUDE.md` + `AGENTS.md` + ADRs; keep changes minimal and to-spec, **do not
redesign**; run `cargo fmt --all`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, `cargo test --workspace` before declaring done;
report exactly what changed + gate results.

**`.claude/agents/deep-implementer.md`** — model `opus`. For non-trivial work:
algorithmic changes (init strategy, local-minima mitigation, stability/CV math,
detector/laser integration), cross-crate refactors, root-causing numerical problems,
changes spanning the sibling repos (`calib-targets-rs`, `vision-metrology` — separate
git repos → separate commits). System prompt: reason hard about numerical correctness
and **determinism** (`AGENTS.md:95-98`); honor ADR conventions (`frame_se3_frame`, SE3
`[qx,qy,qz,qw,tx,ty,tz]`, generic `fn residual<T: RealField>`); test with synthetic
ground truth (tight tol) + real-data (loose); state root cause → approach →
verification; run all quality gates.

Both: `tools` = full implementation set (Read, Edit, Write, Bash, Grep, Glob).
Verification of their output goes through the project's `calibration-review` /
`algo-review` skills and `/gate-check`.

## Phased execution

Each phase names the driver subagent, the deliverable, the gate, and a **user
checkpoint** where the user asked to "pay full attention to problems found."

**Phase 0 — Scaffold + subagents** *(quick-implementer)*
- Create the two subagent definitions above.
- Create `vision-calibration-bench` crate (features, deps, binary), add to workspace
  `members`. Reconcile the **calib-targets 0.8 (examples-private) vs 0.9 (workspace)**
  skew against the sibling — pin one.
- Define `BenchRecord`, `BenchRegistry`/`BenchEntry`/`SpecRef`, `Tolerances`,
  `FrozenFixture`/`FrozenPayload` (serde; reuse core stats types). Stub CLI subcommands.
- **Gate:** workspace builds; fmt/clippy/test green on the skeleton.

**Phase 1 — Detection consolidation + Tier-B runner** *(deep for `detect.rs`/laser, quick for plumbing)*
- Move the load+detect bodies of `support/{stereo,handeye,stereo_charuco,rig_handeye}_io.rs`
  into `bench::detect` (chessboard + charuco via calib-targets; laser via the
  `LaserExtractor` trait + gitignored private impl). **Port the examples onto the lib;
  delete the duplication** (fix the `rig_handeye_io` typo/identity-pose drift en route).
- Implement shared `run_problem(entry, config) -> (Export, RunStats)` (the Tauri
  `run.rs` should later sit on the same helper — keep field names aligned).
- Wire `registry/public.toml` + the gitignored private overlay; `calib-bench run --tier B`
  works for the **easy** datasets (stereo, kuka_1, stereo_charuco).
- **Gate:** examples still run + print same errors; bench reproduces them.

**Phase 2 — Wire all 6 datasets + FIRST baseline + surface problems** *(deep-implementer)*
- Wire the hard datasets: DS8 (start RigExtrinsics-only with heterogeneous/color-cam
  handling + real `robot_cali.txt`, then layer handeye, then laser), 3536119669
  (canonical loader for the messy duplicate-timestamped jsons; LaserlineDevice+handeye),
  130x130_puzzle (Scheimpflug RigHandeye **with manual init seeds in the registry** →
  RigLaserlineDevice).
- Run all 6 → emit `BenchRecord`s; eyeball reproj/convergence/detection.
- **Expected problems:** DS8 heterogeneous init, Scheimpflug `cy`/tilt degeneracy,
  laser plane sign/orientation local minima, detection gaps/bugs (→ calib-targets /
  vision-metrology), private-json messiness.
- **★ CHECKPOINT:** present the first baseline table + catalogued problem list; agree
  fix priorities before diving in.

**Phase 3 — Honest metrics: cross-validation + stability** *(deep-implementer)*
- Implement held-out CV (pose-holdout) + cross-seed/subset stability with the
  multi-modality detector. Resampler floors subset size to each problem's minimum
  (Planar ≥3 views; RigExtrinsics ≥3 views & ≥2 cams) and skips seeds that underfill a
  camera. Seeds fixed in the registry (determinism).
- Run across datasets → **quantify local minima** (the core worry).
- **★ CHECKPOINT:** review stability/CV; triage local-minima findings.

**Phase 4 — Diagnose & fix** *(deep-implementer; iterative, per-problem)*
- Root-cause → fix → re-run → confirm improvement, for each confirmed problem. Fixes
  may land in this repo (init/optim/pipeline/API) **or** in `calib-targets-rs` /
  `vision-metrology` (separate PRs). Review each via `calibration-review`/`algo-review`.
- **★ CHECKPOINT per fix** (or per cluster) — the user wants to weigh in on algorithm/
  API changes.

**Phase 5 — Lock baselines + build the gate** *(quick for plumbing, deep for tolerances)*
- `freeze-fixtures` the public Tier-A datasets (refuses to write private outside
  `privatedata/`); commit fixtures + golden records + `tolerances.toml`.
- Implement `compare` + `tests/tier_a.rs` (one `#[test]` per public dataset) + a
  determinism double-run test (record-minus-`ident` must be byte-stable).
- Wire Tier A into CI (one line in the existing `cargo test` job — no new job).
- Add committed `scripts/bench_tier_b.sh` for the local/self-hosted full run.
- **Gate:** CI green; perturbing a param trips the gate; Tier-B script runs locally.

**Phase 6 — Document & institutionalize** *(quick-implementer + docs)*
- Benchmark report (per-dataset results: reproj/CV/stability/deltas + known issues &
  mitigations). ADR (e.g. `0015-performance-benchmark.md`). Update `CLAUDE.md` Quality
  Gates + `docs/ROADMAP.md` + a tutorial. Update memory.

## Risks / sharp edges (condensed, ranked)

1. **Private data leaking into git** — overlay-only private entries; `freeze-fixtures`
   refuses private writes outside `privatedata/`; committed test scans `registry/` +
   `fixtures/` for `privatedata`/known private ids; public crate never names
   `vision-metrology`.
2. **DS8 heterogeneous cams + laser** — no shared image size; color cam → `to_luma8`;
   `rig_handeye_io.rs` is a non-working stub (identity poses + typo); wire real poses.
3. **Scheimpflug puzzle degeneracy** — needs manual intrinsic seeds in the registry or
   it won't reproduce; stability will (correctly) show `cy`/tilt multi-modality.
4. **Laser local minima** — plane normal sign/orientation bimodal; run multimodal
   detector on plane params; report inlier ratio.
5. **Detection fragmentation / calib-targets version skew (0.8 vs 0.9)** — reconcile
   API before coding; consolidate the duplicated `*_io.rs`.
6. **Cross-repo fixes** — record coverage + inlier ratio precisely so a sibling
   regression is attributable; pin sibling versions.
7. **CV/stability cost** — Tier-B only; per-entry fold/run counts; CI Tier-A uses
   `n_runs=4` or multimodal-flag-only.
8. **Record purity / determinism** — inject `git_sha`/`timestamp`; double-run diff test.

## Verification (end-to-end)

- **Builds/gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`; `cargo test --workspace` (now includes
  `tests/tier_a.rs`). The bench crate's default (`tier-a`) features compile with **no
  images and no detection deps**.
- **Tier A correctness:** `cargo test -p vision-calibration-bench` reproduces golden
  records from committed fixtures; manually perturb a golden value → the test fails
  (gate proven).
- **Tier B locally:** `bash crates/vision-calibration-bench/scripts/bench_tier_b.sh`
  runs all 6 (incl. private via the overlay) → `BenchRecord`s + HTML/markdown report;
  cross-check bench reproj vs the existing examples' stdout (must match pre-fix).
- **Determinism:** double-run each Tier-A fixture; records (minus `ident`) are byte-identical.
- **Privacy:** `git status` after a full local run shows **nothing** under
  `privatedata/` staged; the scan test passes.

## Execution discipline (context budget)

The orchestrator (me) stays lean: **delegate** reads to `Explore` and implementation
to `quick-`/`deep-implementer`; keep only conclusions, file:line refs, and the latest
`BenchRecord` summaries in context. Checkpoint at the ★ points rather than running the
whole arc unattended — problem triage (Phases 2–4) is collaborative by design.

## Out of scope / notes

- Extending the sealed pipeline `Detector` with ChArUco/puzzle (nice follow-up so the
  Tauri runner gains them; not required for the bench).
- Self-hosted CI for Tier B (the local script suffices initially).
- Fixes inside `calib-targets-rs` / `vision-metrology` are separate-repo commits.

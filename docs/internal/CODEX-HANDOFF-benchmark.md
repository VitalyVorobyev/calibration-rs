# Codex handoff — vision-calibration-bench (multi-level report + dataset wiring)

Written 2026-05-31. Branch `feat/bench-multilevel-report`, **PR #49** (open).
HEAD = `bd8d257`, pushed, working tree clean. Pick up from here.

---

## 1. What this work is

`vision-calibration-bench` is a benchmark harness for the calibration-rs
workspace. The headline deliverable is a **hierarchical multi-level reprojection
report**: instead of one reproj number per dataset, it re-evaluates the same
detected corners at successive constraint levels and aggregates per
camera / view / corner, so you can localize *where* error comes from
(feature detection vs camera model vs rig extrinsics vs robot/hand-eye chain).

- Levels: `Intrinsic` (free per-view PnP floor) → `RigExtrinsic` → `HandEye`
  (→ `Laser`, not yet wired). Each carries `LevelStats{mean,median,rms,p95,max,
  count}` overall + per-camera + per-view, plus the full `Vec<TargetFeatureResidual>`.
- A single `headline_px` = the most-constrained level's mean, which **must equal**
  `export.mean_reproj_error` (regression invariant — assert this for every dataset).
- Library module: `vision-calibration-pipeline::analysis` (re-exported from the
  facade). Builders: `planar_intrinsics_report`, `single_cam_handeye_report`,
  `rig_extrinsics_report`, `rig_handeye_report`.

## 2. What is DONE (committed on this branch)

- `b3c087e` D2: the `analysis` module + multi-level report + the bench crate;
  `BenchRecord.reproj_report: Option<ReprojReport>`, `BENCH_SCHEMA_VERSION=2`.
- `b38910a` W1: `run_rig_handeye` + DS8 (3-cam pinhole rig hand-eye, index-paired).
- `91bff75` W3: closed the stereo_charuco oracle (doc only:
  `docs/internal/w3-oracle-stereo-charuco.md`).
- `bd8d257` W2b: private 6-camera ChArUco rig hand-eye (`charuco_handeye_3536`),
  tiled-frame support, `snap_list_json` pose format, `BoardGeometry.marker_size_rel`.

### Verified diagnostic results (real, from actual runs)
| dataset | intrinsic floor | most-constrained | reading |
|---|---|---|---|
| kuka_1 | 0.157 px | 1.193 px (hand-eye) | chain-limited |
| ds8 | 0.186 px | 4.86 px (hand-eye) | chain-limited |
| charuco_handeye_3536 | 1.05 px (per-cam 0.82–1.40) | 10.53 px (hand-eye) | chain-limited |
| stereo_charuco | 0.40 px | 0.551 px (rig-extr) | healthy |

**Frozen public baselines (must never move):** stereo_left 0.23388,
stereo_right 0.24415, stereo_rig 0.25031, kuka_1 1.19335, stereo_charuco 0.55087.

## 3. Repo / tooling facts you need

- **cargo is OFFLINE** — every cargo command needs `--offline`. macOS has **no
  `timeout`**.
- Build/test only the bench crate with tier-b:
  `cargo build -p vision-calibration-bench --features tier-b --offline`.
- Gates before any commit:
  `cargo fmt -p vision-calibration-bench -- --check` and
  `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`.
- Run a dataset:
  `cargo run -q -p vision-calibration-bench --features tier-b --offline --bin calib-bench -- run --dataset <id>`
  (add `--registry crates/vision-calibration-bench/registry/private.json` for private ones).
- **Private data**: lives under `privatedata/` (gitignored by `privatedata/.gitignore`
  = `*`). `crates/vision-calibration-bench/registry/private.json` is gitignored too
  (rule in repo `.gitignore`). NEVER commit private data. You MAY name
  `vision-metrology` and publish processing code — only the *data* stays private.
- Detection: published `calib-targets` 0.9.2 (crates.io, already a dep). It exposes
  ChArUco, chessboard, AND puzzleboard (`calib_targets::puzzleboard`,
  `calib_targets::detect::detect_puzzleboard`). Laser-line detection is in the
  `vision-metrology` Rust crate (not yet a dep).
- The published `calib-targets` source is unpacked at
  `/Users/vitalyvorobyev/vision/.cargo/registry/src/index.crates.io-*/calib-targets-0.9.2/`
  — read it to confirm exact signatures before wiring.

## 4. Key files

- `crates/vision-calibration-bench/src/run.rs` — runners (`run_planar_intrinsics`,
  `run_rig_extrinsics`, `run_single_cam_handeye`, `run_rig_handeye`) + pose loaders
  (`load_robot_poses_for` dispatches `rowmajor4x4`/`counted4x4`/`snap_list_json`)
  + `apply_tile` (private fn, ~line 990; crops a camera's ROI from a tiled frame;
  `None` tile = no-op) + `detector_for` (chessboard vs charuco from `board.layout`).
- `crates/vision-calibration-bench/src/detect.rs` — `DetectorKind{Chessboard,Charuco}`,
  `detect_charuco_view`, `detect_chessboard_view`, `charuco_params_for`,
  `glob_sorted_images`, `load_image`.
- `crates/vision-calibration-bench/src/registry.rs` — `BenchRegistry`/`BenchEntry`,
  `BoardGeometry` (has optional `dictionary`, `layout`, `marker_size_rel`),
  `PoseSource{path,format,convention,units}`, `CameraLayout{id,folder,filename_glob,
  tile,expected_size,color}`.
- `crates/vision-calibration-bench/registry/public.json` — committed datasets.
- `crates/vision-calibration-bench/registry/private.json` — GITIGNORED; currently
  holds `charuco_handeye_3536`.
- `crates/vision-calibration-pipeline/src/analysis/{mod.rs,tests.rs}` — the report.

## 5. REMAINING WORK (in priority order)

### W2c — wire `130x130_puzzle` (puzzleboard + Scheimpflug rig hand-eye)
Dataset: `privatedata/130x130_puzzle/`. **6 cameras × 19 poses**, images named
`cam_K_pose_N.png` (one file per camera-pose, NOT tiled — unlike 3536). Target:
puzzleboard 130×130, cellsize 1.0 mm, dict 4X4_1000, marker_scale 0.75
(`target.json`). `config.json`: Tsai hand-eye, `tilt_model` lens (Scheimpflug).
Files: `calibration.json` (ORACLE), `camera_metadata.json` (6 cams), `poses.json`,
`images/`, `laser/`.

Two blockers:
1. **Puzzleboard detection** — add a `DetectorKind::Puzzleboard` arm in detect.rs
   + a `detect_puzzleboard_view` (mirror `detect_charuco_view`: `to_luma8` →
   `detect::detect_puzzleboard` → map each corner's `target_position`(x,y,z=0) +
   `position` to a `CorrespondenceView`). Confirm the exact `detect_puzzleboard`
   signature + corner/Detection field names in the unpacked 0.9.2 source first.
   Wire `detector_for` to pick it from `board.layout == "puzzleboard"`.
2. **Scheimpflug** — `run_rig_handeye` is pinhole-only today. `RigHandeyeConfig`
   has a `sensor: SensorMode` (`Pinhole` | `Scheimpflug{..}`) per ADR 0013 /
   CLAUDE.md. Add an optional registry field (e.g. `sensor: "scheimpflug"` or a
   `scheimpflug: true` flag) and a config-construction branch in `run_rig_handeye`
   that sets `config.sensor` accordingly. Check how the `step_*` rig-handeye
   functions consume the sensor mode.

Pairing: per-camera glob `cam_K_pose_*.png` natural-sorts by pose → existing
index-pairing works (like DS8). VERIFY `poses.json` format first (19 poses;
probably the same snap-list shape — check whether it has `target_image`/`tcp2base`
or a different schema, and the units). Try WITHOUT manual init seeds first (user:
"maybe avoid, maybe not"). Oracle = `calibration.json`.

### W2d — Laser level via `vision-metrology`
Both private datasets have `laser_*.png`. Add an optional `vision-metrology` dep
behind a cargo feature (mirror how `calib-targets` is `dep:`-gated by `tier-b`).
Extract the laser line, RANSAC-fit a plane, emit the `Laser` level
(`ReprojLevel::Laser`, metric mm — the analysis types already have the variant).
`config.json` in each dataset carries the RANSAC opts the original pipeline used.

### F — batch diagnose-and-fix (the actual scientific goal)
Every hand-eye dataset shows **intrinsic floor ≪ hand-eye** (kuka 7.6×, DS8 26×,
3536 10×). The optics/detector are fine; the robot/hand-eye chain is the limit.
Investigate together: (a) robot-pose priors are 0.5°/1 mm default-on
(`refine_robot_poses: true`) — loosen them and watch whether hand-eye reproj drops
toward the floor (confirms "prior-limited poses"); (b) pose conventions
(base_se3_gripper vs inverse) per dataset; (c) DS8 has no oracle so its 4.86 px
could be a convention/scale bug — scrutinize. Also F1: stereo board declared 7×11
but the detector finds a larger grid.

### G — gate + docs
Regression-pin gate (single-solve numbers, NOT resampling), tolerances, ADR 0015
(document the multi-level metric), a tutorial. **CRITICAL**: the entire `data/`
dir is gitignored (`.gitignore:6` and more), so NO dataset reaches CI. The gate
must either (a) commit image-free frozen fixtures — `fixtures.rs::FrozenFixture`
IR already exists for exactly this — or (b) skip-if-data-absent. Public datasets
only in CI; private excluded. Proposed stereo_charuco pins (from W3): headline in
[0.45, 0.75] px; recovered baseline ‖t‖ within ~2 mm of 113.79 mm (needs the
recovered `cam_se3_rig` surfaced into `BenchRecord` first — currently `Fit` holds
only reproj stats, see the W3 doc's "follow-up").

CANCELLED earlier by the user (do NOT do): D1 detect-once/solve-many refactor;
stability/cross-validation resampling. "Just find a working workflow per dataset."

## 6. Gotchas learned this session
- `snap_list_json`: `poses.json` is a JSON list of `{target_image, tcp2base[4×4],
  type}`. `tcp2base` = `base_se3_gripper` **direct** (no inversion); translation in
  **mm** (scale 1e-3). Verified plausible on 3536.
- ChArUco glob: use `target_*[0-9].png` not `target_*.png` — the dir contains
  `target_0_xfeat_overlay.png` which must be excluded.
- The 3536 frames are 4320×540 = six 720×540 tiles; cam_k tile = `[k*720,0,720,540]`.
  This is why W2b models it as a 6-cam rig, not single-cam.
- Oracle `artifacts.json` for 3536 has `num_cameras=6`; its cam5 intrinsic is
  broken (127 px) — our bench recovers cam5 at 1.40 px, so don't trust that oracle
  entry blindly.

## 7. Discipline notes (cost + correctness)
- NEVER write a number into a doc/commit you haven't seen in real captured output.
  (A first W3 draft had fabricated per-component baseline numbers; it was amended.)
- Verify subagent claims against the real diff/run before committing — a delegated
  agent mis-modeled 3536 as single-camera; the rig model was the correct one.
- This session's Bash got flaky under build-lock contention and swallowed output
  containing diff `+/-` markers or raw Rust. Reliable pattern: write results to a
  `/tmp` file with python (defensive `.get`, no bare `[key]`), then Read the file;
  keep tool calls SEQUENTIAL (a failing call at the head of a parallel batch
  cascade-cancels the rest). Empty grep output = exit-1-no-match, not a fault.

## 8. Longer-form running notes
See `~/.claude/plans/benchmark-handoff.md` (this session appended SESSION 4
sections; one earlier note guessed a wrong W2b SHA — the correct one is `bd8d257`).

# Backlog

Execution status for agent-driven implementation tasks. Each completed task gets
a short report under `docs/report/` and a task-scoped commit.

## Benchmark

- [x] BENCH-W2C - Compact benchmark reports, puzzle rig wiring, laser extraction, and hand-eye diagnostics.
  Completed 2026-05-31. This task keeps compact JSON records at schema v3,
  writes full residuals only through sidecars, wires the private 130x130 puzzle
  rig and optional laser extraction, and adds deterministic hand-eye diagnostic
  sweeps. Puzzle calibration remains an open benchmark finding: the unseeded
  validation run was stopped after roughly 12 minutes with no solve output.
- [x] BENCH-W2D - Interactive benchmark dashboard, robot-correction visibility, and stage profiling.
  Completed 2026-05-31. Adds compact BenchRecord dashboard mode to the
  calibration viewer, reports robot-pose correction magnitudes in mm/degrees,
  exposes `diagnose stages` for target/laser timing, and records current
  hand-eye and private ChArUco quality findings.
- [x] BENCH-W2E - Fix DS8 hand-eye mode and known-grid checkerboard handling.
  Completed 2026-05-31, superseded by BENCH-W2F on the mode interpretation.
  Confirms DS8 uses a 10x14 checkerboard with 52 mm cells, rejects partial /
  local-grid checkerboard detections for this dataset, and extends hand-eye
  diagnostics with alternate-mode comparison.
- [x] BENCH-W2F - Benchmark viewer script, progress, artifacts, topological chessboard, and DS8 pose convention.
  Completed 2026-05-31. Adds `scripts/bench-viewer.sh`, stderr progress during
  dataset runs, calibration artifact output for the dashboard, topological
  chessboard dispatch for simple checkerboards, and corrects DS8 to physical
  EyeInHand with `gripper_se3_base` robot-pose convention.
- [x] BENCH-W2G - Fix viewer temp output path and validate KUKA topological chessboard run.
  Completed 2026-05-31. Writes script-generated benchmark JSON to `/tmp` so
  Vite can serve it through `/@fs`, and verifies `kuka_1` succeeds through the
  plain chessboard topological detector path.

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

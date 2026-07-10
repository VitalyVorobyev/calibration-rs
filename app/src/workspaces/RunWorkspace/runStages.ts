/** Pure logic for the Run workspace's stage-progress checklist.
 *
 * The runner (`app/src-tauri/src/run.rs`) announces exactly three coarse
 * stages — `detect` → `solve` → `export` — over the progress channel. The
 * UI shows them as a checklist; this module turns "the stage that is
 * currently running" into a per-stage `pending | active | done | cancelled`
 * state without any React or Tauri dependency, so it is unit-testable in a
 * plain Node environment. */
import type { RunStage } from "../../lib/runCalibration";

export interface RunStageInfo {
  id: RunStage;
  label: string;
}

/** Canonical ordered stage list, mirroring `run.rs`'s `RunStage`. The
 * order is load-bearing: a stage is `done` once a later stage becomes
 * active. */
export const RUN_STAGES: readonly RunStageInfo[] = [
  { id: "detect", label: "Detecting features" },
  { id: "solve", label: "Solving" },
  { id: "export", label: "Exporting" },
];

export type StageState = "pending" | "active" | "done" | "cancelled";

export interface StageRow extends RunStageInfo {
  state: StageState;
}

/** Derive each stage's checklist state.
 *
 * - `active` is the stage the runner last announced (`null` before the
 *   first message — everything is still `pending`).
 * - Stages before `active` are `done`; the active one is `active`; later
 *   ones are `pending`.
 * - When `cancelled` is set, the active stage and everything after it are
 *   marked `cancelled` (the run stopped at that boundary); already-finished
 *   stages stay `done`.
 * - When `completed` is set, every stage is `done` (the run finished
 *   successfully; the final `export` message may not have been observed
 *   before the promise resolved). */
export function computeStageRows(
  active: RunStage | null,
  opts: { cancelled?: boolean; completed?: boolean } = {},
): StageRow[] {
  if (opts.completed) {
    const done: StageState = "done";
    return RUN_STAGES.map((info) => ({ ...info, state: done }));
  }
  const activeIdx = active == null ? -1 : RUN_STAGES.findIndex((s) => s.id === active);
  return RUN_STAGES.map((info, i) => {
    let state: StageState;
    if (i < activeIdx) {
      state = "done";
    } else if (i === activeIdx) {
      state = opts.cancelled ? "cancelled" : "active";
    } else {
      state = opts.cancelled ? "cancelled" : "pending";
    }
    return { ...info, state };
  });
}

/** Human-readable elapsed time (e.g. `0.4s`, `12.8s`, `1m 03s`). */
export function formatElapsed(ms: number): string {
  const totalSeconds = ms / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

/** The stages of a calibration as a checklist, from the library's progress
 * reports: detecting the images, solving, and writing the run. */
import { formatCost, formatCount } from "../../lib/format";

export type StageState = "pending" | "active" | "done";

export interface StageRow {
  id: "detect" | "solve" | "export";
  label: string;
  state: StageState;
  /** What the stage is doing now (`41/120`, `iter 12 · cost 3.2e−1`). */
  detail: string | null;
}

/** The checklist at the job's latest report. `finished` is a job that is
 * done; a report of a later stage finishes the earlier ones, and a job
 * that reports no detection (the cache served it) shows detection done. */
export function stageChecklist(event: unknown, finished: boolean): StageRow[] {
  const e = (typeof event === "object" && event !== null ? event : {}) as {
    kind?: string;
    done?: number;
    total?: number;
    name?: string;
    iter?: number;
    cost?: number;
    max_iters?: number;
  };
  const detecting = e.kind === "detection" || e.kind === "laser_extraction";
  const solving = e.kind === "step" || e.kind === "iteration";
  const started = detecting || solving;

  const detect: StageRow = {
    id: "detect",
    label: "Detect targets",
    state:
      finished || solving ? "done" : detecting ? "active" : started ? "done" : "active",
    detail:
      detecting && e.total !== undefined
        ? `${formatCount(e.done ?? 0)}/${formatCount(e.total)}`
        : null,
  };
  const solve: StageRow = {
    id: "solve",
    label: "Solve",
    state: finished ? "done" : solving ? "active" : "pending",
    detail: solving
      ? e.kind === "iteration" && e.iter !== undefined
        ? `iter ${e.iter}/${e.max_iters ?? "?"} · cost ${formatCost(e.cost ?? Number.NaN)}`
        : (e.name ?? null)
      : null,
  };
  const exported: StageRow = {
    id: "export",
    label: "Write the run",
    state: finished ? "done" : "pending",
    detail: null,
  };
  return [detect, solve, exported];
}

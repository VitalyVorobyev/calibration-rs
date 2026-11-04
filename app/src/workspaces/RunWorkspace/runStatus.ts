/** The Run workspace's run state machine: idle → running → a terminal
 * state, as the runner's `RunResponse` decides. */
import type {
  Progress,
  RunProgress,
  RunResponse,
  RunStage,
} from "../../lib/runCalibration";

export type RunStatus =
  | { kind: "idle" }
  | {
      kind: "running";
      /** Correlation id for `cancelRun`. */
      runId: string;
      /** Last stage the runner announced; `null` until the first message. */
      stage: RunStage | null;
      /** The library's latest report within the stage, if any. */
      report: Progress | null;
      /** The calibration step under way, for its iterations. */
      step: string | null;
      /** Epoch ms the run started — UI elapsed clock only, never exported. */
      startedAt: number;
      /** True once the user clicked Cancel (awaiting the stop). */
      cancelRequested: boolean;
    }
  | { kind: "cancelled" }
  | {
      kind: "ok";
      durationMs: number;
      usable: number;
      total: number;
      cacheUsed: boolean;
      /** The project run it was saved as, when a project is open. */
      savedAs: string | null;
    }
  | { kind: "error"; category: string; message: string }
  | { kind: "validation"; message: string }
  | { kind: "ask_user"; field: string; prompt: string; suggestions: string[] };

/** A running status advanced by one progress message: a new stage clears
 * the report, a step is remembered for the iterations that follow. */
export function withProgress(
  status: Extract<RunStatus, { kind: "running" }>,
  progress: RunProgress,
): Extract<RunStatus, { kind: "running" }> {
  const report = progress.report ?? null;
  return {
    ...status,
    stage: progress.stage,
    report,
    step: report?.kind === "step" ? report.name : status.step,
  };
}

/** The status a finished run's response leads to. `savedAs` names the
 * project run a successful export was saved as. */
export function statusFromResponse(
  response: RunResponse,
  savedAs: string | null = null,
): RunStatus {
  switch (response.kind) {
    case "ok":
      return {
        kind: "ok",
        durationMs: response.duration_ms,
        usable: response.usable_views,
        total: response.total_views,
        cacheUsed: response.cache_used,
        savedAs,
      };
    case "cancelled":
      return { kind: "cancelled" };
    case "ask_user":
      return {
        kind: "ask_user",
        field: response.field,
        prompt: response.prompt,
        suggestions: response.suggestions,
      };
    case "validation_failed":
      return { kind: "validation", message: response.message };
    case "failed":
      return { kind: "error", category: response.category, message: response.message };
  }
}

/** Fresh correlation id for one run — routes the progress channel and
 * cancellation. `crypto.randomUUID` is available in every Tauri webview
 * and modern browser; the fallback covers environments without it (older
 * jsdom). */
export function newRunId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `run-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

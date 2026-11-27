/** The Run workspace's run state machine: idle → running → a terminal
 * state, as the `calibrate` job ends: its output (the run it wrote) or its
 * error envelope. */
import { asStudioError } from "../../api";
import type { RunInfo } from "../../api/generated/types";
import { stageOf, type Progress, type RunStage } from "./runStages";

export type RunStatus =
  | { kind: "idle" }
  | {
      kind: "running";
      /** This run's id in the UI (a stale event names another). */
      runId: string;
      /** The stage of the latest report; `null` until the first one. */
      stage: RunStage | null;
      /** The library's latest report, if any. */
      report: Progress | null;
      /** The calibration step under way, for its iterations. */
      step: string | null;
      /** Epoch ms the run started — UI elapsed clock only, never exported. */
      startedAt: number;
      /** True once the user clicked Cancel (awaiting the stop). */
      cancelRequested: boolean;
    }
  | { kind: "cancelled" }
  | { kind: "ok"; run: RunInfo }
  | { kind: "error"; category: string; message: string }
  | { kind: "ask_user"; field: string; prompt: string; suggestions: string[] };

/** A running status advanced by one progress report: its stage, and a step
 * remembered for the iterations that follow. */
export function withProgress(
  status: Extract<RunStatus, { kind: "running" }>,
  report: Progress,
): Extract<RunStatus, { kind: "running" }> {
  return {
    ...status,
    stage: stageOf(report),
    report,
    step: report.kind === "step" ? report.name : status.step,
  };
}

/** The status a failed run's error leads to: a choice the manifest leaves
 * open is a question (`needs_input`), a cancellation is its own state,
 * anything else an error named by its kind. */
export function statusFromError(e: unknown): RunStatus {
  const error = asStudioError(e);
  if (error.kind === "cancelled") return { kind: "cancelled" };
  if (error.kind === "needs_input" && error.field) {
    return {
      kind: "ask_user",
      field: error.field,
      prompt: error.message,
      suggestions: error.suggestions ?? [],
    };
  }
  const where = error.field ? ` (${error.field})` : "";
  return { kind: "error", category: error.kind, message: `${error.message}${where}` };
}

/** What a written run says about itself, for the success banner: its
 * views and verdict. */
export function runSummaryLine(run: RunInfo): string {
  if (run.state.kind !== "recorded") return run.id;
  const parts = [`saved as runs/${run.id}`];
  const source = run.state.source;
  if (source.kind === "calibrated") {
    parts.unshift(
      `${source.duration_ms} ms`,
      `${source.usable_views}/${source.total_views} usable views`,
    );
  }
  const verdict = run.state.summary.verdict;
  if (verdict) parts.push(`verdict ${verdict}`);
  return parts.join(" · ");
}

/** Fresh correlation id for one run in the UI. `crypto.randomUUID` is
 * available in every Tauri webview and modern browser; the fallback covers
 * environments without it (older jsdom). */
export function newRunId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `run-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

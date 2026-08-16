/** Typed wrapper around the `run_calibration_cmd` / `cancel_run_cmd`
 * Tauri commands. Mirrors the Rust `RunResponse` enum (see
 * app/src-tauri/src/run.rs) so the React side can match on `kind` rather
 * than parsing strings, and streams stage progress over a
 * `tauri::ipc::Channel<RunProgress>`. */
import { Channel, invoke } from "@tauri-apps/api/core";
import type { RunProgress, RunStage } from "../types/generated/diagnose-wire";

// Re-export the generated progress wire types so components import the
// stage vocabulary from one place.
export type { RunProgress, RunStage } from "../types/generated/diagnose-wire";

export interface RunSuccess {
  kind: "ok";
  export: unknown;
  duration_ms: number;
  usable_views: number;
  total_views: number;
  cache_used: boolean;
}

export interface RunAskUser {
  kind: "ask_user";
  field: string;
  prompt: string;
  suggestions: string[];
}

export interface RunValidationFailed {
  kind: "validation_failed";
  message: string;
}

export interface RunFailed {
  kind: "failed";
  category: string;
  message: string;
}

/** The user cancelled the run; it stopped at the next stage boundary
 * (solver-internal cancellation is out of scope). */
export interface RunCancelled {
  kind: "cancelled";
}

export type RunResponse =
  RunSuccess | RunAskUser | RunValidationFailed | RunFailed | RunCancelled;

export interface RunInput {
  /** Caller-generated correlation id, echoed to `cancelRun`. */
  runId: string;
  manifest: unknown;
  config: unknown;
  manifestDir: string;
  /** Invoked as each pipeline stage begins (detect → solve → export). */
  onProgress?: (stage: RunStage) => void;
}

/** Launch a calibration run, streaming per-stage progress. The returned
 * promise resolves once the run reaches a terminal state (ok / cancelled
 * / failed / validation / ask_user). */
export async function runCalibration(input: RunInput): Promise<RunResponse> {
  const channel = new Channel<RunProgress>();
  if (input.onProgress) {
    channel.onmessage = (message: RunProgress) => input.onProgress?.(message.stage);
  }
  return invoke<RunResponse>("run_calibration_cmd", {
    runId: input.runId,
    manifestJson: input.manifest,
    configJson: input.config,
    manifestDir: input.manifestDir,
    onProgress: channel,
  });
}

/** Request cancellation of an in-flight run. Resolves `true` when a
 * matching run was still running and got flagged, `false` if it had
 * already finished (a benign race). */
export async function cancelRun(runId: string): Promise<boolean> {
  return invoke<boolean>("cancel_run_cmd", { runId });
}

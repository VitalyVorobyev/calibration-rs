/** Typed wrapper around the `run_calibration_cmd` Tauri command.
 * Mirrors the Rust `RunResponse` enum (see app/src-tauri/src/run.rs)
 * so the React side can match on `kind` rather than parsing strings. */
import { invoke } from "@tauri-apps/api/core";

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

export type RunResponse = RunSuccess | RunAskUser | RunValidationFailed | RunFailed;

export interface RunInput {
  manifest: unknown;
  config: unknown;
  manifestDir: string;
}

export async function runCalibration(input: RunInput): Promise<RunResponse> {
  return invoke<RunResponse>("run_calibration_cmd", {
    manifestJson: input.manifest,
    configJson: input.config,
    manifestDir: input.manifestDir,
  });
}

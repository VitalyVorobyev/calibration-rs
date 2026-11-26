/** A calibration run as a back-end job (`run_calibration`): its progress
 * (stage announcements with the library's per-image, per-step and
 * per-iteration reports) arrives as the job's events, its outcome is the
 * job's output (`RunResponse`), and cancelling the job stops it at the next
 * image or solver iteration. */
import { api, type Job } from "../api";
import type { RunProgress, RunResponse } from "../api/generated/types";

// Re-export the generated wire types so components import the run's
// vocabulary from one place.
export type {
  Progress,
  RunProgress,
  RunResponse,
  RunStage,
  RunSuccess,
} from "../api/generated/types";

export interface RunInput {
  manifest: unknown;
  config: unknown;
  /** The folder the manifest's relative paths resolve against. */
  manifestDir: string;
}

/** Start a calibration run. */
export function startRun(input: RunInput): Promise<Job<RunProgress, RunResponse>> {
  return api.jobs.runCalibration({
    manifest: input.manifest,
    config: input.config,
    manifest_dir: input.manifestDir,
  });
}

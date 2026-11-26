/** The back end's project commands, as the project store calls them. A
 * project is a folder holding `calibration-project.json`, the dataset
 * manifest `dataset.json` and `runs/<name>/`; the wire types are generated
 * from the back end's. */
import { api, asStudioError, StudioError } from "../api";
import type {
  ErrorKind,
  ProjectInfo,
  RunDiagnostics,
  RunRecord,
  RunSummary,
} from "../api/generated/types";
import { joinPath } from "../lib/paths";

export type { ProjectInfo, RunDiagnostics, RunRecord, RunSummary };

/** A failed project command, as the shell shows it. */
export interface ProjectError {
  kind: ErrorKind;
  message: string;
}

/** A rejected project command as a `ProjectError`. The back end rejects
 * with its error envelope; anything else (an IPC failure) is reported as
 * `io`. */
export function asProjectError(e: unknown): ProjectError {
  const envelope =
    e instanceof StudioError ||
    (typeof e === "object" && e !== null && "kind" in e && "message" in e);
  if (envelope) {
    const { kind, message } = asStudioError(e);
    return { kind, message };
  }
  return { kind: "io", message: e instanceof Error ? e.message : String(e) };
}

// Every command below rejects with the back end's error envelope; callers
// narrow it with `asProjectError`.

/** Make `dir` a project for the dataset in `datasetDir`. */
export function createProject(
  dir: string,
  datasetDir: string,
  manifest: unknown,
): Promise<ProjectInfo> {
  return api.createProject({ dir, dataset_dir: datasetDir, manifest });
}

/** Open the project in `dir`; rejects with `not_a_project` for any other
 * folder. */
export function openProject(dir: string): Promise<ProjectInfo> {
  return api.openProject({ dir });
}

/** Replace the project's `dataset.json`. */
export async function saveProjectManifest(dir: string, manifest: unknown): Promise<void> {
  await api.saveProjectManifest({ dir, manifest });
}

/** Save a run as `runs/<name>/`; rejects with `run_exists` rather than
 * overwrite one. */
export function saveProjectRun(
  dir: string,
  name: string,
  record: RunRecord,
): Promise<RunSummary> {
  return api.saveProjectRun({ dir, name, record });
}

/** The `export.json` of a saved run. */
export function runExportPath(projectDir: string, run: string): string {
  return joinPath(projectDir, "runs", run, "export.json");
}

/** The project's manifest file. */
export function projectManifestPath(projectDir: string): string {
  return joinPath(projectDir, "dataset.json");
}

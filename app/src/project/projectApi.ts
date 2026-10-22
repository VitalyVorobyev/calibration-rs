/** Typed wrappers around the backend's project commands
 * (`src-tauri/src/project.rs`). A project is a folder holding
 * `calibration-project.json`, the dataset manifest `dataset.json` and
 * `runs/<name>/`; the wire types are generated from the Rust ones. */
import { invoke } from "@tauri-apps/api/core";
import type {
  ProjectError,
  ProjectErrorKind,
  ProjectInfo,
  RunDiagnostics,
  RunSummary,
} from "../types/generated/diagnose-wire";
import { joinPath } from "../lib/tauri";

export type { ProjectError, ProjectInfo, RunDiagnostics, RunSummary };

/** Everything a saved run records (`runs/<name>/*.json`). */
export interface RunRecord {
  manifest: unknown;
  config: unknown;
  /** The export as the runner returned it. */
  export: unknown;
  diagnostics: RunDiagnostics;
  /** The folder the run resolved the manifest's paths against. */
  dataset_dir: string;
}

const KINDS: readonly ProjectErrorKind[] = [
  "not_a_project",
  "project_exists",
  "unsupported_version",
  "invalid_run_name",
  "run_exists",
  "io",
  "parse",
];

/** A rejected project command as a `ProjectError`. The backend rejects
 * with one; anything else (an IPC failure) is reported as `io`. */
export function asProjectError(e: unknown): ProjectError {
  if (typeof e === "object" && e !== null) {
    const { kind, message } = e as Record<string, unknown>;
    if (KINDS.includes(kind as ProjectErrorKind) && typeof message === "string") {
      return { kind: kind as ProjectErrorKind, message };
    }
  }
  return { kind: "io", message: e instanceof Error ? e.message : String(e) };
}

// Every command below rejects with the backend's `ProjectError`; callers
// narrow it with `asProjectError`.

/** Make `dir` a project for the dataset in `datasetDir`. */
export function createProject(
  dir: string,
  datasetDir: string,
  manifest: unknown,
): Promise<ProjectInfo> {
  return invoke("create_project", { dir, datasetDir, manifest });
}

/** Open the project in `dir`; rejects with `not_a_project` for any other
 * folder. */
export function openProject(dir: string): Promise<ProjectInfo> {
  return invoke("open_project", { dir });
}

/** Replace the project's `dataset.json`. */
export function saveProjectManifest(dir: string, manifest: unknown): Promise<void> {
  return invoke("save_project_manifest", { dir, manifest });
}

/** Save a run as `runs/<name>/`; rejects with `run_exists` rather than
 * overwrite one. */
export function saveProjectRun(
  dir: string,
  name: string,
  record: RunRecord,
): Promise<RunSummary> {
  return invoke("save_project_run", { dir, name, record });
}

/** The `export.json` of a saved run. */
export function runExportPath(projectDir: string, run: string): string {
  return joinPath(projectDir, "runs", run, "export.json");
}

/** The project's manifest file. */
export function projectManifestPath(projectDir: string): string {
  return joinPath(projectDir, "dataset.json");
}

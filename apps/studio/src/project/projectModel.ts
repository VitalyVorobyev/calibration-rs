/** Pure project-model rules over the back end's wire types: folder labels,
 * the failed command as the shell shows it, a run's label and summary, and
 * whether a run was made from the project's current inputs. */
import { asStudioError, StudioError } from "../api";
import type {
  DatasetSpec,
  ErrorKind,
  ProjectDigest,
  ProjectSpec,
  RunInfo,
  RunSummary,
  Verdict,
} from "../api/generated/types";

/** The last path segment, for labels. */
export function folderName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return idx >= 0 ? trimmed.slice(idx + 1) : trimmed;
}

/** A failed project command, as the shell shows it. */
export interface ProjectError {
  kind: ErrorKind;
  message: string;
  field?: string | null;
}

/** A rejected command as a `ProjectError`. The back end rejects with its
 * error envelope; anything else (an IPC failure) is reported as `io`. */
export function asProjectError(e: unknown): ProjectError {
  const envelope =
    e instanceof StudioError ||
    (typeof e === "object" && e !== null && "kind" in e && "message" in e);
  if (envelope) {
    const { kind, message, field } = asStudioError(e);
    return field ? { kind, message, field } : { kind, message };
  }
  return { kind: "io", message: e instanceof Error ? e.message : String(e) };
}

/** The run's label (its notes' label, else its id), starred with `★`. */
export function runLabel(run: RunInfo): string {
  const label = run.notes.label ? `${run.id} · ${run.notes.label}` : run.id;
  return run.notes.starred ? `★ ${label}` : label;
}

/** A short line about a run: its views and verdict, or what its files are. */
export function runMeta(run: RunInfo): string | undefined {
  const state = run.state;
  switch (state.kind) {
    case "recorded": {
      const verdict = state.summary.verdict ?? null;
      if (state.source.kind === "calibrated") {
        const views = `${state.source.usable_views}/${state.source.total_views} views`;
        return verdict ? `${views} · ${verdict}` : views;
      }
      return verdict ? `imported · ${verdict}` : "imported";
    }
    case "unrecorded":
      return state.usable_views != null && state.total_views != null
        ? `${state.usable_views}/${state.total_views} views`
        : "earlier format";
    case "unreadable":
      return "unreadable";
  }
}

/** The parts of a calibrated run's input digest that differ from the
 * project's now (`dataset`, `exclusions`, `config`, `quality`); empty when
 * the run is current, `null` when it cannot be told (an imported or
 * earlier-format run, or a project whose inputs do not resolve). */
export function staleParts(
  run: RunInfo,
  current: ProjectDigest | null | undefined,
): (keyof ProjectDigest)[] | null {
  if (!current || run.state.kind !== "recorded") return null;
  const source = run.state.source;
  if (source.kind !== "calibrated") return null;
  const parts: (keyof ProjectDigest)[] = ["dataset", "exclusions", "config", "quality"];
  return parts.filter((part) => source.digest[part] !== current[part]);
}

/** The manifest a project's file holds; `null` for a manifest of its own
 * (a referenced file, never written by the app). */
export function inlineManifest(spec: ProjectSpec): DatasetSpec | null {
  const manifest = spec.dataset.manifest;
  return typeof manifest === "string" ? null : manifest;
}

/** `spec` with `manifest` written in its file (an inline dataset only). */
export function withManifest(spec: ProjectSpec, manifest: DatasetSpec): ProjectSpec {
  return { ...spec, dataset: { ...spec.dataset, manifest } };
}

/** A new project's spec for a manifest drafted from its folder (the
 * project is that folder: the data root is `.`). */
export function inlineSpec(manifest: DatasetSpec): ProjectSpec {
  return { version: 2, dataset: { root: ".", manifest } };
}

/** A new project's spec referencing the manifest file `name` beside it. */
export function referencedSpec(name: string): ProjectSpec {
  return { version: 2, dataset: { manifest: name } };
}

/** The runs, newest first (by id: `run-010` is after `run-009`). */
export function runsNewestFirst(runs: readonly RunInfo[]): RunInfo[] {
  return [...runs].sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true }));
}

/** The run a project opens on: its newest that reads. */
export function defaultRun(runs: readonly RunInfo[]): RunInfo | null {
  return runsNewestFirst(runs).find((r) => r.state.kind !== "unreadable") ?? null;
}

/** What a run's files are: made here (`Recorded`), brought in from a file
 * (`Imported`), of the app's earlier format (`Legacy`), or not readable. */
export type RunStateName = "Recorded" | "Imported" | "Legacy" | "Unreadable";

export function runStateName(run: RunInfo): RunStateName {
  switch (run.state.kind) {
    case "recorded":
      return run.state.source.kind === "imported" ? "Imported" : "Recorded";
    case "unrecorded":
      return "Legacy";
    case "unreadable":
      return "Unreadable";
  }
}

/** The run's summary, when its files carry one. */
export function runSummary(run: RunInfo): RunSummary | null {
  return run.state.kind === "unreadable" ? null : (run.state.summary ?? null);
}

/** The run's verdict, when it has one. */
export function runVerdict(run: RunInfo): Verdict | null {
  return runSummary(run)?.verdict ?? null;
}

/** The reprojection RMS over all the run's cameras, pixels: the cameras'
 * own RMS pooled by their feature counts (the RMS of every error length at
 * once); `null` when no camera has one. */
export function runRms(run: RunInfo): number | null {
  const cameras = runSummary(run)?.cameras ?? [];
  let squares = 0;
  let features = 0;
  for (const camera of cameras) {
    if (camera.rms_px == null || camera.features <= 0) continue;
    squares += camera.features * camera.rms_px * camera.rms_px;
    features += camera.features;
  }
  return features > 0 ? Math.sqrt(squares / features) : null;
}

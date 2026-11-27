/** A run as the history shows it: the numbers of its record read into one
 * row, what its settings changed from the project's, and which run has the
 * lowest error. Pure rules over the wire types. */
import type {
  ProjectDigest,
  ProjectView,
  RunInfo,
  Verdict,
} from "../api/generated/types";
import { patchChanges, patchLeaves, shortValue } from "./configEdit";
import {
  runRms,
  runStateName,
  runSummary,
  staleParts,
  type RunStateName,
} from "./projectModel";

export interface CameraRms {
  id: string;
  rms: number | null;
  features: number;
}

export interface RunRow {
  id: string;
  label: string | null;
  starred: boolean;
  notes: string | null;
  state: RunStateName;
  verdict: Verdict | null;
  /** Pooled over the cameras, pixels. */
  rms: number | null;
  cameras: CameraRms[];
  /** Views that entered the calibration, and the views the dataset formed
   * (a run's own record; an earlier format may have them without the rest). */
  views: { used: number; total: number } | null;
  /** Features over every camera. */
  features: number | null;
  /** When it was written, seconds since the epoch. */
  timestamp: number | null;
  durationMs: number | null;
  /** The leaves its settings set otherwise than the project's now; `null`
   * for a run with no record of them. */
  changes: string[] | null;
  /** The inputs of the project that differ from the run's; `null` when it
   * cannot be told. */
  stale: (keyof ProjectDigest)[] | null;
  /** The message of an unreadable run. */
  problem: string | null;
}

/** The recorded configuration changes of a run: the project's, then its
 * own; `null` for a run that has none on record. */
export function recordedPatch(run: RunInfo): Record<string, unknown> | null {
  if (run.state.kind !== "recorded" || run.state.source.kind !== "calibrated")
    return null;
  return run.state.source.inputs.config_patch;
}

/** The inputs of the project that changed since `run` was made. A run
 * taken with settings of its own differs from the project's configuration
 * by those settings (the library counts that as a difference of its
 * digest): when the run's recorded changes explain the whole difference of
 * the configuration, it is not "changed since", and the changes say it. */
export function staleSince(
  run: RunInfo,
  digest: ProjectDigest | null | undefined,
  projectPatch: unknown,
): (keyof ProjectDigest)[] | null {
  const parts = staleParts(run, digest);
  if (parts === null || !parts.includes("config")) return parts;
  const patch = recordedPatch(run);
  return patch && patchChanges(patch, projectPatch ?? {}).length > 0
    ? parts.filter((p) => p !== "config")
    : parts;
}

export function runRow(run: RunInfo, project: ProjectView): RunRow {
  const summary = runSummary(run);
  const cameras = (summary?.cameras ?? []).map((c) => ({
    id: c.id,
    rms: c.rms_px ?? null,
    features: c.features,
  }));
  const state = run.state;
  const calibrated =
    state.kind === "recorded" && state.source.kind === "calibrated" ? state.source : null;
  const patch = recordedPatch(run);
  let views: RunRow["views"] = null;
  if (calibrated)
    views = { used: calibrated.usable_views, total: calibrated.total_views };
  else if (
    state.kind === "unrecorded" &&
    state.usable_views != null &&
    state.total_views != null
  ) {
    views = { used: state.usable_views, total: state.total_views };
  }
  return {
    id: run.id,
    label: run.notes.label ?? null,
    starred: run.notes.starred ?? false,
    notes: run.notes.notes ?? null,
    state: runStateName(run),
    verdict: summary?.verdict ?? null,
    rms: runRms(run),
    cameras,
    views,
    features: cameras.length > 0 ? cameras.reduce((n, c) => n + c.features, 0) : null,
    timestamp: state.kind === "recorded" ? state.timestamp : null,
    durationMs: calibrated ? calibrated.duration_ms : null,
    changes: patch ? patchChanges(patch, project.spec.config ?? {}) : null,
    stale: staleSince(run, project.input_digest, project.spec.config),
    problem: state.kind === "unreadable" ? state.message : null,
  };
}

/** The run with the lowest pooled RMS among those whose verdict is not
 * `fail`; `null` when fewer than two runs have an RMS (nothing to be
 * lowest among). RMS alone does not make a model right: this marks the
 * lowest error, and the star marks the run a person accepts. */
export function lowestRmsRun(rows: readonly RunRow[]): string | null {
  const candidates = rows.filter((r) => r.rms !== null && r.verdict !== "fail");
  if (candidates.length < 2) return null;
  const best = candidates.reduce((a, b) => (b.rms! < a.rms! ? b : a));
  // Runs that read the same to the pixel's thousandth have no lowest.
  const shown = (r: RunRow) => Math.round(r.rms! * 1000);
  return candidates.filter((r) => shown(r) === shown(best)).length > 1 ? null : best.id;
}

/** The leaves of `run`'s recorded patch that differ from `editor`'s patch
 * (from the library defaults), each with the run's value and the editor's,
 * as short text. */
export function differencesFromEditor(
  run: Record<string, unknown>,
  editor: Record<string, unknown>,
): { path: string; run: string; editor: string }[] {
  const left = patchLeaves(run);
  const right = patchLeaves(editor);
  return patchChanges(run, editor).map((path) => ({
    path,
    run: left.get(path) ?? "default",
    editor: right.get(path) ?? "default",
  }));
}

/** A recorded leaf as text, `"huber"` → `huber`. */
export function leafText(json: string | undefined): string {
  if (json === undefined) return "default";
  try {
    return shortValue(JSON.parse(json) as unknown);
  } catch {
    return json;
  }
}

/** A written time, `27 Nov, 09:41`, from seconds since the epoch. */
export function formatWhen(seconds: number, locale?: string): string {
  const date = new Date(seconds * 1000);
  return date.toLocaleString(locale, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

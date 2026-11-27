/** What "rerun from this run" has to say about the dataset: a run records
 * the exclusions it used, and the project may have changed them since, so
 * the rerun would not be like for like. */
import type { Exclusions, RunInfo } from "../api/generated/types";

function stable(value: unknown): string {
  return JSON.stringify(value, (_k, v: unknown) =>
    v !== null && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(
          Object.entries(v as Record<string, unknown>).sort(([a], [b]) =>
            a < b ? -1 : a > b ? 1 : 0,
          ),
        )
      : v,
  );
}

const KINDS = ["images", "views", "features"] as const;
export type ExclusionKind = (typeof KINDS)[number];

/** The exclusions a calibrated run used; `null` for a run with no record. */
export function runExclusions(run: RunInfo): Exclusions | null {
  if (run.state.kind !== "recorded" || run.state.source.kind !== "calibrated")
    return null;
  return run.state.source.inputs.exclusions;
}

export interface ExclusionDifference {
  kind: ExclusionKind;
  /** In the run's, not in the project's now. */
  onlyInRun: number;
  /** In the project's now, not in the run's. */
  onlyInProject: number;
}

/** Where the run's exclusions and the project's differ, per kind; empty
 * when they are the same set. */
export function exclusionDifferences(
  run: Exclusions | null | undefined,
  project: Exclusions | null | undefined,
): ExclusionDifference[] {
  const out: ExclusionDifference[] = [];
  for (const kind of KINDS) {
    const a = new Set((run?.[kind] ?? []).map(stable));
    const b = new Set((project?.[kind] ?? []).map(stable));
    const onlyInRun = [...a].filter((x) => !b.has(x)).length;
    const onlyInProject = [...b].filter((x) => !a.has(x)).length;
    if (onlyInRun > 0 || onlyInProject > 0) out.push({ kind, onlyInRun, onlyInProject });
  }
  return out;
}

/** The differences as a sentence fragment. */
export function describeDifferences(diffs: readonly ExclusionDifference[]): string {
  return diffs
    .map((d) => {
      const parts: string[] = [];
      if (d.onlyInRun > 0) parts.push(`${d.onlyInRun} the run had`);
      if (d.onlyInProject > 0) parts.push(`${d.onlyInProject} only the project has`);
      return `${d.kind}: ${parts.join(", ")}`;
    })
    .join("; ");
}

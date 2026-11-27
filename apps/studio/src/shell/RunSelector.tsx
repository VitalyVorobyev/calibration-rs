import { Badge, Select, Tooltip } from "@vitavision/ui";
import { useMemo } from "react";
import { runLabel, runMeta, runsNewestFirst } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { exportKindLabel } from "../store/exportKind";
import { runPills, type Pill } from "./runPills";

/** The run the app is about, chosen once, here: a `Select` of the open
 * project's runs, newest first, and the pills that say what the run is
 * (the state of its files, its verdict in words, whether the project's
 * inputs changed since). Inspect, Compare (as its baseline) and Device read
 * this run. */
export function RunSelector() {
  const project = useProject((s) => s.project);
  const shown = useStore((s) => s.run);
  const openRun = useProject((s) => s.openRun);
  const kind = useStore((s) => s.kind);

  const runs = useMemo(() => runsNewestFirst(project?.runs ?? []), [project?.runs]);
  const run =
    project && shown && shown.dir === project.dir
      ? (project.runs.find((r) => r.id === shown.id) ?? null)
      : null;
  const pills =
    run && project ? runPills(run, project.input_digest, project.spec.config) : null;

  if (!project) return null;
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="text-[10px] font-semibold tracking-wider text-fg-subtle uppercase">
        Run
      </span>
      <Select
        aria-label="Run"
        className="w-56 [&>span:first-child]:truncate [&>span:first-child]:text-left"
        value={run?.id ?? ""}
        onValueChange={(id) => void openRun(id)}
        disabled={runs.length === 0}
        placeholder={runs.length === 0 ? "No runs yet" : "Choose a run"}
        options={runs.map((r) => {
          const meta = runMeta(r);
          return {
            value: r.id,
            label: runLabel(r),
            ...(meta ? { note: meta } : {}),
          };
        })}
      />
      {pills && (
        <div className="flex items-center gap-1.5" aria-label="State of the run">
          <PillView pill={pills.state} />
          {pills.verdict && <PillView pill={pills.verdict} />}
          {pills.stale && <PillView pill={pills.stale} />}
          {kind && (
            <span className="ml-1 truncate text-xs text-fg-muted">
              {exportKindLabel(kind)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function PillView({ pill }: { pill: Pill }) {
  const badge = <Badge tone={pill.tone}>{pill.label}</Badge>;
  if (!pill.title) return badge;
  return (
    <Tooltip content={pill.title}>
      <span
        tabIndex={0}
        className="inline-flex rounded-control outline-none focus-visible:outline-2 focus-visible:outline-signal"
      >
        {badge}
      </span>
    </Tooltip>
  );
}

import { DensityProvider, Empty, Panel } from "@vitavision/ui";
import { TreeView, type TreeNode } from "@vitavision/workbench";
import { FlaskConical } from "lucide-react";
import { useMemo } from "react";
import type { ProjectView, RunInfo } from "../api/generated/types";
import { runLabel, runMeta, staleParts } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { RunActions } from "./RunActions";

/** The navigator of an open project: its name, its data folder, and its
 * runs. Choosing a run loads its export into the viewers; the chosen run's
 * label, star, report and deletion are below the list. */
export function ProjectPanel({ project }: { project: ProjectView }) {
  const shown = useStore((s) => s.run);
  const openRun = useProject((s) => s.openRun);

  const nodes = useMemo<TreeNode[]>(
    () => project.runs.map((run) => runNode(run, project)),
    [project],
  );
  const selected =
    shown !== null && shown.dir === project.dir
      ? (project.runs.find((r) => r.id === shown.id) ?? null)
      : null;

  return (
    <DensityProvider value="compact">
      <div className="flex flex-col gap-2.5 p-2.5">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="truncate text-xs font-semibold text-fg" title={project.dir}>
            {project.name}
          </span>
          <span
            className="truncate font-mono text-[10px] text-fg-muted"
            title={project.data_root}
          >
            data {project.data_root === project.dir ? "." : project.data_root}
          </span>
        </div>
        <Panel title="Runs">
          {nodes.length === 0 ? (
            <Empty>No runs yet. Run a calibration from the Run workspace.</Empty>
          ) : (
            <TreeView
              aria-label="Saved runs"
              nodes={nodes}
              selectedId={selected?.id ?? null}
              onSelect={(id) => void openRun(id)}
            />
          )}
        </Panel>
        {selected && <RunActions key={selected.id} project={project} run={selected} />}
      </div>
    </DensityProvider>
  );
}

function runNode(run: RunInfo, project: ProjectView): TreeNode {
  const stale = staleParts(run, project.input_digest);
  const meta = runMeta(run);
  return {
    id: run.id,
    label: runLabel(run),
    icon: <FlaskConical className="size-3.5" />,
    meta: stale && stale.length > 0 ? `${meta ?? ""} · stale` : meta,
  };
}

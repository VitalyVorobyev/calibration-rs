import { DensityProvider, Empty, Panel } from "@vitavision/ui";
import { TreeView, type TreeNode } from "@vitavision/workbench";
import { FlaskConical } from "lucide-react";
import { useMemo } from "react";
import { joinPath } from "../lib/tauri";
import { runExportPath, type ProjectInfo, type RunSummary } from "../project/projectApi";
import { folderName } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";

/** The navigator of an open project: where it is, its dataset folder, and
 * its saved runs. Choosing a run loads its export into the viewers. */
export function ProjectPanel({ project }: { project: ProjectInfo }) {
  const exportPath = useStore((s) => s.exportPath);
  const openRun = useProject((s) => s.openRun);

  const nodes = useMemo<TreeNode[]>(() => project.runs.map(runNode), [project.runs]);
  // The run whose export is loaded, compared as `/`-separated paths.
  const selectedId = useMemo(() => {
    if (exportPath === null) return null;
    const loaded = joinPath(exportPath);
    return (
      project.runs.find((r) => runExportPath(project.dir, r.name) === loaded)?.name ??
      null
    );
  }, [exportPath, project]);

  return (
    <DensityProvider value="compact">
      <div className="flex flex-col gap-2.5 p-2.5">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="truncate text-xs font-semibold text-fg" title={project.dir}>
            {folderName(project.dir)}
          </span>
          <span
            className="truncate font-mono text-[10px] text-fg-muted"
            title={project.dataset_dir}
          >
            dataset {project.dataset_dir === project.dir ? "." : project.dataset_dir}
          </span>
        </div>
        <Panel title="Runs">
          {nodes.length === 0 ? (
            <Empty>
              No saved runs yet. A run that finishes while this project is open is saved
              here.
            </Empty>
          ) : (
            <TreeView
              aria-label="Saved runs"
              nodes={nodes}
              selectedId={selectedId}
              onSelect={(id) => void openRun(id)}
            />
          )}
        </Panel>
      </div>
    </DensityProvider>
  );
}

function runNode(run: RunSummary): TreeNode {
  const d = run.diagnostics;
  return {
    id: run.name,
    label: run.name,
    icon: <FlaskConical className="size-3.5" />,
    meta: d ? `${d.usable_views}/${d.total_views} views` : undefined,
  };
}

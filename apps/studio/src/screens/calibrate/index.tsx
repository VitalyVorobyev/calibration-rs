import { Empty } from "@vitavision/ui";
import { useProject } from "../../project/projectStore";
import { projectTopology, useRunEditor, workingConfig } from "../../project/runEditor";
import { ParameterEditor } from "./ParameterEditor";
import { RunPanel } from "./RunPanel";
import { useEditorCheck } from "./useEditorCheck";

const NO_PATCH: Record<string, unknown> = {};

/** Calibrate: set up what to estimate, run it, and keep the history of
 * attempts (the run list on the left, the shell's). The centre is the
 * parameter editor; the right is "This run": what will happen, Run, and
 * its progress. */
export function CalibrateScreen() {
  const project = useProject((s) => s.project);
  const editor = useRunEditor();
  const working = project ? workingConfig(editor, project) : null;
  const check = useEditorCheck(
    projectTopology(project),
    working,
    editor.edits !== null,
    project?.spec.config ?? NO_PATCH,
  );
  if (!project) return <Empty>Open a project to calibrate it.</Empty>;

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto lg:grid-cols-[minmax(0,1fr)_21rem] lg:overflow-hidden">
      <div className="min-h-0 overflow-y-auto px-1 py-1 lg:pr-5">
        <ParameterEditor project={project} check={check} />
      </div>
      <div className="min-h-0 overflow-y-auto border-line pt-4 lg:border-l lg:pt-1 lg:pl-5">
        <RunPanel project={project} check={check} onRun={(o) => void editor.run(o)} />
      </div>
    </div>
  );
}

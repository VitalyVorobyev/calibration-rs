import { Button, Callout, ErrorBox, Toaster } from "@vitavision/ui";
import { AppShell, StatusBar } from "@vitavision/workbench";
import { Outlet } from "react-router";
import { folderName } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { ProjectPanel } from "./ProjectPanel";
import { ShellHeader } from "./ShellHeader";
import { WorkspaceRail } from "./WorkspaceRail";

/** The app's frame, on the workbench `AppShell`: the header, the workspace
 * rail, the project navigator (while a project is open), the active
 * workspace, and the status line. */
export function Shell() {
  const project = useProject((s) => s.project);
  return (
    <>
      <AppShell
        header={<ShellHeader />}
        rail={<WorkspaceRail />}
        left={project ? <ProjectPanel project={project} /> : undefined}
        leftLabel="Project"
        leftSize={{ defaultSize: 240, minSize: 180, maxSize: "40%", collapsible: true }}
        main={<MainSurface />}
        bottom={<ShellStatusBar />}
        storageKey="calib-shell"
      />
      <Toaster />
    </>
  );
}

/** The active workspace, under the export and project errors. */
function MainSurface() {
  const loadError = useStore((s) => s.loadError);
  const projectError = useProject((s) => s.error);
  const clearProjectError = useProject((s) => s.clearError);
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-3">
      {loadError && <ErrorBox className="mb-2">{loadError}</ErrorBox>}
      {projectError && (
        <Callout
          tone="error"
          title={PROJECT_ERROR_TITLE[projectError.kind]}
          className="mb-2"
          actions={
            <Button size="sm" variant="ghost" onClick={clearProjectError}>
              Dismiss
            </Button>
          }
        >
          {projectError.message}
        </Callout>
      )}
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
    </div>
  );
}

const PROJECT_ERROR_TITLE = {
  not_a_project: "Not a project folder",
  project_exists: "That folder already holds a project",
  unsupported_version: "Unsupported project version",
  invalid_run_name: "Invalid run name",
  run_exists: "A run of that name is already saved",
  io: "Project error",
  parse: "A project file does not parse",
} as const;

/** The status line: the open project and the size of the loaded export
 * (its kind is in the header). */
function ShellStatusBar() {
  const project = useProject((s) => s.project);
  const frames = useStore((s) => s.frames.length);
  const cameras = useStore((s) => s.cameraValues.length);
  const loaded = useStore((s) => s.data !== null);
  return (
    <StatusBar
      start={[
        { label: "project", value: project ? folderName(project.dir) : "none" },
        { label: "runs", value: project ? project.runs.length : undefined },
      ]}
      end={[
        { label: "cameras", value: loaded ? cameras : undefined },
        { label: "frames", value: loaded ? frames : undefined },
      ]}
    />
  );
}

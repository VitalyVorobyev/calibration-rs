import { Button, Callout, ErrorBox, Toaster } from "@vitavision/ui";
import { AppShell, StatusBar } from "@vitavision/workbench";
import { useEffect } from "react";
import { Outlet } from "react-router";
import { PathPickerHost } from "../components/PathPicker/PathPicker";
import type { ErrorKind } from "../api/generated/types";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { ProjectPanel } from "./ProjectPanel";
import { useSession } from "./session";
import { SessionNotice } from "./SessionNotice";
import { ShellHeader } from "./ShellHeader";
import { WorkspaceRail } from "./WorkspaceRail";

/** The app's frame, on the workbench `AppShell`: the header, the workspace
 * rail, the project navigator (while a project is open), the active
 * workspace, and the status line. */
export function Shell() {
  const project = useProject((s) => s.project);
  const start = useSession((s) => s.start);
  useEffect(() => {
    void start();
  }, [start]);
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
      <PathPickerHost />
      <Toaster />
    </>
  );
}

/** The active workspace, under the export and project errors. */
function MainSurface() {
  const loadError = useStore((s) => s.loadError);
  const projectError = useProject((s) => s.error);
  const clearProjectError = useProject((s) => s.clearError);
  const login = useSession((s) => s.status.kind === "login");
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-3">
      <SessionNotice />
      {loadError && <ErrorBox className="mb-2">{loadError}</ErrorBox>}
      {projectError && (
        <Callout
          tone="error"
          title={PROJECT_ERROR_TITLE[projectError.kind] ?? "Project error"}
          className="mb-2"
          actions={
            <>
              {projectError.kind === "changed_on_disk" && (
                <Button size="sm" onClick={() => void useProject.getState().reload()}>
                  Reload
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={clearProjectError}>
                Dismiss
              </Button>
            </>
          }
        >
          {projectError.message}
        </Callout>
      )}
      {!login && (
        <div className="flex min-h-0 flex-1 flex-col">
          <Outlet />
        </div>
      )}
    </div>
  );
}

/** The title of each project error kind; any other kind (a refused path,
 * a failure of the back end) is a "Project error". */
const PROJECT_ERROR_TITLE: Partial<Record<ErrorKind, string>> = {
  not_a_project: "Not a project folder",
  project_exists: "That folder already holds a project",
  unsupported_version: "Unsupported project version",
  ambiguous_project: "Two project files in one folder",
  changed_on_disk: "The project changed on disk: reload it and apply the edit again",
  run_not_found: "The project has no such run",
  import_mismatch: "The export does not belong to this project",
  exclusion: "An exclusion names something the dataset does not have",
  manifest: "The manifest is not valid",
  config: "The configuration is not valid",
  io: "Project error",
  parse: "A project file does not parse",
  forbidden_path: "Outside the folders this back end serves",
};

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
        { label: "project", value: project ? project.name : "none" },
        { label: "runs", value: project ? project.runs.length : undefined },
      ]}
      end={[
        { label: "cameras", value: loaded ? cameras : undefined },
        { label: "frames", value: loaded ? frames : undefined },
      ]}
    />
  );
}

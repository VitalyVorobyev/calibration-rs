import { Button, Callout, Empty, ErrorBox, SkeletonRows, Toaster } from "@vitavision/ui";
import { AppShell } from "@vitavision/workbench";
import { useEffect, type ReactNode } from "react";
import { Link, Outlet, useLocation } from "react-router";
import type { ErrorKind } from "../api/generated/types";
import { PathPickerHost } from "../components/PathPicker/PathPicker";
import { useProject } from "../project/projectStore";
import { useSession } from "../session/session";
import { startSync } from "../session/sync";
import { useStore } from "../store";
import { CommandPalette } from "./CommandPalette";
import { ConnectionStrip } from "./ConnectionStrip";
import { ContextBar } from "./ContextBar";
import { JobTray } from "./JobTray";
import { LoginPanel } from "./LoginPanel";
import { RunList } from "./RunList";
import { ProjectDialogs } from "./ProjectDialogs";
import { parsePlace, placePath, screenById, screenPlace, type Place } from "./screens";
import { ScreenRail } from "./ScreenRail";
import { SettingsDialog } from "./SettingsDialog";
import { ShortcutsDialog } from "./ShortcutsDialog";
import { SCREEN_ID, SkipLink } from "./SkipLink";
import { StatusLine } from "./StatusLine";
import { useShellKeys } from "./useShellKeys";
import { useUrlSync } from "./useUrlSync";

/** The app's frame, on the workbench `AppShell`: the rail of seven screens,
 * the context bar (project, run, its state) with the connection strip under
 * it, the run list beside the screens that read runs, the routed screen, and
 * the job tray over the status line. While the server wants its token the
 * page is the sign-in panel and nothing else. */
export function Shell() {
  const { pathname, search } = useLocation();
  const place = parsePlace(pathname, search);
  const status = useSession((s) => s.status);
  const start = useSession((s) => s.start);
  const project = useProject((s) => s.project);
  const ready = status.kind === "ready";

  useEffect(() => {
    void start();
  }, [start]);
  // The standing watch of the back end exists while there is a session.
  useEffect(() => (ready ? startSync() : undefined), [ready]);
  useUrlSync();
  useShellKeys();

  if (status.kind === "login") {
    return (
      <>
        <LoginPanel error={status.error} />
        <Toaster />
      </>
    );
  }

  const withRuns =
    project !== null && ["calibrate", "inspect", "compare"].includes(place.screen);

  return (
    <>
      <SkipLink />
      <AppShell
        railLabel="Screens"
        rail={<ScreenRail place={place} />}
        header={
          <>
            <ContextBar place={place} />
            <ConnectionStrip />
          </>
        }
        left={withRuns ? <RunList project={project} /> : undefined}
        leftLabel="Runs"
        leftSize={{ defaultSize: 300, minSize: 200, maxSize: "40%", collapsible: true }}
        main={<MainSurface place={place} connecting={status.kind === "connecting"} />}
        bottom={
          <>
            <JobTray place={place} />
            <StatusLine />
          </>
        }
        storageKey="calib-shell"
      />
      <PathPickerHost />
      <ProjectDialogs />
      <SettingsDialog />
      <ShortcutsDialog />
      <CommandPalette />
      {/* Above the status line, which the package's corner would cover. */}
      <Toaster className="bottom-12" />
    </>
  );
}

/** The routed screen, under the errors of loading a run or a project. A
 * screen that needs a project says so, and links to the Library, while
 * none is open. */
function MainSurface({ place, connecting }: { place: Place; connecting: boolean }) {
  const loadError = useStore((s) => s.loadError);
  const project = useProject((s) => s.project);
  const projectError = useProject((s) => s.error);
  const clearProjectError = useProject((s) => s.clearError);
  const screen = screenById(place.screen);
  const needsProject = screen.needsProject && screen.built && project === null;

  return (
    <div
      id={SCREEN_ID}
      tabIndex={-1}
      className="flex h-full min-h-0 flex-col overflow-hidden p-3 outline-none"
    >
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
      {/* The Library draws its own title; the other screens are named here. */}
      {place.screen !== "library" && <h1 className="sr-only">{screen.label}</h1>}
      <div className="flex min-h-0 flex-1 flex-col">
        {connecting ? (
          <Loading>Connecting…</Loading>
        ) : needsProject ? (
          place.project && projectError === null ? (
            <Loading>Opening the project…</Loading>
          ) : (
            <NeedsProject place={place} />
          )
        ) : (
          <Outlet />
        )}
      </div>
    </div>
  );
}

function Loading({ children }: { children: ReactNode }) {
  return (
    <div role="status" aria-busy className="mx-auto w-full max-w-sm pt-16">
      <p className="mb-3 text-center text-xs text-fg-muted">{children}</p>
      <SkeletonRows rows={3} />
    </div>
  );
}

function NeedsProject({ place }: { place: Place }) {
  return (
    <div className="grid h-full place-items-center">
      <Empty
        action={
          <Button asChild size="sm" variant="secondary">
            <Link
              to={placePath(
                screenPlace("library", { ...place, project: null, run: null }),
              )}
            >
              Open from the Library
            </Link>
          </Button>
        }
      >
        {screenById(place.screen).label} works on a project. Open one first.
      </Empty>
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

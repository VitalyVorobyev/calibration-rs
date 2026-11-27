import { Badge, Button, Empty, PageHeader } from "@vitavision/ui";
import { FolderOpen, FolderPlus } from "lucide-react";
import {
  openProject,
  newProjectFromFolder,
  openSample,
} from "../../project/projectActions";
import { folderName } from "../../project/projectModel";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { userRoots, useSession } from "../../session/session";
import { AddRootButton } from "../../shell/AddRootButton";

/** The Library, until its own screen is built: the way into a project. On
 * the desktop's first run it is the one prompt, "Add a folder"; with a
 * folder it lists the recent projects and the Quick Start samples. */
export function LibraryScreen() {
  const status = useSession((s) => s.status);
  const recent = useProject((s) => s.recent);
  const readOnly = useReadOnly();
  if (status.kind !== "ready") {
    // No back end to list: the connection strip says why.
    return (
      <div className="max-w-3xl p-6">
        <PageHeader title="Library" />
      </div>
    );
  }
  const { info } = status;
  const firstRun = info.roots_editable && userRoots(info).length === 0;

  return (
    <div className="flex h-full w-full max-w-3xl flex-col gap-6 overflow-y-auto p-6">
      <PageHeader
        title="Library"
        meta={
          firstRun
            ? "No folders yet"
            : `${userRoots(info).length} ${userRoots(info).length === 1 ? "folder" : "folders"}`
        }
      />

      {firstRun && (
        <Empty action={<AddRootButton />}>
          Add a folder that holds your projects or datasets. Calibration Studio reads
          nothing else.
        </Empty>
      )}

      {!firstRun && (
        <>
          <section aria-label="Open" className="flex flex-wrap items-center gap-2">
            <Button variant="primary" size="sm" onClick={() => void openProject()}>
              <FolderOpen aria-hidden className="size-3.5" />
              Open project…
            </Button>
            <Button
              size="sm"
              disabled={readOnly}
              onClick={() => void newProjectFromFolder()}
            >
              <FolderPlus aria-hidden className="size-3.5" />
              New from a dataset folder…
            </Button>
          </section>

          <section aria-labelledby="recent-title" className="flex flex-col gap-2">
            <h2
              id="recent-title"
              className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase"
            >
              Recent projects
            </h2>
            {recent.length === 0 ? (
              <p className="text-sm text-fg-muted">
                None yet. Open a project, or start one from a dataset folder.
              </p>
            ) : (
              <ul className="flex flex-col divide-y divide-line rounded-panel border border-line bg-surface">
                {recent.map((dir) => (
                  <li key={dir}>
                    <button
                      type="button"
                      onClick={() => void openProject(dir)}
                      className="flex w-full items-baseline gap-3 px-3 py-2 text-left outline-none hover:bg-raised focus-visible:bg-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal"
                    >
                      <span className="shrink-0 text-sm font-medium whitespace-nowrap text-fg">
                        {folderName(dir)}
                      </span>
                      <span className="truncate font-mono text-[11px] text-fg-muted">
                        {dir}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}

      {info.samples.length > 0 && (
        <section aria-labelledby="samples-title" className="flex flex-col gap-2">
          <h2
            id="samples-title"
            className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase"
          >
            Quick start
          </h2>
          <ul className="flex flex-col divide-y divide-line rounded-panel border border-line bg-surface">
            {info.samples.map((sample) => (
              <li key={sample.id} className="flex items-center gap-3 px-3 py-2">
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="text-sm font-medium text-fg">{sample.id}</span>
                  <span className="truncate text-xs text-fg-muted">
                    {sample.description}
                  </span>
                </div>
                <Badge>{sample.topology}</Badge>
                <Button
                  size="sm"
                  disabled={readOnly}
                  onClick={() => void openSample(sample.id)}
                >
                  Open sample
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

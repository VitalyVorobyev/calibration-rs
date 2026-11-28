import {
  Button,
  Callout,
  DensityProvider,
  Empty,
  PageHeader,
  SkeletonRows,
} from "@vitavision/ui";
import { FileInput, FolderOpen, FolderPlus, RefreshCw } from "lucide-react";
import { useCallback, useMemo } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { useProjectDialog } from "../../project/dialogs";
import { openProject } from "../../project/projectActions";
import { useLibraryScan } from "../../project/useLibraryScan";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { userRoots, useSession } from "../../session/session";
import { AddRootButton } from "../../shell/AddRootButton";
import { placePath } from "../../shell/screens";
import { deviceParamsOf } from "../device/deviceModel";
import {
  containerOf,
  countLine,
  groupLibrary,
  isFiltering,
  matches,
  NO_DEVICE,
  NO_DEVICE_ID,
  NO_FILTER,
  openScreen,
  projectNodeId,
  projectRow,
  relativeToRoots,
  rowsOf,
  type LibraryGroups,
  type ProjectRow,
} from "./libraryModel";
import { fromParams, toParams } from "./libraryUrl";
import { LibraryToolbar } from "./LibraryToolbar";
import { LibraryTree } from "./LibraryTree";
import { ProjectTable } from "./ProjectTable";
import { QuickStart } from "./QuickStart";

/** The Library: every project under the back end's folders, grouped by
 * device type and device, with how its latest calibration stands. Find one
 * (search and filters, in the address so a link reopens them), open it, or
 * start one: from a folder of images, a manifest, or an export calibrated
 * elsewhere. */
export function LibraryScreen() {
  const status = useSession((s) => s.status);
  const readOnly = useReadOnly();
  const show = useProjectDialog((s) => s.show);
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const scan = useLibraryScan(true);
  const openDir = useProject((s) => s.project?.dir ?? null);

  const view = useMemo(() => fromParams(params), [params]);
  const setView = useCallback(
    (next: typeof view) => setParams((p) => toParams(p, next), { replace: true }),
    [setParams],
  );

  const allRows = useMemo(() => (scan.data?.projects ?? []).map(projectRow), [scan.data]);
  const rows = useMemo(
    () => allRows.filter((r) => matches(r, view.filter)),
    [allRows, view.filter],
  );
  const groups = useMemo(() => groupLibrary(rows), [rows]);
  const topologies = useMemo(
    () =>
      [...new Set(allRows.map((r) => r.entry.topology).filter((t) => t != null))].sort(),
    [allRows],
  );

  if (status.kind !== "ready") {
    // No back end to list: the connection strip says why.
    return (
      <div className="p-6">
        <PageHeader title="Library" />
      </div>
    );
  }
  const { info } = status;
  const roots = userRoots(info);
  const firstRun = info.roots_editable && roots.length === 0;

  const selected =
    view.selected !== null && rowsOf(groups, view.selected).length > 0
      ? view.selected
      : null;
  const container = containerOf(groups, selected);
  const tableRows = rowsOf(groups, container);
  const activeDir =
    selected?.startsWith("project:") === true
      ? (tableRows.find((r) => projectNodeId(r.entry.dir) === selected)?.entry.dir ??
        null)
      : openDir;

  const goOpen = async (row: ProjectRow, screen = openScreen(row)) => {
    if (!(await openProject(row.entry.dir))) return;
    void navigate(
      placePath({
        screen,
        tab: screen === "inspect" ? "frames" : null,
        project: row.entry.dir,
        run: null,
      }),
    );
  };

  const title = tableTitle(groups, container);
  const deviceParams = deviceParamsOf(groups, container);
  const deviceHref = deviceParams
    ? `${placePath({ screen: "device", tab: null, project: openDir, run: null })}${openDir ? "&" : "?"}${deviceParams.toString()}`
    : null;
  const filtering = isFiltering(view.filter);
  const loading = scan.data === undefined && scan.error === null;
  const empty = scan.data !== undefined && allRows.length === 0;

  return (
    <DensityProvider value="comfortable">
      <div className="flex h-full min-h-0 flex-col gap-4 overflow-hidden p-3">
        <PageHeader
          title="Library"
          meta={
            firstRun ? (
              "No folders yet"
            ) : (
              <>
                <span className="font-mono">{countLine(allRows)}</span>
                <span aria-hidden>·</span>
                <span>
                  {roots.length} {roots.length === 1 ? "folder" : "folders"}
                </span>
              </>
            )
          }
          actions={
            !firstRun && (
              <>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-label="Rescan the folders"
                  title="Rescan the folders"
                  onClick={() => scan.refetch()}
                >
                  <RefreshCw
                    aria-hidden
                    className={`size-3.5 ${scan.loading ? "animate-spin" : ""}`}
                  />
                </Button>
                <Button size="sm" variant="secondary" onClick={() => void openProject()}>
                  <FolderOpen aria-hidden className="size-3.5" />
                  Open…
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={readOnly}
                  onClick={() => show({ kind: "import" })}
                >
                  <FileInput aria-hidden className="size-3.5" />
                  Import export
                </Button>
                <Button
                  size="sm"
                  variant="primary"
                  disabled={readOnly}
                  onClick={() => show({ kind: "new" })}
                >
                  <FolderPlus aria-hidden className="size-3.5" />
                  New project
                </Button>
              </>
            )
          }
        />

        {firstRun && (
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 pt-6">
            <Empty action={<AddRootButton />}>
              Add a folder that holds your projects or datasets. Calibration Studio reads
              nothing else.
            </Empty>
            <QuickStart samples={info.samples} />
          </div>
        )}

        {!firstRun && (
          <>
            {!empty && (
              <LibraryToolbar
                filter={view.filter}
                onChange={(filter) => setView({ ...view, filter })}
                topologies={topologies}
              />
            )}
            {scan.error && (
              <Callout
                tone="error"
                title="The folders could not be scanned"
                actions={
                  <Button size="sm" onClick={() => scan.refetch()}>
                    Try again
                  </Button>
                }
              >
                {scan.error.message}
              </Callout>
            )}

            {loading ? (
              <div
                role="status"
                aria-busy
                aria-label="Scanning the folders"
                className="max-w-md"
              >
                <SkeletonRows rows={5} />
              </div>
            ) : empty ? (
              <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 pt-6">
                <Empty
                  action={
                    <div className="flex gap-2">
                      <Button
                        variant="primary"
                        size="sm"
                        disabled={readOnly}
                        onClick={() => show({ kind: "new" })}
                      >
                        <FolderPlus aria-hidden className="size-3.5" />
                        New project
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => void openProject()}
                      >
                        Open…
                      </Button>
                    </div>
                  }
                >
                  No project under {roots.length === 1 ? "this folder" : "these folders"}{" "}
                  yet. Start one from a folder of images, or open a Quick Start sample.
                </Empty>
                <QuickStart samples={info.samples} />
              </div>
            ) : (
              <div className="grid min-h-0 flex-1 grid-cols-[17rem_minmax(0,1fr)] gap-3">
                <nav
                  aria-label="Library"
                  className="min-h-0 overflow-y-auto rounded-panel border border-line bg-surface p-1.5"
                >
                  <button
                    type="button"
                    aria-current={selected === null ? "true" : undefined}
                    onClick={() => setView({ ...view, selected: null })}
                    className="mb-1 flex w-full items-center justify-between rounded-control px-2 py-1.5 text-left text-xs font-medium text-fg-muted outline-none hover:bg-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal aria-[current=true]:bg-signal/12 aria-[current=true]:text-fg"
                  >
                    All projects
                    <span className="font-mono text-[11px] text-fg-subtle">
                      {rows.length}
                    </span>
                  </button>
                  {rows.length > 0 && (
                    <LibraryTree
                      groups={groups}
                      selected={selected}
                      expandAll={filtering}
                      onSelect={(id) => setView({ ...view, selected: id })}
                    />
                  )}
                </nav>

                <section
                  aria-label={title.label}
                  className="flex min-h-0 min-w-0 flex-col gap-3 overflow-y-auto"
                >
                  <header className="flex flex-wrap items-baseline justify-between gap-x-4">
                    <h2 className="text-base font-semibold tracking-tight text-fg">
                      {title.label}
                    </h2>
                    <span className="flex items-center gap-3">
                      {deviceHref && (
                        <Button asChild size="sm" variant="secondary">
                          <Link to={deviceHref}>Device history</Link>
                        </Button>
                      )}
                      <span className="font-mono text-xs text-fg-muted">
                        {countLine(tableRows)}
                      </span>
                    </span>
                  </header>
                  {tableRows.length === 0 ? (
                    <Empty
                      action={
                        filtering ? (
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => setView({ ...view, filter: NO_FILTER })}
                          >
                            Clear the filters
                          </Button>
                        ) : undefined
                      }
                    >
                      {filtering ? "No project matches the search." : "No project here."}
                    </Empty>
                  ) : (
                    <div className="shrink-0 overflow-x-auto rounded-panel border border-line bg-surface">
                      <ProjectTable
                        rows={tableRows}
                        roots={roots}
                        query={view.filter.query}
                        activeDir={activeDir}
                        onSelect={(row) =>
                          setView({ ...view, selected: projectNodeId(row.entry.dir) })
                        }
                        onOpen={(row) => void goOpen(row)}
                        onCalibrate={(row) => void goOpen(row, "calibrate")}
                      />
                    </div>
                  )}
                  {scan.data && scan.data.failures.length > 0 && (
                    <Failures failures={scan.data.failures} roots={roots} />
                  )}
                  <QuickStart samples={info.samples} collapsed />
                </section>
              </div>
            )}
          </>
        )}
      </div>
    </DensityProvider>
  );
}

/** The heading of the table: what the selected node is. */
function tableTitle(groups: LibraryGroups, container: string | null): { label: string } {
  if (container === null) return { label: "All projects" };
  if (container === NO_DEVICE_ID) return { label: NO_DEVICE };
  for (const product of groups.products) {
    if (product.id === container) return { label: product.label };
    const device = product.devices.find((d) => d.id === container);
    if (device)
      return { label: `${product.label} · ${device.device.label ?? device.device.id}` };
  }
  return { label: "Projects" };
}

export function Failures({
  failures,
  roots,
}: {
  failures: { path: string; message: string }[];
  roots: readonly string[];
}) {
  return (
    <details className="group rounded-control border border-warn/30 bg-warn/8 px-3 py-2 text-sm">
      <summary className="cursor-pointer text-fg outline-none focus-visible:outline-2 focus-visible:outline-signal">
        {failures.length} {failures.length === 1 ? "folder" : "folders"} could not be read
      </summary>
      <ul className="mt-2 flex flex-col gap-1.5">
        {failures.map((f) => (
          <li key={f.path} className="flex flex-col text-xs">
            <span className="font-mono text-fg">{relativeToRoots(f.path, roots)}</span>
            <span className="text-fg-muted">{f.message}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

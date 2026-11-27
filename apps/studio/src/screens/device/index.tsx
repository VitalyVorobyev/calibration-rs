import {
  Button,
  Callout,
  DensityProvider,
  Empty,
  PageHeader,
  SkeletonRows,
  Switch,
  Tabs,
} from "@vitavision/ui";
import { RefreshCw } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useProjectDialog } from "../../project/dialogs";
import { useProject } from "../../project/projectStore";
import { useLibraryScan } from "../../project/useLibraryScan";
import { useReadOnly } from "../../session/jobs";
import { userRoots, useSession } from "../../session/session";
import { placePath } from "../../shell/screens";
import { groupLibrary, projectRow } from "../library/libraryModel";
import { Failures } from "../library/index";
import {
  fromParams,
  historyRows,
  representativeOf,
  targetOfDevice,
  toParams,
  typeOf,
  unitRows,
  type DeviceTab,
  type DeviceTarget,
  type DeviceView,
  type HistoryRow,
  type UnitRow,
} from "./deviceModel";
import { DevicePicker } from "./DevicePicker";
import { HistoryTable } from "./HistoryTable";
import { SeedPanel } from "./SeedPanel";
import { runMetaMap } from "./seriesModel";
import { TrendsTab } from "./TrendsTab";
import { UnitsTable } from "./UnitsTable";
import { useDeviceHistory, useDeviceSeries } from "./useDeviceData";

/** The Device screen: how one physical device's calibration has behaved
 * across its projects (or how a device type's units compare), and the
 * starting values to seed the next calibration with. The device or type is
 * in the address, so a link reopens it; with none there, it is the open
 * project's device. */
export function DeviceScreen() {
  const status = useSession((s) => s.status);
  const readOnly = useReadOnly();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const project = useProject((s) => s.project);
  const showDialog = useProjectDialog((s) => s.show);
  const scan = useLibraryScan(true);
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [lastDevice, setLastDevice] = useState<string | null>(null);

  const view = useMemo(() => fromParams(params), [params]);
  const setView = useCallback(
    (next: DeviceView) => setParams((p) => toParams(p, next), { replace: true }),
    [setParams],
  );
  const groups = useMemo(
    () => groupLibrary((scan.data?.projects ?? []).map(projectRow)),
    [scan.data],
  );
  const target: DeviceTarget | null = view.target ?? targetOfDevice(project?.spec.device);
  const typeTarget = target ? typeOf(target) : null;

  const history = useDeviceHistory(target);
  const series = useDeviceSeries(target, view.starredOnly);
  const typeHistory = useDeviceHistory(view.tab === "seed" ? typeTarget : null);

  const views = useMemo(() => history.data?.devices ?? [], [history.data]);
  const meta = useMemo(() => runMetaMap(views), [views]);
  const activeKey = hover ?? selected;

  if (status.kind !== "ready") {
    return (
      <div className="p-6">
        <PageHeader title="Device" />
      </div>
    );
  }
  const roots = userRoots(status.info);

  const choose = (next: DeviceTarget) => {
    if (next.id) setLastDevice(next.id);
    setSelected(null);
    setView({ ...view, target: next });
  };
  const openRun = (row: HistoryRow) =>
    void navigate(
      placePath({
        screen: "inspect",
        tab: "frames",
        project: row.run.project,
        run: row.run.id,
      }),
    );
  const openUnit = (row: UnitRow) =>
    choose({
      scope: "device",
      product: row.device.product ?? null,
      id: row.device.id,
    });

  const isType = target?.scope === "type";
  const loading = target !== null && history.data === undefined && history.error === null;
  const none = target !== null && history.data !== undefined && views.length === 0;
  const first = views[0] ?? null;
  const rows = first && !isType ? historyRows(first, view.starredOnly) : [];
  const runCount = views.reduce((n, v) => n + v.runs.length, 0);
  const starCount = views.reduce((n, v) => n + v.runs.filter((r) => r.starred).length, 0);
  const projectCount = new Set(views.flatMap((v) => v.runs.map((r) => r.project))).size;
  const representative = first ? representativeOf(first) : null;

  const tabs: { id: DeviceTab; label: string }[] = [
    { id: "history", label: isType ? "Units" : "History" },
    { id: "trends", label: "Trends" },
    { id: "seed", label: "Seed" },
  ];
  const failures = [
    ...(history.data?.failures ?? []),
    ...(series.data?.failures ?? []),
  ].filter((f, i, all) => all.findIndex((g) => g.path === f.path) === i);

  return (
    <DensityProvider value="comfortable">
      <div className="flex h-full min-h-0 flex-col gap-3 overflow-hidden p-3">
        <PageHeader
          title="Device"
          meta={
            target && first ? (
              <>
                {isType ? (
                  <span>
                    <span className="font-mono">{target.product}</span> · {views.length}{" "}
                    {views.length === 1 ? "unit" : "units"}
                  </span>
                ) : (
                  <span>
                    {first.device.product && (
                      <span className="font-mono">{first.device.product} · </span>
                    )}
                    <span className="font-mono">
                      {first.device.label ?? first.device.id}
                    </span>
                    {first.device.label && (
                      <span className="font-mono text-fg-subtle"> {first.device.id}</span>
                    )}
                  </span>
                )}
                <span aria-hidden>·</span>
                <span className="font-mono">
                  {runCount} {runCount === 1 ? "run" : "runs"} ({starCount} starred) in{" "}
                  {projectCount} {projectCount === 1 ? "project" : "projects"}
                </span>
              </>
            ) : undefined
          }
          actions={
            <Button
              size="sm"
              variant="ghost"
              aria-label="Read the history again"
              title="Read the history again"
              onClick={() => {
                scan.refetch();
                history.refetch();
                series.refetch();
              }}
            >
              <RefreshCw aria-hidden className="size-3.5" />
            </Button>
          }
        />

        <DevicePicker
          groups={groups}
          target={target}
          lastDevice={lastDevice ?? target?.id ?? null}
          onChange={choose}
        />

        {target === null ? (
          <div className="mx-auto flex max-w-xl flex-col gap-4 pt-6">
            <Empty
              action={
                project && !readOnly ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => showDialog({ kind: "device", dir: project.dir })}
                  >
                    Set device…
                  </Button>
                ) : undefined
              }
            >
              {project
                ? `${project.name} names no device. Set one to see its history here, or choose a device above.`
                : "Choose a device above, or open a project that names one. A project names its device with Set device."}
            </Empty>
          </div>
        ) : (
          <>
            {history.error && (
              <Callout
                tone="error"
                title="The history could not be read"
                actions={
                  <Button size="sm" onClick={() => history.refetch()}>
                    Try again
                  </Button>
                }
              >
                {history.error.message}
              </Callout>
            )}
            {loading ? (
              <div
                role="status"
                aria-busy
                aria-label="Reading the history"
                className="max-w-md"
              >
                <SkeletonRows rows={5} />
              </div>
            ) : none ? (
              <Empty>
                No project under the folders names{" "}
                {isType ? `the type ${target.product}` : `the device ${target.id}`}. A
                project names its device with Set device.
              </Empty>
            ) : history.data !== undefined ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <Tabs
                    label="Device"
                    idPrefix="device"
                    active={view.tab}
                    items={tabs}
                    onSelect={(tab) => setView({ ...view, tab })}
                  />
                  {view.tab !== "seed" && (
                    <Switch
                      checked={view.starredOnly}
                      label="Starred runs only"
                      onCheckedChange={(starredOnly) => setView({ ...view, starredOnly })}
                    />
                  )}
                </div>
                <div
                  role="tabpanel"
                  id={`device-panel-${view.tab}`}
                  aria-labelledby={`device-tab-${view.tab}`}
                  className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pb-2 [&>*]:shrink-0"
                >
                  {view.tab === "history" &&
                    (isType ? (
                      <UnitsTable
                        rows={unitRows(views)}
                        activeKey={activeKey}
                        onHover={setHover}
                        onOpenUnit={openUnit}
                      />
                    ) : (
                      <>
                        <p className="text-xs text-fg-muted">
                          The representative run is the device&apos;s latest starred run,
                          else its latest passing run calibrated in a project
                          {representative
                            ? `: ${representative.id} of ${representative.project_name}`
                            : ""}
                          .
                        </p>
                        <HistoryTable
                          rows={rows}
                          activeKey={activeKey}
                          onHover={setHover}
                          onSelect={setSelected}
                          onOpen={openRun}
                        />
                      </>
                    ))}
                  {view.tab === "trends" && (
                    <TrendsTab
                      series={series.data}
                      error={series.error}
                      rows={rows}
                      showRuns={!isType}
                      meta={meta}
                      xMode={view.xMode}
                      onXMode={(xMode) => setView({ ...view, xMode })}
                      activeKey={activeKey}
                      onHover={setHover}
                      onSelect={setSelected}
                      onOpen={openRun}
                      starredOnly={view.starredOnly}
                    />
                  )}
                  {view.tab === "seed" && (
                    <SeedPanel
                      key={`${target.scope}:${target.product}:${target.id}:${typeHistory.data ? "t" : "-"}`}
                      views={views}
                      units={typeHistory.data?.devices ?? []}
                      hasType={typeTarget !== null}
                    />
                  )}
                  {failures.length > 0 && <Failures failures={failures} roots={roots} />}
                </div>
              </>
            ) : null}
          </>
        )}
      </div>
    </DensityProvider>
  );
}

import {
  Callout,
  Empty,
  SegmentedControl,
  Select,
  Tabs,
  toneColor,
} from "@vitavision/ui";
import { useMemo, useState } from "react";
import type { DeviceSeriesSet } from "../../api/generated/types";
import type { XMode } from "./deviceModel";
import {
  camerasIn,
  chartSpecs,
  GROUP_TABS,
  groupsPresent,
  MARK_KEY,
  type GroupTab,
  type RunMeta,
} from "./seriesModel";
import { TrendChart } from "./TrendChart";

/** Every parameter over the runs as small charts, by group and camera.
 * Plainly what the library has: no threshold, no verdict on drift. */
export function TrendsPanel({
  series,
  meta,
  xMode,
  onXMode,
  hoverKey,
  onHover,
  onPick,
  starredOnly,
}: {
  series: DeviceSeriesSet;
  meta: ReadonlyMap<string, RunMeta>;
  xMode: XMode;
  onXMode: (mode: XMode) => void;
  hoverKey: string | null;
  onHover: (key: string | null) => void;
  onPick: (key: string) => void;
  starredOnly: boolean;
}) {
  const sets = series.devices;
  const groups = useMemo(() => groupsPresent(sets), [sets]);
  const [chosenGroup, setGroup] = useState<GroupTab | null>(null);
  const [chosenCamera, setCamera] = useState<string | null>(null);
  const group =
    chosenGroup !== null && groups.includes(chosenGroup)
      ? chosenGroup
      : (groups[0] ?? null);
  const cameras = useMemo(() => (group ? camerasIn(sets, group) : []), [sets, group]);
  const camera =
    chosenCamera !== null && cameras.includes(chosenCamera)
      ? chosenCamera
      : (cameras[0] ?? null);
  const specs = useMemo(
    () => (group ? chartSpecs(sets, group, camera, xMode) : []),
    [sets, group, camera, xMode],
  );
  const unitIndex = useMemo(
    () => new Map(sets.map((s, i) => [s.device.id, i] as const)),
    [sets],
  );
  const runs = sets.reduce((n, s) => n + s.runs.length, 0);

  if (sets.length === 0) {
    return <Empty>No project names this device, so there is nothing to plot.</Empty>;
  }
  if (runs === 0) {
    return (
      <Empty>
        {starredOnly
          ? "No starred run. Star a run in Calibrate or Inspect, or show every run."
          : "This device has no run yet."}
      </Empty>
    );
  }
  if (group === null) {
    return <Empty>None of the runs in view has a parameter to plot.</Empty>;
  }

  const unavailable = sets.filter((s) => s.statistics.status === "unavailable");
  const leftOut = sets.flatMap((s) =>
    s.statistics.left_out.map((l) => ({
      unit: s.device.id,
      run: s.runs[l.run],
      reason: l.reason.message,
    })),
  );

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <Tabs
          label="Parameter group"
          idPrefix="trend-group"
          active={group}
          items={GROUP_TABS.filter((g) => groups.includes(g.id))}
          onSelect={setGroup}
        />
        <div className="flex flex-wrap items-end gap-3">
          {cameras.length > 0 && (
            <label className="flex flex-col gap-1 text-xs text-fg-muted">
              Camera
              <Select
                aria-label="Camera"
                className="min-w-28"
                value={camera ?? ""}
                options={cameras.map((id) => ({ value: id, label: id }))}
                onValueChange={setCamera}
              />
            </label>
          )}
          <div className="flex flex-col gap-1 text-xs text-fg-muted">
            x axis
            <SegmentedControl
              aria-label="x axis"
              value={xMode}
              options={[
                { value: "order", label: "Run order" },
                { value: "time", label: "Recorded time" },
              ]}
              onValueChange={(v) => onXMode(v === "time" ? "time" : "order")}
            />
          </div>
        </div>
      </div>

      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-fg-muted">
        Ticks on the axis:
        {MARK_KEY.map((m) => (
          <span key={m.text} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-2.5 w-0.5"
              style={{ background: toneColor(m.tone) }}
            />
            {m.text}
          </span>
        ))}
        <span>Band: ±1σ. A run without the value is a gap.</span>
      </p>

      {unavailable.length > 0 && (
        <Callout tone="info" title="No spread statistics">
          {unavailable.map((s) => (
            <span key={s.device.id} className="block">
              {sets.length > 1 ? `${s.device.id}: ` : ""}
              {s.statistics.status === "unavailable" ? s.statistics.reason.message : ""}
            </span>
          ))}
        </Callout>
      )}
      {leftOut.length > 0 && (
        <details className="rounded-control border border-line bg-surface px-3 py-2 text-xs">
          <summary className="cursor-pointer text-fg-muted outline-none focus-visible:outline-2 focus-visible:outline-signal">
            {leftOut.length} {leftOut.length === 1 ? "run is" : "runs are"} left out of
            the spread
          </summary>
          <ul className="mt-2 flex flex-col gap-1">
            {leftOut.map((l) => (
              <li
                key={`${l.unit}/${l.run?.project}/${l.run?.id}`}
                className="flex flex-col"
              >
                <span className="font-mono text-fg">
                  {sets.length > 1 ? `${l.unit} · ` : ""}
                  {l.run?.id ?? "run"}
                </span>
                <span className="text-fg-muted">{l.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {specs.length === 0 ? (
        <Empty>No parameter of this group for {camera ?? "the device"}.</Empty>
      ) : (
        <div
          className={`grid gap-3 ${
            // Dates are wider than run numbers: wider charts keep their ticks apart.
            xMode === "time"
              ? "grid-cols-[repeat(auto-fill,minmax(27rem,1fr))]"
              : "grid-cols-[repeat(auto-fill,minmax(19rem,1fr))]"
          }`}
        >
          {specs.map((spec) => (
            <TrendChart
              key={spec.key}
              spec={spec}
              xMode={xMode}
              meta={meta}
              unitIndex={unitIndex}
              hoverKey={hoverKey}
              onHover={onHover}
              onPick={onPick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

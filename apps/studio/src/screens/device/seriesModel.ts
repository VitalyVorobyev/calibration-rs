/** `device_series` shaped for the trend charts: the parameter groups and
 * cameras to choose from, one chart per parameter with a line per unit, a
 * missing value a gap (never a zero), σ as a band, the runs worth marking,
 * the linked selection of a hovered run, and the library's spread beside
 * each chart. Pure. */
import { seriesColour, type Marker, type Series } from "@vitavision/charts";
import type {
  DeviceHistoryView,
  DeviceSeries,
  ParameterSeries,
  SeedSpread,
  SeriesGroup,
  SigmaBasis,
} from "../../api/generated/types";
import { fixed, significant } from "../../lib/format";
import { dateOf } from "../library/libraryModel";
import { runKey, type XMode } from "./deviceModel";

export type GroupTab =
  "intrinsics" | "distortion" | "sensor" | "mount" | "handeye" | "laser_plane";

export const GROUP_TABS: { id: GroupTab; label: string }[] = [
  { id: "intrinsics", label: "Intrinsics" },
  { id: "distortion", label: "Distortion" },
  { id: "sensor", label: "Sensor" },
  { id: "mount", label: "Rig poses" },
  { id: "handeye", label: "Hand-eye" },
  { id: "laser_plane", label: "Laser planes" },
];

/** The tab a series group is listed under (the projection's parameters sit
 * with the intrinsics). */
export function groupTab(group: SeriesGroup): GroupTab {
  return group === "projection" ? "intrinsics" : group;
}

/** The groups that have a parameter, in the tabs' order. */
export function groupsPresent(sets: readonly DeviceSeries[]): GroupTab[] {
  const present = new Set(
    sets.flatMap((s) => s.parameters.map((p) => groupTab(p.group))),
  );
  return GROUP_TABS.map((g) => g.id).filter((id) => present.has(id));
}

/** The cameras that have a parameter in `tab`, in first-seen order. */
export function camerasIn(sets: readonly DeviceSeries[], tab: GroupTab): string[] {
  const ids: string[] = [];
  for (const set of sets) {
    for (const p of set.parameters) {
      if (groupTab(p.group) === tab && p.camera != null && !ids.includes(p.camera)) {
        ids.push(p.camera);
      }
    }
  }
  return ids;
}

/** What a run is, apart from its values: for the marks on the axis. */
export interface RunMeta {
  starred: boolean;
  representative: boolean;
  /** `run-003` or `run-003 · tight loss`. */
  name: string;
}

/** The meta of every run of `histories`, by `runKey`. */
export function runMetaMap(
  histories: readonly DeviceHistoryView[],
): Map<string, RunMeta> {
  const map = new Map<string, RunMeta>();
  for (const view of histories) {
    for (const run of view.runs) {
      map.set(runKey(run.project, run.id), {
        starred: run.starred,
        representative: run.representative,
        name: run.label ? `${run.id} · ${run.label}` : run.id,
      });
    }
  }
  return map;
}

/** A run's point on a line. */
export interface Entry {
  key: string;
  x: number;
  /** `NaN` where the run lacks the parameter: a gap. */
  y: number;
  /** The band's edges: `NaN` without a σ. */
  lower: number;
  upper: number;
  atBound: boolean;
}

/** What is said beside a chart about the spread of its values. */
export type StatsView =
  { kind: "spread"; spread: SeedSpread; runs: number } | { kind: "none"; reason: string };

/** One unit's values of one parameter. */
export interface Line {
  /** The unit's id. */
  unit: string;
  entries: Entry[];
  /** Runs with no value of it. */
  missing: number;
  /** Runs whose value has no σ. */
  withoutSigma: number;
  /** Runs that cannot be placed on a time axis (no recorded time). */
  undated: number;
  stats: StatsView;
}

/** One parameter's chart. */
export interface ChartSpec {
  key: string;
  name: string;
  unit: string;
  sigmaOf: SigmaBasis;
  lines: Line[];
}

function statsOf(set: DeviceSeries, parameter: ParameterSeries): StatsView {
  const stats = set.statistics;
  if (stats.status === "unavailable")
    return { kind: "none", reason: stats.reason.message };
  if (parameter.spread == null) {
    return {
      kind: "none",
      reason:
        "The library's spread of this value is over the seed's own parametrization, not these values: see the Seed tab.",
    };
  }
  const spread = stats.spread.find((s) => s.label === parameter.spread);
  return spread
    ? { kind: "spread", spread, runs: stats.runs.length }
    : {
        kind: "none",
        reason: "No spread: not every run that makes a seed has this value.",
      };
}

function lineOf(set: DeviceSeries, parameter: ParameterSeries, xMode: XMode): Line {
  const entries: Entry[] = [];
  let missing = 0;
  let withoutSigma = 0;
  let undated = 0;
  set.runs.forEach((run, i) => {
    const point = parameter.points[i] ?? null;
    if (point === null) missing += 1;
    // Run numbers count from 1, as the history table's # does.
    const x = xMode === "order" ? run.order + 1 : run.timestamp;
    if (x == null) {
      undated += 1;
      return;
    }
    const sigma = point?.sigma.kind === "estimated" ? point.sigma.sigma : null;
    if (point !== null && sigma === null) withoutSigma += 1;
    entries.push({
      key: runKey(run.project, run.id),
      x,
      y: point?.value ?? Number.NaN,
      lower: point !== null && sigma !== null ? point.value - sigma : Number.NaN,
      upper: point !== null && sigma !== null ? point.value + sigma : Number.NaN,
      atBound: point?.at_bound ?? false,
    });
  });
  return {
    unit: set.device.id,
    entries,
    missing,
    withoutSigma,
    undated,
    stats: statsOf(set, parameter),
  };
}

/** One chart per parameter of `tab` (of `camera`, for a group that has
 * cameras), each with a line per unit that has it. */
export function chartSpecs(
  sets: readonly DeviceSeries[],
  tab: GroupTab,
  camera: string | null,
  xMode: XMode,
): ChartSpec[] {
  const specs = new Map<string, ChartSpec>();
  for (const set of sets) {
    for (const p of set.parameters) {
      if (groupTab(p.group) !== tab) continue;
      if (p.camera != null && camera !== null && p.camera !== camera) continue;
      let spec = specs.get(p.key);
      if (!spec) {
        spec = { key: p.key, name: p.name, unit: p.unit, sigmaOf: p.sigma_of, lines: [] };
        specs.set(p.key, spec);
      }
      spec.lines.push(lineOf(set, p, xMode));
    }
  }
  return [...specs.values()];
}

/** A line as the chart takes it: gaps where a run lacks the value, the σ
 * band where it has one. */
export function chartSeries(
  line: Line,
  index: number,
  name: string,
  basis: SigmaBasis,
): Series {
  // No band, and no legend entry for one, when no run has a σ.
  const banded = line.entries.some((e) => Number.isFinite(e.lower));
  return {
    name,
    colour: seriesColour(index),
    points: line.entries.map((e) => ({ x: e.x, y: e.y })),
    ...(banded
      ? {
          band: {
            lower: line.entries.map((e) => e.lower),
            upper: line.entries.map((e) => e.upper),
            label: basis === "perturbation" ? "±1σ of the perturbation" : "±1σ",
          },
        }
      : {}),
  };
}

/** What the marks on the x axis say, in words (never by colour alone: each
 * has its text on hover). */
export const MARK_KEY: { tone: Marker["tone"]; text: string }[] = [
  { tone: "signal", text: "representative run" },
  { tone: "normal", text: "starred run" },
  { tone: "warn", text: "ended on a bound" },
];

/** The ticks on a chart's x axis: the representative and starred runs, and
 * the runs whose value ended on a bound. */
export function markersFor(
  lines: readonly Line[],
  meta: ReadonlyMap<string, RunMeta>,
): Marker[] {
  const markers: Marker[] = [];
  for (const line of lines) {
    const prefix = lines.length > 1 ? `${line.unit}: ` : "";
    for (const e of line.entries) {
      const m = meta.get(e.key);
      if (m?.representative) {
        markers.push({
          position: e.x,
          tone: "signal",
          label: `${prefix}${m.name} (representative)`,
        });
      } else if (m?.starred) {
        markers.push({
          position: e.x,
          tone: "normal",
          label: `${prefix}${m.name} (starred)`,
        });
      }
      if (e.atBound) {
        markers.push({
          position: e.x,
          tone: "warn",
          label: `${prefix}${m?.name ?? "run"}: ended on a bound`,
        });
      }
    }
  }
  return markers;
}

/** The run nearest to `x` on any of `lines` (a run with a value, so a
 * hover never lands on a gap); `null` without one. */
export function nearestRun(lines: readonly Line[], x: number): string | null {
  let best: { key: string; d: number } | null = null;
  for (const line of lines) {
    for (const e of line.entries) {
      if (Number.isNaN(e.y)) continue;
      const d = Math.abs(e.x - x);
      if (best === null || d < best.d) best = { key: e.key, d };
    }
  }
  return best?.key ?? null;
}

/** Where run `key` is on the chart's x axis; `null` when it is not there. */
export function xOfRun(lines: readonly Line[], key: string | null): number | null {
  if (key === null) return null;
  for (const line of lines) {
    const e = line.entries.find((entry) => entry.key === key);
    if (e) return e.x;
  }
  return null;
}

/** The x axis's tick and readout text. */
export function xFormatter(xMode: XMode): (x: number) => string {
  // A run number is whole: a tick between two runs says nothing.
  return xMode === "time"
    ? (x) => dateOf(x)
    : (x) => (Number.isInteger(x) ? fixed(x, 0) : "");
}

export const xAxisLabel = (xMode: XMode): string =>
  xMode === "time" ? "recorded time" : "run number in the history";

/** A spread as one line: the median and the robust σ, then the extremes. */
export function spreadLine(spread: SeedSpread): string {
  return `median ${significant(spread.median)} · robust σ ${significant(spread.mad_sigma, 3)} · min ${significant(spread.min)} · max ${significant(spread.max)}`;
}

/** What a chart says about the values it could not draw. */
export function gapNote(line: Line, xMode: XMode): string | null {
  const parts: string[] = [];
  if (line.missing > 0) {
    parts.push(
      `${line.missing} ${line.missing === 1 ? "run has" : "runs have"} no value (a gap)`,
    );
  }
  if (line.withoutSigma > 0) {
    parts.push(`${line.withoutSigma} without a σ (no band)`);
  }
  if (xMode === "time" && line.undated > 0) {
    parts.push(`${line.undated} with no recorded time (not drawn)`);
  }
  return parts.length > 0 ? parts.join("; ") : null;
}

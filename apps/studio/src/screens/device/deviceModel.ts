/** The Device screen's pure rules over `device_history` and the library
 * scan: which device or type is shown (and how the address says so), the
 * rows of its history, one row per unit of a type, and how a run's seed is
 * described. Nothing here reads the back end or draws. */
import type {
  DeviceHistoryView,
  DeviceIdentity,
  DeviceKey,
  DeviceRunRow,
  DeviceSelector,
  RunSeedUse,
} from "../../api/generated/types";
import { deviceOf, UNSPECIFIED_TYPE, type LibraryGroups } from "../library/libraryModel";

export type Scope = "device" | "type";
export const DEVICE_TABS = ["history", "trends", "seed"] as const;
export type DeviceTab = (typeof DEVICE_TABS)[number];
/** What the trends' x axis is: the run's position, or its recorded time. */
export type XMode = "order" | "time";

/** What the screen shows: one device, or every unit of a type. */
export interface DeviceTarget {
  scope: Scope;
  /** The device type; `null` for a device that names none. */
  product: string | null;
  /** The device's id; `null` for a type. */
  id: string | null;
}

/** The screen's state in the address (`#/device?product=…&device=…`). The
 * project and run keep their own parameters. */
export interface DeviceView {
  /** `null`: the open project's device. */
  target: DeviceTarget | null;
  tab: DeviceTab;
  starredOnly: boolean;
  xMode: XMode;
}

export function fromParams(params: URLSearchParams): DeviceView {
  const product = params.get("product") || null;
  const id = params.get("device") || null;
  let target: DeviceTarget | null = null;
  if (params.get("scope") === "type" && product) {
    target = { scope: "type", product, id: null };
  } else if (id) {
    target = { scope: "device", product, id };
  }
  const tab = params.get("tab");
  return {
    target,
    tab: (DEVICE_TABS as readonly string[]).includes(tab ?? "")
      ? (tab as DeviceTab)
      : "history",
    starredOnly: params.get("starred") === "1",
    xMode: params.get("x") === "time" ? "time" : "order",
  };
}

/** `params` with the screen's keys replaced by `view`'s (absent at their
 * defaults). */
export function toParams(params: URLSearchParams, view: DeviceView): URLSearchParams {
  const next = new URLSearchParams(params);
  const set = (key: string, value: string | null) => {
    if (value) next.set(key, value);
    else next.delete(key);
  };
  set("product", view.target?.product ?? null);
  set("device", view.target?.id ?? null);
  set("scope", view.target?.scope === "type" ? "type" : null);
  set("tab", view.tab === "history" ? null : view.tab);
  set("starred", view.starredOnly ? "1" : null);
  set("x", view.xMode === "time" ? "time" : null);
  return next;
}

/** The target a project's device names. */
export function targetOfDevice(
  device: DeviceIdentity | null | undefined,
): DeviceTarget | null {
  if (!device) return null;
  return { scope: "device", product: device.product?.trim() || null, id: device.id };
}

/** The type `target` belongs to; `null` for a device of no stated type. */
export function typeOf(target: DeviceTarget): DeviceTarget | null {
  return target.product ? { scope: "type", product: target.product, id: null } : null;
}

/** What `device_history` and `device_series` are asked for. */
export function selectorOf(target: DeviceTarget): DeviceSelector {
  if (target.scope === "type" && target.product) {
    return { kind: "type", product: target.product };
  }
  return target.product
    ? { kind: "device", id: target.id ?? "", product: target.product }
    : { kind: "device", id: target.id ?? "" };
}

export function sameTarget(a: DeviceTarget | null, b: DeviceTarget | null): boolean {
  return a?.scope === b?.scope && a?.product === b?.product && a?.id === b?.id;
}

/** The picker's value for a type: its name, `UNSPECIFIED_TYPE` for none. */
export const productValue = (product: string | null): string =>
  product ?? UNSPECIFIED_TYPE;
export const productOf = (value: string): string | null =>
  value === UNSPECIFIED_TYPE ? null : value;

/** A run of a history, named by its project and id (ids repeat across
 * projects). */
export function runKey(project: string, id: string): string {
  return `${project}\u0000${id}`;
}

/** One row of a device's history. */
export interface HistoryRow {
  key: string;
  /** The run's position in the whole history (the trends' run order),
   * whatever is filtered out. */
  order: number;
  run: DeviceRunRow;
}

/** The runs of `view` as rows, in the library's order; `starredOnly` keeps
 * the starred ones. */
export function historyRows(view: DeviceHistoryView, starredOnly: boolean): HistoryRow[] {
  return view.runs
    .map((run, order) => ({ key: runKey(run.project, run.id), order, run }))
    .filter((row) => !starredOnly || row.run.starred);
}

/** The ids of the cameras the rows have an RMS for, in first-seen order. */
export function cameraIds(rows: readonly HistoryRow[]): string[] {
  const ids: string[] = [];
  for (const { run } of rows) {
    for (const camera of run.cameras) if (!ids.includes(camera.id)) ids.push(camera.id);
  }
  return ids;
}

export function rmsOf(run: DeviceRunRow, camera: string): number | null {
  return run.cameras.find((c) => c.id === camera)?.rms_px ?? null;
}

/** The run that stands for the device. */
export function representativeOf(view: DeviceHistoryView): DeviceRunRow | null {
  return view.runs.find((r) => r.representative) ?? null;
}

/** The text a run's seed is shown as, and the lists behind it. */
export function seedLine(
  use: RunSeedUse | null | undefined,
): { text: string; title: string } | null {
  if (!use) return null;
  const source =
    "run" in use.source
      ? use.source.run
      : "device_spec" in use.source
        ? use.source.device_spec
        : use.source.file;
  const name = use.description?.trim() || lastSegments(source);
  const counts = `${use.applied.length} applied · ${use.ignored.length} ignored`;
  const lines = [
    source,
    use.applied.length > 0 ? `applied: ${use.applied.join(", ")}` : "applied: nothing",
    use.ignored.length > 0 ? `ignored: ${use.ignored.join(", ")}` : "ignored: nothing",
  ];
  return { text: `${name} · ${counts}`, title: lines.join("\n") };
}

function lastSegments(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.slice(-2).join("/");
}

/** One unit of a type: its identity, how many runs and projects it has,
 * and the run that stands for it. */
export interface UnitRow {
  device: DeviceIdentity;
  runs: number;
  projects: number;
  starred: number;
  representative: DeviceRunRow | null;
}

export function unitRows(units: readonly DeviceHistoryView[]): UnitRow[] {
  return units.map((unit) => ({
    device: unit.device,
    runs: unit.runs.length,
    projects: new Set(unit.runs.map((r) => r.project)).size,
    starred: unit.runs.filter((r) => r.starred).length,
    representative: representativeOf(unit),
  }));
}

export function unitKey(device: DeviceKey | DeviceIdentity): string {
  return `${device.product ?? ""}\u0000${device.id}`;
}

/** `2026-11-02 14:05` of seconds since the epoch, local; `–` when absent. */
export function whenText(seconds: number | null | undefined): string {
  if (seconds == null) return "–";
  const d = new Date(seconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** The Device screen's parameters for a Library node that is a device or a
 * device type; `null` for any other node (and for the unspecified type,
 * which has no units to compare). */
export function deviceParamsOf(
  groups: LibraryGroups,
  nodeId: string | null,
): URLSearchParams | null {
  if (nodeId === null) return null;
  const params = new URLSearchParams();
  const product = groups.products.find((p) => p.id === nodeId);
  if (product) {
    if (product.label === UNSPECIFIED_TYPE) return null;
    params.set("product", product.label);
    params.set("scope", "type");
    return params;
  }
  if (!nodeId.startsWith("device:")) return null;
  const device = deviceOf(groups, nodeId)?.device;
  if (!device) return null;
  if (device.product?.trim()) params.set("product", device.product.trim());
  params.set("device", device.id);
  return params;
}

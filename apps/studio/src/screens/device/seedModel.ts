/** The Seed tab's pure rules: what a source choice asks the back end for
 * (or why it cannot yet), the values of a seed as rows, the file an export
 * writes, and what the open project's `[seed]` becomes. */
import type {
  DeviceHistoryView,
  DeviceKey,
  DeviceRunRow,
  DeviceSeed,
  Iso3Schema,
  SeedFrom,
  SeedSource,
} from "../../api/generated/types";
import { joinPath } from "../../lib/paths";
import { runKey, unitKey } from "./deviceModel";

export type SourceKind = "run" | "runs" | "units";

/** What the person chose to seed from. */
export interface SeedChoice {
  kind: SourceKind;
  /** The run (`runKey`) of "this run". */
  run: string | null;
  /** The runs (`runKey`s) of "chosen runs". */
  runs: readonly string[];
  /** The units (`unitKey`s) of "the type's units". */
  units: readonly string[];
}

export const NO_CHOICE: SeedChoice = { kind: "run", run: null, runs: [], units: [] };

/** Whether a run can be offered as a source: one whose files read. */
export function seedable(run: DeviceRunRow): boolean {
  return run.origin.kind !== "unreadable";
}

/** The request `choice` makes, or what is still missing. `runs` and `units`
 * are the device's and the type's histories. */
export function seedRequest(
  choice: SeedChoice,
  runs: readonly DeviceRunRow[],
  units: readonly DeviceHistoryView[],
): { from: SeedFrom } | { problem: string } {
  const byKey = new Map(runs.map((r) => [runKey(r.project, r.id), r]));
  switch (choice.kind) {
    case "run": {
      const run = choice.run === null ? undefined : byKey.get(choice.run);
      if (!run) return { problem: "Choose the run to seed from." };
      return { from: { kind: "run", run: { dir: run.project, id: run.id } } };
    }
    case "runs": {
      const chosen = choice.runs.map((k) => byKey.get(k)).filter((r) => r !== undefined);
      if (chosen.length < 2) {
        return {
          problem:
            "Choose at least two runs to aggregate; for one run, seed from “This run”.",
        };
      }
      return {
        from: {
          kind: "runs",
          runs: chosen.map((r) => ({ dir: r.project, id: r.id })),
        },
      };
    }
    case "units": {
      const chosen = units.filter(
        (u) =>
          choice.units.includes(unitKey(u.device)) &&
          u.runs.some((r) => r.representative),
      );
      if (chosen.length < 2) {
        return {
          problem: "Choose at least two units that have a representative run.",
        };
      }
      return {
        from: {
          kind: "units",
          devices: chosen.map((u): DeviceKey =>
            u.device.product
              ? { id: u.device.id, product: u.device.product }
              : { id: u.device.id },
          ),
        },
      };
    }
  }
}

/** One value of a seed. */
export interface SeedValueRow {
  /** The camera, or `null` for the hand-eye. */
  camera: string | null;
  label: string;
  value: number;
  unit: string;
}

function pose(camera: string | null, prefix: string, iso: Iso3Schema): SeedValueRow[] {
  const [qx, qy, qz, qw] = iso.rotation;
  const [tx, ty, tz] = iso.translation;
  const row = (label: string, value: number, unit: string): SeedValueRow => ({
    camera,
    label: `${prefix}.${label}`,
    value,
    unit,
  });
  return [
    row("qx", qx, ""),
    row("qy", qy, ""),
    row("qz", qz, ""),
    row("qw", qw, ""),
    row("tx", tx, "m"),
    row("ty", ty, "m"),
    row("tz", tz, "m"),
  ];
}

/** The numbers of a camera's distortion or sensor, by name. */
function numbers(
  camera: string,
  object: Record<string, unknown>,
  skip: readonly string[],
): SeedValueRow[] {
  return Object.entries(object)
    .filter(([k, v]) => typeof v === "number" && !skip.includes(k))
    .map(([k, v]) => ({
      camera,
      label: k,
      value: v as number,
      unit: k.startsWith("tilt") ? "rad" : "",
    }));
}

/** Every value of `seed`, camera by camera, then the hand-eye. */
export function seedValues(seed: DeviceSeed): SeedValueRow[] {
  const rows: SeedValueRow[] = [];
  for (const camera of seed.cameras) {
    const id = camera.id;
    const lens = camera.lens;
    const intr = lens.kind === "calibrated" ? lens.camera.intrinsics : lens.intrinsics;
    for (const name of ["fx", "fy", "cx", "cy"] as const) {
      rows.push({ camera: id, label: name, value: intr[name], unit: "px" });
    }
    if (lens.kind === "calibrated") {
      rows.push(...numbers(id, lens.camera.distortion, ["iters"]));
      rows.push(...numbers(id, lens.camera.sensor, []));
    } else {
      rows.push(...numbers(id, lens.sensor as unknown as Record<string, unknown>, []));
    }
    if (camera.cam_se3_rig) rows.push(...pose(id, "cam_se3_rig", camera.cam_se3_rig));
    if (camera.laser_plane) {
      const [nx, ny, nz] = camera.laser_plane.normal;
      for (const [label, value] of [
        ["nx", nx],
        ["ny", ny],
        ["nz", nz],
        ["distance", camera.laser_plane.distance],
      ] as const) {
        rows.push({ camera: id, label: `laser_plane.${label}`, value, unit: "" });
      }
    }
  }
  const handeye = seed.handeye;
  if (handeye) {
    rows.push(
      ...(handeye.mode === "eye_in_hand"
        ? pose(null, "gripper_se3_rig", handeye.gripper_se3_rig)
        : pose(null, "rig_se3_base", handeye.rig_se3_base)),
    );
  }
  return rows;
}

/** A one-line description of a seed's lens, per camera. */
export function lensLine(seed: DeviceSeed["cameras"][number]): string {
  const [w, h] = seed.image_size;
  const kind = seed.lens.kind === "calibrated" ? "calibrated lens" : "nominal lens";
  return `${w}×${h} px · ${kind}`;
}

/** The file an export writes: `folder` and `name`, with the format the
 * name's extension asks for; `problem` when the name cannot be used. */
export function exportDest(
  folder: string | null,
  name: string,
): { dest: string; format: "json" | "toml" } | { problem: string } {
  const trimmed = name.trim();
  if (folder === null) return { problem: "Choose the folder to write to." };
  if (trimmed === "") return { problem: "Name the file." };
  if (/[\\/]/.test(trimmed)) return { problem: "The name is a file name, not a path." };
  const lower = trimmed.toLowerCase();
  if (lower.endsWith(".json")) return { dest: joinPath(folder, trimmed), format: "json" };
  if (lower.endsWith(".toml")) return { dest: joinPath(folder, trimmed), format: "toml" };
  return { problem: "End the name with .json or .toml: the extension picks the format." };
}

/** The `[seed]` a single run makes directly (its folder); an aggregate has
 * to be a file first, so it makes none. */
export function runSeedSource(from: SeedFrom): SeedSource | null {
  return from.kind === "run"
    ? { run: joinPath(from.run.dir, "runs", from.run.id) }
    : null;
}

/** The seed of a project as a line. */
export function describeSource(source: SeedSource | null | undefined): string {
  if (!source) return "none";
  if ("run" in source) return `run ${source.run}`;
  if ("device_spec" in source) return `device specification ${source.device_spec}`;
  return `file ${source.file}`;
}

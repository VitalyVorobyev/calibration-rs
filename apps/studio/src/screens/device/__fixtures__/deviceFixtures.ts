/** Wire-typed fixtures of the Device screen: a device of one type with
 * four runs in two projects (one starred, one the representative, one that
 * cannot be read), a second unit of the type, their series with a gap, a σ
 * of a perturbation and a value on a bound, and a seed preview. */
import type {
  DeviceHistories,
  DeviceHistoryView,
  DeviceRunRow,
  DeviceSeries,
  DeviceSeriesSet,
  ParameterSeries,
  SeedPreview,
  SeedSpread,
} from "../../../api/generated/types";

export const BENCH = "/data/bench";
export const FIELD = "/data/field";

export const run = (
  project: string,
  id: string,
  extra: Partial<DeviceRunRow> = {},
): DeviceRunRow => ({
  cameras: [
    { id: "cam0", rms_px: 0.231 },
    { id: "cam1", rms_px: 0.247 },
  ],
  id,
  origin: { kind: "calibrated" },
  project,
  project_name: project === BENCH ? "Bench calibration" : "Field recheck",
  representative: false,
  starred: false,
  timestamp: 1_700_000_000,
  verdict: "pass",
  ...extra,
});

export const SN1: DeviceHistoryView = {
  device: { id: "SN-0042", product: "Stereo-X1", label: "Bench cell 3" },
  runs: [
    run(BENCH, "run-001", { timestamp: null, verdict: "warn" }),
    run(BENCH, "run-002", {
      starred: true,
      representative: true,
      timestamp: 1_700_100_000,
    }),
    run(FIELD, "run-001", {
      timestamp: 1_700_200_000,
      verdict: "fail",
      seed: {
        applied: ["cam0.lens", "cam1.lens"],
        ignored: ["handeye"],
        description: "the calibrated run run-002",
        source: { run: "/data/bench/runs/run-002" },
      },
    }),
    run(FIELD, "run-002", {
      origin: { kind: "unreadable", message: "run.json: not valid JSON" },
      cameras: [],
      verdict: null,
      timestamp: null,
    }),
  ],
};

export const SN2: DeviceHistoryView = {
  device: { id: "SN-0057", product: "Stereo-X1" },
  runs: [
    run("/data/second", "run-001", { representative: true, timestamp: 1_700_150_000 }),
  ],
};

export const HISTORY: DeviceHistories = { devices: [SN1], failures: [] };
export const TYPE_HISTORY: DeviceHistories = { devices: [SN1, SN2], failures: [] };

const point = (
  value: number,
  sigma: number | null,
  at_bound = false,
): NonNullable<ParameterSeries["points"][number]> => ({
  at_bound,
  sigma:
    sigma === null
      ? { kind: "unavailable", reason: { kind: "fixed" } }
      : { kind: "estimated", sigma },
  value,
});

export const SERIES_RUNS: DeviceSeries["runs"] = SN1.runs.map((r, order) => ({
  id: r.id,
  order,
  project: r.project,
  timestamp: r.timestamp ?? null,
  ...(r.origin.kind === "unreadable"
    ? { error: { kind: "parse", message: "run.json: not valid JSON" } as const }
    : {}),
}));

const FX: ParameterSeries = {
  camera: "cam0",
  group: "intrinsics",
  key: "cam0.fx",
  name: "fx",
  // The third run has no value of it: a gap.
  points: [point(200, 0.5), point(200.6, 0.4), null, null],
  sigma_of: "value",
  spread: "cam0.fx",
  unit: "px",
};

const K1: ParameterSeries = {
  camera: "cam0",
  group: "distortion",
  key: "cam0.k1",
  name: "k1",
  points: [point(-0.12, 0.01), point(-0.1, null, true), point(-0.11, 0.01), null],
  sigma_of: "value",
  spread: "cam0.k1",
  unit: "",
};

const HANDEYE: ParameterSeries = {
  camera: null,
  group: "handeye",
  key: "gripper_se3_rig.tx",
  name: "tx",
  points: [point(0.1, 0.002), point(0.101, 0.002), point(0.099, 0.002), null],
  sigma_of: "perturbation",
  spread: null,
  unit: "m",
};

export const SPREAD_FX: SeedSpread = {
  label: "cam0.fx",
  mad_sigma: 0.4,
  max: 200.6,
  median: 200,
  min: 199.8,
  values: [200, 200.6, 199.8],
};

export const SERIES_SN1: DeviceSeries = {
  device: SN1.device,
  parameters: [FX, K1, HANDEYE],
  runs: SERIES_RUNS,
  statistics: {
    status: "available",
    left_out: [
      {
        run: 3,
        reason: { kind: "seed", message: "run-002: the run's export cannot be read" },
      },
    ],
    omitted: [],
    runs: [0, 1, 2],
    spread: [
      SPREAD_FX,
      { ...SPREAD_FX, label: "cam0.k1", median: -0.11, min: -0.12, max: -0.1 },
    ],
  },
};

export const SERIES: DeviceSeriesSet = { devices: [SERIES_SN1], failures: [] };

export const SERIES_UNAVAILABLE: DeviceSeriesSet = {
  devices: [
    {
      ...SERIES_SN1,
      statistics: {
        status: "unavailable",
        left_out: [],
        reason: {
          kind: "seed",
          message:
            "the runs have different cameras, so their values cannot be aggregated",
        },
      },
    },
  ],
  failures: [
    { path: "/data/broken", message: "calibration-project.toml: not valid TOML" },
  ],
};

export const PREVIEW_RUN: SeedPreview = {
  omitted: [],
  runs: [{ id: "run-002", project: BENCH }],
  seed: {
    version: 1,
    cameras: [
      {
        id: "cam0",
        image_size: [96, 72],
        lens: {
          kind: "calibrated",
          camera: {
            intrinsics: { fx: 200.6, fy: 200.4, cx: 48, cy: 36 },
            distortion: {
              type: "brown_conrady5",
              k1: -0.1,
              k2: 0,
              k3: 0,
              p1: 0,
              p2: 0,
              iters: 8,
            },
            projection: { type: "pinhole" },
            sensor: { type: "identity" },
          },
        },
      },
    ],
  },
  spread: [],
};

export const PREVIEW_AGGREGATE: SeedPreview = {
  ...PREVIEW_RUN,
  omitted: ["cam0.laser_plane"],
  runs: [
    { id: "run-002", project: BENCH },
    { id: "run-001", project: FIELD },
  ],
  spread: [SPREAD_FX],
};

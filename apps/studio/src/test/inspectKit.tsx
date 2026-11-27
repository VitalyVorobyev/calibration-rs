/** What the Inspect tabs' component tests share: a mock back end answering
 * the commands the tabs read (and the planes their blobs name), a run shown in
 * the export store, and the reset between tests. */
import { render } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, vi } from "vitest";
import { setTransport } from "../api";
import type {
  CameraParameters,
  QualityReport,
  ResidualRecords,
  ResidualStats,
  RunRef,
} from "../api/generated/types";
import { createMockTransport, type CommandHandlers } from "../api/transport/mock";
import { queryCache } from "../session/query";
import { useSelection } from "../session/selection";
import { useStore } from "../store";
import { PLANAR_EXPORT_FIXTURE } from "./exportFixtures";
import {
  distortionPlane,
  distortionReply,
  residualPlane,
  residualReply,
  sigmaPlane,
  sigmaReply,
  vam1,
} from "./planeFixtures";

export const RUN: RunRef = { dir: "/work/cell", id: "run-001" };

export const PARAMETERS: CameraParameters = {
  cameras: [
    {
      camera: 0,
      projection: "pinhole",
      distortion: "brown_conrady5",
      sensor: "identity",
      parameters: [
        {
          at_bound: false,
          group: "intrinsics",
          label: "fx",
          name: "fx",
          sigma: { kind: "estimated", sigma: 0.21 },
          unit: "px",
          value: 1412.372,
        },
        {
          at_bound: false,
          group: "distortion",
          label: "k3",
          name: "k3",
          sigma: { kind: "unavailable", reason: { kind: "fixed" } },
          unit: "",
          value: 0,
        },
        {
          at_bound: true,
          group: "distortion",
          label: "k2",
          name: "k2",
          sigma: {
            kind: "unavailable",
            reason: {
              kind: "no_covariance",
              reason: { kind: "at_bound", labels: ["k2"] },
            },
          },
          unit: "",
          value: 0.3,
        },
      ],
    },
  ],
};

export const QUALITY: QualityReport = {
  verdict: "warn",
  checks: [
    { kind: "convergence", status: "pass", message: "the final solve converged" },
    {
      kind: "projection_sigma",
      camera: 0,
      status: "warn",
      message: "projection σ up to 1.4 px",
      measurement: { value: 1.4, threshold: 1, unit: "pixel" },
    },
    { kind: "feature_coverage", camera: 1, status: "fail", message: "other camera" },
  ],
};

/** Three records: |r| is 0.1 (view 0, feature 4), 0.5 (view 0, feature 5) and
 * 1 (view 1, feature 6). */
export const RECORDS: ResidualRecords = {
  index: [0, 1, 2],
  view: [0, 0, 1],
  feature: [4, 5, 6],
  target_xyz_m: [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ],
  observed_px: [
    [10, 10],
    [20, 10],
    [30, 10],
  ],
  projected_px: [
    [9.9, 10],
    [20.5, 10],
    [30, 9],
  ],
  residual_px: [
    [0.1, 0],
    [-0.5, 0],
    [0, 1],
  ],
};

export const STATS: ResidualStats = {
  cameras: [
    {
      camera: 0,
      stats: { count: 3, mean: 0.5, rms: 0.6, max: 1 },
      unprojected: 0,
      views: 2,
    },
  ],
  views: [
    { view: 0, stats: { count: 2, mean: 0.3, rms: 0.36, max: 0.5 }, unprojected: 0 },
    { view: 1, stats: { count: 1, mean: 1, rms: 1, max: 1 }, unprojected: 0 },
  ],
  cells: [],
};

const PLANES: Record<string, ArrayBuffer> = {
  lens: vam1(distortionPlane()),
  tilt: vam1(distortionPlane()),
  total: vam1(distortionPlane()),
  residual: vam1(residualPlane()),
  sigma: vam1(sigmaPlane()),
};

/** A mock back end that answers the tabs' commands; `commands` replaces any. */
export function backend(commands: CommandHandlers = {}) {
  const mock = createMockTransport({
    commands: {
      camera_parameter_values: () => PARAMETERS,
      run_quality: () => QUALITY,
      distortion_map: () => distortionReply(),
      residual_map: () => residualReply(),
      projection_sigma_map: () => sigmaReply(),
      residual_records: () => RECORDS,
      residual_stats: () => STATS,
      ...commands,
    },
    resources: { blob: ({ id }) => `mock://blob/${id}` },
  });
  setTransport(mock);
  return mock;
}

/** Show `RUN` in the export store (its export with `extra` fields). */
export function show(extra: Record<string, unknown> = {}) {
  useStore
    .getState()
    .showRun(RUN, { export: { ...PLANAR_EXPORT_FIXTURE, ...extra }, image_root: null });
}

export function mount(ui: ReactElement) {
  return render(
    <TooltipProvider>
      <MemoryRouter initialEntries={["/inspect/quality?project=/work/cell&run=run-001"]}>
        {ui}
      </MemoryRouter>
    </TooltipProvider>,
  );
}

/** Reset the stores, serve the planes by blob id and draw no canvas (jsdom
 * has none); undone after each test. */
export function installInspectHooks() {
  beforeEach(() => {
    queryCache.clear();
    useSelection.getState().setSelection({ camera: null, view: null, feature: null });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    vi.stubGlobal("fetch", (url: string) => {
      const id = url.split("/").pop() ?? "";
      const body = PLANES[id];
      return Promise.resolve(
        body ? new Response(body) : new Response("", { status: 404 }),
      );
    });
  });
  afterEach(() => {
    setTransport(null);
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    useStore.getState().resetExport();
  });
}

/** What the Compare screen's component tests share: a mock back end
 * answering the two commands (and the map's plane by its blob id), the
 * project with three runs in the store, the screen at an address, and the
 * reset between tests. */
import { act, cleanup, render } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, beforeEach, vi } from "vitest";
import type { CommandHandlers } from "../../../api/transport/mock";
import type { ProjectionDifferenceMap, RunInfo } from "../../../api/generated/types";
import { useProject } from "../../../project/projectStore";
import { useCompareSet } from "../../../session/compareSet";
import { calibratedRun, projectView } from "../../../test/projectFixtures";
import { GRID, plane, vam1 } from "../../../test/planeFixtures";
import { backend, resetShellStores } from "../../../test/shellKit";
import { CompareScreen } from "..";
import { DIR, comparison } from "./compareFixtures";

export const IDS = ["run-001", "run-002", "run-003"];
const CHANNELS = ["dx", "dy", "magnitude", "fitted"].map((name) => ({
  name,
  unit: name === "fitted" ? "" : "px",
  domain: [0, 1] as [number, number],
}));
export const MAP: ProjectionDifferenceMap = {
  status: "compared",
  camera: "cam0",
  compensation: "rotation",
  fit_nodes: 9,
  valid_nodes: 12,
  max_fit_px: 0.12,
  rms_fit_px: 0.05,
  max_valid_px: 0.3,
  rms_valid_px: 0.11,
  rotation_deg: 0.004,
  grid: GRID,
  plane: {
    blob: { id: "diff", len: 100, media_type: "application/x-vam1" },
    channels: CHANNELS,
  },
};
const PLANE = vam1(
  plane(4, 3, 4, (c, col, row) =>
    c === 0
      ? col * 0.01
      : c === 1
        ? row * 0.01
        : c === 2
          ? Math.hypot(col, row) * 0.01
          : col < 3
            ? 1
            : 0,
  ),
);

export function runs(starred: string[] = []): RunInfo[] {
  return IDS.map((id) => ({
    ...calibratedRun(DIR, id),
    notes: starred.includes(id) ? { starred: true } : {},
  }));
}

export function mount(
  commands: CommandHandlers = {},
  initial = `/compare?project=${DIR}`,
  starred: string[] = [],
) {
  const mock = backend(undefined, {
    commands: {
      compare_runs: () => comparison(),
      projection_difference_map: () => MAP,
      set_run_notes: ({ id, notes }) => ({ ...runs().find((r) => r.id === id)!, notes }),
      ...commands,
    },
    resources: { blob: ({ id }) => `mock://blob/${id}` },
  });
  act(() => useProject.setState({ project: projectView(DIR, runs(starred)) }));
  const router = createMemoryRouter([{ path: "/compare", element: <CompareScreen /> }], {
    initialEntries: [initial],
  });
  render(
    <TooltipProvider>
      <RouterProvider router={router} />
    </TooltipProvider>,
  );
  return { mock, router };
}

/** Draw no canvas, serve the plane, empty the set; undone after each test. */
export function installCompareHooks() {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    vi.stubGlobal("fetch", (url: string) =>
      Promise.resolve(
        url.endsWith("/diff") ? new Response(PLANE) : new Response("", { status: 404 }),
      ),
    );
    useCompareSet.setState({ dir: null, ids: [] });
  });
  afterEach(() => {
    cleanup();
    resetShellStores();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });
}

export const tick = (ids: string[]) =>
  act(() => useCompareSet.getState().setIds(DIR, ids));

// @vitest-environment jsdom
/** The residual statistics panels read the run's statistics from the back
 * end (`residual_stats`), not from the export's records. */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport } from "../../api";
import type { ResidualStats } from "../../api/generated/types";
import { createMockTransport } from "../../api/transport/mock";
import { queryCache } from "../../session/query";
import { ResidualStatsPanels } from "./ResidualStatsPanels";

const RUN = { dir: "/p", id: "run-001" };

const STATS: ResidualStats = {
  cameras: [],
  views: [
    { view: 0, stats: { count: 35, mean: 0.12, rms: 0.15, max: 0.4 }, unprojected: 0 },
    { view: 1, stats: { count: 33, mean: 0.2, rms: 0.25, max: 0.9 }, unprojected: 2 },
  ],
  cells: [
    {
      camera: 0,
      view: 0,
      stats: { count: 35, mean: 0.12, rms: 0.15, max: 0.4 },
      unprojected: 0,
    },
    {
      camera: 1,
      view: 1,
      stats: { count: 33, mean: 0.2, rms: 0.25, max: 0.9 },
      unprojected: 2,
    },
  ],
};

let asked: unknown[] = [];

beforeEach(() => {
  asked = [];
  queryCache.clear();
  setTransport(
    createMockTransport({
      commands: {
        residual_stats: (args) => {
          asked.push(args);
          return STATS;
        },
      },
    }),
  );
});

afterEach(() => {
  cleanup();
  setTransport(null);
});

describe("ResidualStatsPanels", () => {
  it("shows the back end's per-pose rows and per-camera cells", async () => {
    render(
      <ResidualStatsPanels
        run={RUN}
        selectedPose={0}
        selectedCamera={0}
        onSelectPose={() => {}}
        onSelectFrame={() => {}}
      />,
    );
    // Pose 1: RMS 0.25 px, two diverged corners.
    expect(await screen.findByText("0.250")).toBeTruthy();
    expect(screen.getByTitle("2 diverged corners")).toBeTruthy();
    expect(
      screen.getByRole("button", {
        name: /cam 1 · pose 1: mean 0\.200 px over 33 corners/,
      }),
    ).toBeTruthy();
    expect(asked).toEqual([{ run: RUN }]);
  });
});

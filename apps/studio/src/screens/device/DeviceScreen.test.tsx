// @vitest-environment jsdom
/** The Device screen over a mock back end: the history with its
 * representative and unreadable runs, the trend charts with their spread,
 * the units of a type, and the seed (previewed, exported, set on the open
 * project, refused). */
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it } from "vitest";
import type { LibraryScan, ProjectEntry, ProjectView } from "../../api/generated/types";
import { StudioError } from "../../api";
import { useProjectDialog } from "../../project/dialogs";
import { useProject } from "../../project/projectStore";
import { calibratedRun, projectView } from "../../test/projectFixtures";
import { backend, resetShellStores, SERVER } from "../../test/shellKit";
import { Shell } from "../../shell/Shell";
import { useJobs } from "../../session/jobs";
import { DeviceScreen } from ".";
import {
  BENCH,
  HISTORY,
  PREVIEW_AGGREGATE,
  PREVIEW_RUN,
  SERIES,
  SERIES_UNAVAILABLE,
  TYPE_HISTORY,
} from "./__fixtures__/deviceFixtures";

afterEach(() => {
  cleanup();
  resetShellStores();
  // A dialog left open would hide the next test's page.
  useProjectDialog.getState().close();
});

const entry = (name: string, id: string): ProjectEntry => ({
  name,
  dir: `/data/${name}`,
  device: { id, product: "Stereo-X1" },
  runs: [],
  topology: "rig_extrinsics",
});
const SCAN: LibraryScan = {
  failures: [],
  projects: [entry("bench", "SN-0042"), entry("second", "SN-0057")],
};

function Where() {
  return <p>device screen</p>;
}

function renderDevice(initial = "/device?product=Stereo-X1&device=SN-0042") {
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: <Shell />,
        children: [
          { path: "device", element: <DeviceScreen /> },
          { path: "inspect/:tab", element: <Where /> },
        ],
      },
    ],
    { initialEntries: [initial] },
  );
  const view = render(
    <TooltipProvider>
      <RouterProvider router={router} />
    </TooltipProvider>,
  );
  return { router, ...view };
}

/** `view` as a project of the device the history is of. */
const withDevice = (view: ProjectView): ProjectView => ({
  ...view,
  spec: { ...view.spec, device: { id: "SN-0042", product: "Stereo-X1" } },
});

const device = (overrides: Record<string, unknown> = {}) =>
  backend(SERVER, {
    commands: {
      scan_library: () => SCAN,
      device_history: ({ device: d }) => (d.kind === "type" ? TYPE_HISTORY : HISTORY),
      device_series: () => SERIES,
      seed_preview: ({ from }) => (from.kind === "run" ? PREVIEW_RUN : PREVIEW_AGGREGATE),
      ...overrides,
    },
  });

describe("History", () => {
  it("lists the device's runs, names the representative one, and shows what cannot be read", async () => {
    device();
    renderDevice();
    const table = await screen.findByRole("table", { name: "Runs of the device" });
    const rows = within(table).getAllByRole("row");
    const text = (id: string, project: string) =>
      rows.find((r) => {
        // The run's own cell, not the seed it started from.
        const own = within(r).queryAllByRole("cell")[1]?.textContent ?? "";
        return own.startsWith(id) && own.includes(project);
      })!;
    // The representative run is named, not only coloured.
    expect(within(text("run-002", "Bench")).getByText("representative")).toBeTruthy();
    // The run that started from a seed says which, and what it applied and ignored.
    expect(
      within(text("run-001", "Field")).getByText(
        "the calibrated run run-002 · 2 applied · 1 ignored",
      ),
    ).toBeTruthy();
    // An unreadable run is a row with its reason.
    const bad = text("run-002", "Field");
    expect(within(bad).getByText("unreadable")).toBeTruthy();
    expect(within(bad).getByText("run.json: not valid JSON")).toBeTruthy();
    // Per-camera RMS.
    expect(within(text("run-002", "Bench")).getByText("0.247")).toBeTruthy();
  });

  it("keeps only the starred runs on request", async () => {
    device();
    const { router } = renderDevice();
    await screen.findByRole("table", { name: "Runs of the device" });
    fireEvent.click(screen.getByRole("switch", { name: "Starred runs only" }));
    await waitFor(() => {
      const table = screen.getByRole("table", { name: "Runs of the device" });
      // The header row and the one starred run.
      expect(within(table).getAllByRole("row")).toHaveLength(3);
    });
    expect(router.state.location.search).toContain("starred=1");
  });

  it("opens a run in Inspect with its project in the address", async () => {
    device();
    const { router } = renderDevice();
    await screen.findByRole("table", { name: "Runs of the device" });
    fireEvent.click(
      screen.getByRole("button", {
        name: "Open run-002 of Bench calibration in Inspect",
      }),
    );
    await waitFor(() => expect(router.state.location.pathname).toBe("/inspect/frames"));
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("project")).toBe(BENCH);
    expect(params.get("run")).toBe("run-002");
  });

  it("says when no project names the device", async () => {
    device({ device_history: () => ({ devices: [], failures: [] }) });
    renderDevice("/device?product=Stereo-X1&device=SN-9999");
    expect(
      await screen.findByText(/No project under the folders names the device SN-9999/),
    ).toBeTruthy();
  });

  it("says why it could not read the history, and offers to try again", async () => {
    device({
      device_history: () => {
        throw new StudioError({ kind: "io", message: "the library is unreadable" });
      },
    });
    renderDevice();
    expect(await screen.findByText("The history could not be read")).toBeTruthy();
    expect(screen.getByText("the library is unreadable")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });

  it("starts from the open project's device when the address names none", async () => {
    device();
    renderDevice("/device");
    // No device, no project: how to set one.
    expect(
      await screen.findByText(/Choose a device above, or open a project/),
    ).toBeTruthy();
    act(() =>
      useProject.setState({
        project: withDevice(projectView("/data/bench")),
      }),
    );
    expect(await screen.findByRole("table", { name: "Runs of the device" })).toBeTruthy();
  });

  it("offers Set device for an open project with none", async () => {
    device();
    renderDevice("/device");
    act(() => useProject.setState({ project: projectView("/data/scratch") }));
    expect(await screen.findByText(/names no device/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Set device…" }));
    expect(await screen.findByRole("dialog", { name: "Set device" })).toBeTruthy();
  });
});

describe("Trends", () => {
  const trends = "/device?product=Stereo-X1&device=SN-0042&tab=trends";

  it("draws a chart per parameter of the group, with the library's spread beside it", async () => {
    device();
    renderDevice(trends);
    const fx = await screen.findByRole("img", { name: "cam0.fx over the runs" });
    expect(fx).toBeTruthy();
    expect(
      screen.getByText(/median 200 · robust σ 0.4 · min 199.8 · max 200.6/),
    ).toBeTruthy();
    // A run lacking the value is a gap, and it says so.
    expect(screen.getByText(/2 runs have no value \(a gap\)/)).toBeTruthy();
    // Groups are tabs; the hand-eye's σ is said to be of a perturbation.
    fireEvent.click(screen.getByRole("tab", { name: "Hand-eye" }));
    expect(await screen.findByText("σ of a perturbation")).toBeTruthy();
    expect(screen.getByText(/own parametrization/)).toBeTruthy();
  });

  it("says plainly when the statistics are unavailable, and what could not be read", async () => {
    device({ device_series: () => SERIES_UNAVAILABLE });
    renderDevice(trends);
    expect(await screen.findByText("No spread statistics")).toBeTruthy();
    expect(screen.getAllByText(/different cameras/).length).toBeGreaterThan(0);
    expect(screen.getByText("1 folder could not be read")).toBeTruthy();
  });

  it("lists the runs left out of the spread, with their reasons", async () => {
    device();
    renderDevice(trends);
    fireEvent.click(await screen.findByText(/1 run is left out of the spread/));
    expect(screen.getByText("run-002: the run's export cannot be read")).toBeTruthy();
  });

  it("marks a run in the list when it is hovered, and keeps the mark on a click", async () => {
    device();
    renderDevice(trends);
    const table = await screen.findByRole("table", { name: "Runs of the device" });
    const row = within(table)
      .getAllByRole("row")
      .find(
        (r) =>
          r.textContent?.includes("Field recheck") && r.textContent.includes("run-001"),
      )!;
    fireEvent.pointerEnter(row);
    expect(row.getAttribute("aria-current")).toBe("true");
    fireEvent.pointerLeave(row);
    expect(row.getAttribute("aria-current")).toBeNull();
    fireEvent.click(row);
    fireEvent.pointerLeave(row);
    expect(row.getAttribute("aria-current")).toBe("true");
  });

  it("switches the x axis to the recorded time in the address", async () => {
    device();
    const { router } = renderDevice(trends);
    await screen.findByRole("img", { name: "cam0.fx over the runs" });
    fireEvent.click(screen.getByRole("radio", { name: "Recorded time" }));
    await waitFor(() => expect(router.state.location.search).toContain("x=time"));
    expect(await screen.findByText(/with no recorded time \(not drawn\)/)).toBeTruthy();
  });
});

describe("a device type", () => {
  it("has a row per unit with its representative run, and opens a unit's history", async () => {
    device();
    const { router } = renderDevice("/device?product=Stereo-X1&scope=type");
    const table = await screen.findByRole("table", { name: "Units of the device type" });
    expect(within(table).getByText("Bench cell 3")).toBeTruthy();
    expect(within(table).getByText("SN-0057")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show the history of SN-0057" }));
    await waitFor(() => expect(router.state.location.search).toContain("device=SN-0057"));
    expect(router.state.location.search).not.toContain("scope=type");
  });
});

describe("Seed", () => {
  const seed = "/device?product=Stereo-X1&device=SN-0042&tab=seed";
  const bench = () =>
    withDevice(projectView("/data/bench", [calibratedRun("/data/bench", "run-002")]));

  it("keeps the device and tab in the address when a project opens", async () => {
    device();
    const { router } = renderDevice(seed);
    await screen.findByRole("table", { name: "Values of the seed" });
    act(() => useProject.setState({ project: bench() }));
    await waitFor(() => expect(router.state.location.search).toContain("project="));
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("tab")).toBe("seed");
    expect(params.get("device")).toBe("SN-0042");
    expect(params.get("project")).toBe("/data/bench");
  });

  it("says a seed is a starting value, and previews the representative run's values", async () => {
    device();
    renderDevice(seed);
    expect(await screen.findByText("A seed is a starting value")).toBeTruthy();
    expect(screen.getByText(/does not constrain the result/)).toBeTruthy();
    const values = await screen.findByRole("table", { name: "Values of the seed" });
    expect(within(values).getByText("cam0.fx")).toBeTruthy();
    expect(within(values).getByText("200.6")).toBeTruthy();
  });

  it("previews an aggregate with its spread and what it leaves out", async () => {
    device();
    renderDevice(seed);
    await screen.findByRole("table", { name: "Values of the seed" });
    fireEvent.click(screen.getByRole("radio", { name: "Chosen runs" }));
    expect(await screen.findByText(/at least two runs/)).toBeTruthy();
    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[1]!);
    fireEvent.click(boxes[2]!);
    const spread = await screen.findByRole("table", {
      name: "Values of the seed and their spread across the runs",
    });
    expect(within(spread).getByText("199.8")).toBeTruthy();
    expect(screen.getByText("Left out of the seed")).toBeTruthy();
    expect(screen.getByText(/cam0\.laser_plane/)).toBeTruthy();
  });

  it("shows why no seed can be made", async () => {
    device({
      seed_preview: () => {
        throw new StudioError({
          kind: "seed",
          message: "run-002 has no record of its image sizes",
        });
      },
    });
    renderDevice(seed);
    expect(await screen.findByText("No seed can be made from this")).toBeTruthy();
    expect(screen.getByText("run-002 has no record of its image sizes")).toBeTruthy();
  });

  it("sets a run as the open project's seed, and shows the back end's refusal", async () => {
    const calls: unknown[] = [];
    let refuse = true;
    device({
      set_project_seed: (args: { seed?: unknown }) => {
        calls.push(args.seed);
        if (refuse) {
          throw new StudioError({
            kind: "seed",
            message: "the seed's camera model does not fit the project's",
            field: "seed",
          });
        }
        return { ...bench(), spec: { ...bench().spec, seed: args.seed } };
      },
    });
    renderDevice(seed);
    act(() => useProject.setState({ project: bench() }));
    await screen.findByRole("table", { name: "Values of the seed" });
    const use = screen.getByRole("button", {
      name: "Use this run as the project's seed",
    });
    fireEvent.click(use);
    expect(await screen.findByText("The project cannot take this seed")).toBeTruthy();
    expect(
      screen.getByText("the seed's camera model does not fit the project's"),
    ).toBeTruthy();
    expect(calls).toEqual([{ run: `${BENCH}/runs/run-002` }]);

    refuse = false;
    fireEvent.click(use);
    await waitFor(() =>
      expect(screen.getByText(`run ${BENCH}/runs/run-002`)).toBeTruthy(),
    );
    expect(screen.queryByText("The project cannot take this seed")).toBeNull();
    // Clearing it is one click.
    fireEvent.click(screen.getByRole("button", { name: "Clear seed" }));
    await waitFor(() => expect(calls).toHaveLength(3));
    expect(calls[2]).toBeNull();
  });

  it("asks before a file is replaced, then writes it", async () => {
    const writes: boolean[] = [];
    device({
      export_seed: (args: { overwrite?: boolean; dest: string }) => {
        writes.push(args.overwrite === true);
        if (!args.overwrite) {
          throw new StudioError({
            kind: "file_exists",
            message: "seed.json already exists",
          });
        }
        return { path: args.dest, preview: PREVIEW_RUN };
      },
    });
    renderDevice(seed);
    act(() => useProject.setState({ project: bench() }));
    await screen.findByRole("table", { name: "Values of the seed" });
    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    const confirm = await screen.findByRole("dialog", { name: "Replace the file?" });
    expect(within(confirm).getByText(/\/data\/bench\/seed\.json/)).toBeTruthy();
    fireEvent.click(within(confirm).getByRole("button", { name: "Replace" }));
    expect(await screen.findByText("Seed written")).toBeTruthy();
    expect(writes).toEqual([false, true]);
  });

  it("needs a name with a format, and a folder", async () => {
    device();
    renderDevice(seed);
    await screen.findByRole("table", { name: "Values of the seed" });
    // No project: no folder yet.
    expect(screen.getByText("Choose the folder to write to.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Export" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(screen.getByText("Open a project to use a seed for it.")).toBeTruthy();
  });

  it("waits while the connection is lost", async () => {
    device();
    renderDevice(seed);
    act(() => useProject.setState({ project: bench() }));
    await screen.findByRole("table", { name: "Values of the seed" });
    act(() => useJobs.setState({ connection: "lost" }));
    expect(screen.getByRole("button", { name: "Export" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(
      screen
        .getByRole("button", { name: "Use this run as the project's seed" })
        .hasAttribute("disabled"),
    ).toBe(true);
  });
});

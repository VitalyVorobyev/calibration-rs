// @vitest-environment jsdom
/** The Library over a mock back end: the scan as a tree by device type and
 * device and a table of projects with their latest verdict and RMS, the
 * search, the failures, the empty library, and the dialogs it opens. */
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type {
  LibraryScan,
  ProjectEntry,
  RunInfo,
  Verdict,
} from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { calibratedRun, projectView } from "../../test/projectFixtures";
import { backend, renderShell, resetShellStores, SERVER } from "../../test/shellKit";

afterEach(() => {
  cleanup();
  resetShellStores();
});

function run(dir: string, id: string, verdict: Verdict, rms: number): RunInfo {
  const base = calibratedRun(dir, id);
  if (base.state.kind !== "recorded") throw new Error("fixture");
  return {
    ...base,
    state: {
      ...base.state,
      summary: {
        ...base.state.summary,
        verdict,
        cameras: [{ id: "cam0", features: 100, views: 9, rms_px: rms }],
      },
    },
  };
}

const project = (
  name: string,
  device: ProjectEntry["device"] | undefined,
  runs: RunInfo[] = [],
): ProjectEntry => ({
  name,
  dir: `/data/${name}`,
  device: device ?? null,
  runs,
  topology: "planar_intrinsics",
});

const SCAN: LibraryScan = {
  failures: [
    { path: "/data/broken", message: "calibration-project.toml: not valid TOML" },
  ],
  skipped: [
    {
      path: "/data/bench/old",
      message: "inside the project /data/bench: projects do not nest",
    },
  ],
  projects: [
    project("bench", { id: "SN1", product: "Stereo-X1", label: "Cell 3" }, [
      run("/data/bench", "run-001", "pass", 0.231),
    ]),
    project("field", { id: "SN2", product: "Stereo-X1" }, [
      run("/data/field", "run-001", "fail", 1.9),
    ]),
    project("scratch", null),
  ],
};

/** The table row of the project `name` (its name leads the row's text). */
function rowOf(table: HTMLElement, name: string): HTMLElement | null {
  return (
    within(table)
      .queryAllByRole("row")
      .find((r) => r.textContent?.startsWith(name)) ?? null
  );
}

const library = (scan: LibraryScan = SCAN) =>
  backend(SERVER, { commands: { scan_library: () => scan } });

describe("the Library", () => {
  it("groups projects by device type and device, with the worst verdict in words", async () => {
    library();
    renderShell("/library");
    const tree = await screen.findByRole("tree", { name: "Projects by device" });
    const product = within(tree).getByRole("treeitem", { name: /Stereo-X1/ });
    expect(product.textContent).toContain("Worst verdict: Fail");
    expect(within(tree).getByRole("treeitem", { name: /Cell 3 · SN1/ })).toBeTruthy();
    expect(within(tree).getByRole("treeitem", { name: /\(no device\)/ })).toBeTruthy();
    // The table lists every project with its latest run's verdict and RMS.
    const table = screen.getByRole("table", { name: "Projects" });
    const row = rowOf(table, "bench")!;
    expect(within(row).getByText("Pass")).toBeTruthy();
    expect(within(row).getByText("0.231")).toBeTruthy();
    expect(rowOf(table, "scratch")!.textContent).toContain("no run");
  });

  it("narrows the table to the selected device", async () => {
    library();
    renderShell("/library");
    const tree = await screen.findByRole("tree", { name: "Projects by device" });
    fireEvent.click(within(tree).getByRole("treeitem", { name: /Cell 3 · SN1/ }));
    const table = await screen.findByRole("table", { name: "Projects" });
    expect(rowOf(table, "field")).toBeNull();
    expect(rowOf(table, "bench")).not.toBeNull();
    expect(screen.getByRole("heading", { name: /Stereo-X1 · Cell 3/ })).toBeTruthy();
  });

  it("filters by search and by verdict, and says when nothing matches", async () => {
    library();
    renderShell("/library");
    await screen.findByRole("tree", { name: "Projects by device" });
    fireEvent.change(screen.getByRole("searchbox", { name: "Search projects" }), {
      target: { value: "cell 3" },
    });
    await waitFor(() => {
      const table = screen.getByRole("table", { name: "Projects" });
      expect(rowOf(table, "field")).toBeNull();
      expect(rowOf(table, "bench")).not.toBeNull();
    });
    fireEvent.change(screen.getByRole("searchbox", { name: "Search projects" }), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("switch", { name: "Fail" }));
    await waitFor(() => {
      const table = screen.getByRole("table", { name: "Projects" });
      expect(rowOf(table, "bench")).toBeNull();
      expect(rowOf(table, "field")).not.toBeNull();
    });
    fireEvent.change(screen.getByRole("searchbox", { name: "Search projects" }), {
      target: { value: "zzz" },
    });
    expect(await screen.findByText("No project matches the search.")).toBeTruthy();
  });

  it("does not hide what it could not read", async () => {
    library();
    renderShell("/library");
    const summary = await screen.findByText("1 folder could not be read");
    expect(summary).toBeTruthy();
    expect(screen.getByText("calibration-project.toml: not valid TOML")).toBeTruthy();
    // Nor the project files it did not read, inside another project.
    expect(
      await screen.findByText("1 project inside another project, not read"),
    ).toBeTruthy();
    expect(
      screen.getByText("inside the project /data/bench: projects do not nest"),
    ).toBeTruthy();
  });

  it("opens New project from an empty library, and lists the samples", async () => {
    library({ failures: [], projects: [] });
    renderShell("/library");
    expect(await screen.findByText(/No project under this folder yet/)).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "New project" })[0]!);
    const dialog = await screen.findByRole("dialog", { name: "New project" });
    // Closed again: an open modal would hide the next test's page from queries.
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("opens a project at Inspect when it has a run to read", async () => {
    backend(SERVER, {
      commands: {
        scan_library: () => SCAN,
        open_project: ({ dir }) => projectView(dir, [calibratedRun(dir, "run-001")]),
      },
    });
    renderShell("/library");
    const table = await screen.findByRole("table", { name: "Projects" });
    const row = rowOf(table, "bench")!;
    fireEvent.click(within(row).getByRole("button", { name: "Open" }));
    await waitFor(() => expect(useProject.getState().project?.dir).toBe("/data/bench"));
    await waitFor(() =>
      expect(screen.getByTestId("where").textContent).toContain("/inspect/frames"),
    );
  });
});

describe("the breadcrumb", () => {
  it("offers the device type's other devices and sets a missing device", async () => {
    const saved: unknown[] = [];
    backend(SERVER, {
      commands: {
        scan_library: () => SCAN,
        save_project: (args) => {
          saved.push(args.spec.device);
          return projectView(args.dir, []);
        },
      },
    });
    renderShell("/library");
    await screen.findByRole("tree", { name: "Projects by device" });
    // A project with no device says so, and clicking it sets one.
    act(() => useProject.setState({ project: projectView("/data/scratch") }));
    fireEvent.click(await screen.findByRole("button", { name: "(no device)" }));
    const dialog = await screen.findByRole("dialog", { name: "Set device" });
    fireEvent.change(within(dialog).getByLabelText("Device id"), {
      target: { value: "SN9" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(saved).toEqual([{ id: "SN9" }]));
  });
});

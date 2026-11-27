// @vitest-environment jsdom
/** The Calibrate screen over a mock back end: the editor's groups and
 * its changed-from-default markers, the library's complaint on a field,
 * a run taking only the one-off patch, a failure said usefully, and
 * "Save as project default" as the one write of the project. */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../../api";
import type { ProjectView, RunInfo } from "../../api/generated/types";
import { createMockTransport, type MockTransport } from "../../api/transport/mock";
import { useProject } from "../../project/projectStore";
import { useRunEditor } from "../../project/runEditor";
import { queryCache } from "../../session/query";
import { useJobs } from "../../session/jobs";
import { calibratedRun, projectView } from "../../test/projectFixtures";
import planar from "./__fixtures__/config_form_planar_intrinsics.json";
import { CalibrateScreen } from ".";

const DIR = "/work/cell-a";

function project(over: Partial<ProjectView> = {}): ProjectView {
  return {
    ...projectView(DIR),
    config: { kind: "ok", config: planar.default },
    spec: {
      version: 2,
      dataset: { root: ".", manifest: "dataset.toml" },
    },
    ...over,
  };
}

interface Setup {
  mock: MockTransport;
  runs: RunInfo[];
}

function setup(
  options: {
    calibrate?: (args: unknown) => RunInfo | Promise<RunInfo>;
    view?: ProjectView;
  } = {},
): Setup {
  const written: RunInfo[] = [];
  const mock = createMockTransport({
    commands: {
      config_form: () => ({ schema: planar.schema, default: planar.default }),
      // The tests' patches are one field deep: enough to tell edits apart.
      config_patch: ({ config }) => {
        const c = config as { solver: { max_iters: number } };
        return c.solver.max_iters === planar.default.solver.max_iters
          ? {}
          : { solver: { max_iters: c.solver.max_iters } };
      },
      resolve_config: ({ patch }) => {
        const p = patch as { solver?: { max_iters?: number } };
        if (p.solver?.max_iters === 0) {
          throw new StudioError({
            kind: "config",
            message: "invalid input: solver.max_iters must be positive",
          });
        }
        return planar.default;
      },
      project_views: () => ({ cameras: ["cam0"], views: [], messages: [] }),
      list_runs: () => written,
      save_project: (args) => ({
        ...(options.view ?? project()),
        revision: "sha256:r2",
        spec: args.spec,
      }),
    },
    jobs: {
      calibrate: async (args) => {
        const run = await (options.calibrate?.(args) ??
          calibratedRun(DIR, `run-00${written.length + 1}`));
        written.push(run);
        return run;
      },
    },
  });
  setTransport(mock);
  useProject.setState({ project: options.view ?? project() });
  return { mock, runs: written };
}

function renderScreen() {
  return render(
    <TooltipProvider>
      <MemoryRouter>
        <CalibrateScreen />
      </MemoryRouter>
    </TooltipProvider>,
  );
}

const jobArgs = (mock: MockTransport) =>
  mock.calls.find((c) => c.command === "calibrate")?.args as
    | {
        dir: string;
        config_patch?: unknown;
        notes?: { label?: string };
        redetect?: boolean;
      }
    | undefined;

beforeEach(() => {
  queryCache.clear();
});

afterEach(() => {
  cleanup();
  setTransport(null);
  useProject.setState({ project: null, error: null });
  useRunEditor.setState({
    dir: null,
    form: null,
    edits: null,
    fromRun: null,
    failure: null,
    cancelled: false,
    starting: false,
    jobId: null,
    editorPatch: null,
    baseline: { kind: "default" },
  });
  useJobs.setState({ rows: [] });
});

async function editMaxIters(value: string) {
  const field = await screen.findByRole("spinbutton", { name: /max iters/i });
  fireEvent.change(field, { target: { value } });
  fireEvent.blur(field);
}

describe("the editor", () => {
  it("shows the curated groups first and every other field folded behind them", async () => {
    setup();
    renderScreen();
    expect(await screen.findByRole("heading", { name: "Camera" })).toBeTruthy();
    expect(screen.getByText("Fix and free")).toBeTruthy();
    expect(screen.getByText("Robust loss")).toBeTruthy();
    expect(screen.getByText("Solver and iterations")).toBeTruthy();
    expect(screen.getByText("No changes")).toBeTruthy();
    // The patch of the defaults is empty: the run takes the project's settings.
    expect(screen.getByText("the project's")).toBeTruthy();
  });

  it("marks an edited field as changed, counts it, and resets it", async () => {
    setup();
    renderScreen();
    await editMaxIters("20");
    await waitFor(() => expect(screen.getAllByText("1 change")).toHaveLength(2));
    expect(screen.getAllByText("changed").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Reset all" }));
    await waitFor(() => expect(screen.getByText("No changes")).toBeTruthy());
    expect(screen.queryAllByText("1 change")).toHaveLength(0);
  });

  it("says what the library says on the field, and will not run", async () => {
    setup();
    renderScreen();
    await editMaxIters("0");
    expect(await screen.findByText("solver.max_iters must be positive")).toBeTruthy();
    const run = screen.getByRole("button", { name: "Run" }) as HTMLButtonElement;
    expect(run.disabled).toBe(true);
    expect(screen.getByText(/A parameter does not check/)).toBeTruthy();
  });

  it("shows the seed as not available yet", async () => {
    setup();
    renderScreen();
    expect(await screen.findByText(/not available yet/)).toBeTruthy();
  });
});

describe("a run", () => {
  it("takes the edited fields as a one-off patch over the project's, with its label", async () => {
    const { mock } = setup();
    renderScreen();
    await editMaxIters("20");
    fireEvent.change(screen.getByLabelText("Label for this run"), {
      target: { value: "fewer iterations" },
    });
    const run = screen.getByRole("button", { name: "Run" }) as HTMLButtonElement;
    await waitFor(() => expect(run.disabled).toBe(false));
    fireEvent.click(run);
    await waitFor(() => expect(jobArgs(mock)).toBeDefined());
    expect(jobArgs(mock)).toMatchObject({
      dir: DIR,
      config_patch: { solver: { max_iters: 20 } },
      notes: { label: "fewer iterations" },
    });
    // The project file is not written by a run.
    expect(mock.calls.some((c) => c.command === "save_project")).toBe(false);
  });

  it("writes the project only for Save as project default", async () => {
    const { mock } = setup();
    renderScreen();
    await editMaxIters("20");
    const save = await screen.findByRole("button", { name: /Save as project default/ });
    await waitFor(() => expect((save as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(save);
    await waitFor(() =>
      expect(mock.calls.find((c) => c.command === "save_project")).toBeDefined(),
    );
    const saved = mock.calls.find((c) => c.command === "save_project")?.args as {
      revision: string;
      spec: { config: unknown };
    };
    expect(saved.revision).toBe("sha256:r1");
    expect(saved.spec.config).toEqual({ solver: { max_iters: 20 } });
    expect(mock.calls.some((c) => c.command === "calibrate")).toBe(false);
  });

  it("says what failed, where, and what to try, and keeps the settings", async () => {
    setup({
      calibrate: () => {
        throw new StudioError({
          kind: "calibration",
          message: "rig extrinsics: need 3, got 2 views",
        });
      },
    });
    renderScreen();
    await editMaxIters("20");
    const run = screen.getByRole("button", { name: "Run" }) as HTMLButtonElement;
    await waitFor(() => expect(run.disabled).toBe(false));
    fireEvent.click(run);
    expect(await screen.findByText("The calibration failed")).toBeTruthy();
    expect(screen.getByTestId("failure-message").textContent).toContain("need 3, got 2");
    expect(screen.getByText(/too few usable views/)).toBeTruthy();
    expect(screen.getByText(/still holds the settings that failed/)).toBeTruthy();
    expect(useRunEditor.getState().edits).not.toBeNull();
  });
});

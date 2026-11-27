// @vitest-environment jsdom
/** The shell's session with the back end: the address and the open
 * project say the same thing, the standing event stream keeps the project
 * and the jobs current (a `project_changed` from another tab refetches), the
 * job indicator and tray follow a job and cancel it, and a dropped
 * connection says so, pauses what writes, and recovers. */
import { act, cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { api, StudioError } from "../api";
import { useProject } from "../project/projectStore";
import { useStore } from "../store";
import { backend, renderShell, resetShellStores, SERVER } from "../test/shellKit";
import { calibratedRun, projectView } from "../test/projectFixtures";

afterEach(() => {
  cleanup();
  resetShellStores();
});

const DIR = "/work/cell-a";
const view = () =>
  projectView(DIR, [calibratedRun(DIR, "run-001"), calibratedRun(DIR, "run-002")]);
const where = () => screen.getByTestId("where").textContent;
const calls = (mock: ReturnType<typeof backend>, command: string) =>
  mock.calls.filter((c) => c.command === command);

describe("the address and the project", () => {
  it("opens the project and run an address names", async () => {
    const mock = backend(SERVER, { commands: { open_project: view } });
    renderShell(`/inspect/frames?project=${encodeURIComponent(DIR)}&run=run-001`);
    await waitFor(() => expect(useProject.getState().project?.dir).toBe(DIR));
    await waitFor(() => expect(useStore.getState().run?.id).toBe("run-001"));
    expect(calls(mock, "open_project")[0]!.args).toEqual({ dir: DIR });
    expect(calls(mock, "run_export")[0]!.args).toEqual({
      run: { dir: DIR, id: "run-001" },
    });
    expect(where()).toContain("run=run-001");
  });

  it("opens a project that has no run in its address on its newest run, and writes it", async () => {
    backend(SERVER, { commands: { open_project: view } });
    renderShell(`/inspect/frames?project=${encodeURIComponent(DIR)}`);
    await waitFor(() => expect(useStore.getState().run?.id).toBe("run-002"));
    await waitFor(() => expect(where()).toContain("run=run-002"));
  });

  it("writes a project opened another way, and the run chosen, to the address", async () => {
    backend(SERVER, { commands: { open_project: view } });
    renderShell("/library");
    await screen.findByText("library body");
    await act(async () => {
      await useProject.getState().open(DIR);
    });
    await waitFor(() =>
      expect(where()).toContain(
        `project=${encodeURIComponent(DIR).replace(/%20/g, "+")}`,
      ),
    );
    await act(async () => {
      await useProject.getState().openRun("run-001");
    });
    await waitFor(() => expect(where()).toContain("run=run-001"));
  });

  it("drops the project from the address when it is closed", async () => {
    backend(SERVER, { commands: { open_project: view } });
    renderShell(`/library?project=${encodeURIComponent(DIR)}`);
    await waitFor(() => expect(useProject.getState().project?.dir).toBe(DIR));
    act(() => useProject.getState().close());
    await waitFor(() => expect(where()).toBe("/library"));
  });

  it("keeps the screen when the project changes", async () => {
    backend(SERVER, {
      commands: {
        open_project: ({ dir }) => projectView(dir, [calibratedRun(dir, "run-001")]),
      },
    });
    renderShell(`/inspect/epipolar?project=${encodeURIComponent(DIR)}`);
    await screen.findByText("inspect body");
    await act(async () => {
      await useProject.getState().open("/work/cell-b");
    });
    await waitFor(() => expect(where()).toContain("project=%2Fwork%2Fcell-b"));
    expect(where()).toContain("/inspect/epipolar?");
    expect(screen.getByText("inspect body")).toBeTruthy();
  });
});

describe("the standing event stream", () => {
  async function openProjectShell() {
    const mock = backend(SERVER, { commands: { open_project: view } });
    renderShell(`/calibrate?project=${encodeURIComponent(DIR)}`);
    await waitFor(() => expect(useProject.getState().project?.dir).toBe(DIR));
    await screen.findByText("calibrate body");
    return mock;
  }

  it("refetches the open project when another tab changes it", async () => {
    const mock = await openProjectShell();
    const before = calls(mock, "open_project").length;
    act(() => mock.emit({ kind: "project_changed", dir: DIR }));
    await waitFor(() => expect(calls(mock, "open_project").length).toBe(before + 1), {
      timeout: 2000,
    });
  });

  it("gathers a burst of changes into one refetch, and ignores another project's", async () => {
    const mock = await openProjectShell();
    const before = calls(mock, "open_project").length;
    act(() => {
      mock.emit({ kind: "project_changed", dir: "/elsewhere" });
      mock.emit({ kind: "project_changed", dir: DIR });
      mock.emit({ kind: "project_changed", dir: DIR });
      mock.emit({ kind: "project_changed", dir: DIR });
    });
    await waitFor(() => expect(calls(mock, "open_project").length).toBe(before + 1), {
      timeout: 2000,
    });
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(calls(mock, "open_project").length).toBe(before + 1);
  });

  it("is subscribed with no job running, and follows one that starts elsewhere", async () => {
    const mock = backend();
    renderShell("/library");
    await screen.findByText("library body");
    // Nothing yet: no indicator.
    expect(screen.queryByRole("button", { name: /^Jobs/ })).toBeNull();
    await waitFor(() => expect(calls(mock, "list_jobs").length).toBeGreaterThan(0));
    act(() =>
      mock.emit({
        kind: "job",
        id: "7-abc",
        command: "calibrate",
        state: { kind: "running" },
        event: { kind: "detection", done: 1, total: 4 },
      }),
    );
    expect(
      await screen.findByRole("button", { name: "Jobs: Detecting 1/4" }),
    ).toBeTruthy();
  });

  it("lists the jobs that were already running when the page loaded", async () => {
    backend(SERVER, {
      commands: {
        list_jobs: () => [
          {
            id: "3-old",
            command: "calibrate",
            state: { kind: "running" },
            last_event: { kind: "iteration", iter: 12, max_iters: 50, cost: 0.32 },
          },
        ],
      },
    });
    renderShell("/library");
    expect(
      await screen.findByRole("button", {
        name: "Jobs: Solving · iter 12 · cost 3.2e−1",
      }),
    ).toBeTruthy();
  });
});

describe("the job tray", () => {
  it("opens from the indicator, shows the job's stage, and cancels it", async () => {
    const mock = backend(SERVER, {
      jobs: {
        calibrate: async (_args, control) => {
          control.emit({ kind: "detection", done: 41, total: 120 });
          await control.cancelled;
          throw new StudioError({ kind: "cancelled", message: "cancelled" });
        },
      },
    });
    renderShell("/library");
    await screen.findByText("library body");
    await waitFor(() => expect(calls(mock, "list_jobs").length).toBeGreaterThan(0));
    await act(async () => {
      const job = await api.jobs.calibrate({ dir: DIR });
      // Its cancellation is this test's, not an unhandled rejection.
      job.result.catch(() => {});
    });
    fireEvent.click(await screen.findByRole("button", { name: /^Jobs/ }));
    const tray = await screen.findByRole("region", { name: "Jobs" });
    await waitFor(() => expect(tray.textContent).toContain("Detecting 41/120"));
    expect(tray.textContent).toContain("Calibration");
    expect(
      screen.getByRole("progressbar", { name: "Calibration progress" }),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(calls(mock, "cancel_job").length).toBe(1));
    await waitFor(() => expect(tray.textContent).toContain("Cancelled"));
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("keeps a failure, with its typed error, until it is dismissed", async () => {
    const mock = backend();
    renderShell("/library");
    await screen.findByText("library body");
    await waitFor(() => expect(calls(mock, "list_jobs").length).toBeGreaterThan(0));
    act(() =>
      mock.emit({
        kind: "job",
        id: "9-bad",
        command: "calibrate",
        state: {
          kind: "failed",
          error: { kind: "calibration", message: "the solver did not converge" },
        },
      }),
    );
    fireEvent.click(await screen.findByRole("button", { name: /^Jobs/ }));
    expect(await screen.findByText(/the solver did not converge/)).toBeTruthy();
    expect(screen.getByText("(calibration)")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /^Dismiss Calibration/ }));
    expect(screen.queryByText(/the solver did not converge/)).toBeNull();
  });
});

describe("a lost connection", () => {
  it("says so, pauses what writes, and recovers by itself with everything read again", async () => {
    const mock = backend(SERVER, { commands: { open_project: view } });
    renderShell(`/calibrate?project=${encodeURIComponent(DIR)}`);
    await waitFor(() => expect(useProject.getState().project?.dir).toBe(DIR));
    await screen.findByText("calibrate body");
    expect(screen.queryByText(/Connection lost/)).toBeNull();

    act(() => mock.setConnected(false));
    expect(await screen.findByText(/Connection lost/)).toBeTruthy();
    expect(screen.getByText(/Reconnecting…/)).toBeTruthy();

    const jobsBefore = calls(mock, "list_jobs").length;
    const openBefore = calls(mock, "open_project").length;
    act(() => mock.setConnected(true));
    await waitFor(() => expect(screen.queryByText(/Connection lost/)).toBeNull());
    await waitFor(() =>
      expect(calls(mock, "list_jobs").length).toBeGreaterThan(jobsBefore),
    );
    await waitFor(() =>
      expect(calls(mock, "open_project").length).toBeGreaterThan(openBefore),
    );
  });

  it("is told by a command that never reached the back end", async () => {
    const mock = backend(SERVER, { commands: { open_project: view } });
    renderShell(`/calibrate?project=${encodeURIComponent(DIR)}`);
    await waitFor(() => expect(useProject.getState().project?.dir).toBe(DIR));
    mock.setReachable(false);
    await act(async () => {
      await useProject.getState().reload();
    });
    expect(await screen.findByText(/Connection lost/)).toBeTruthy();
  });
});

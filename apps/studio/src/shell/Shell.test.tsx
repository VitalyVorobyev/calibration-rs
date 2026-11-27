// @vitest-environment jsdom
/** The shell: the rail of seven screens (disabled where a screen is not
 * built or needs a project), the context bar's project and run with their
 * pills, project errors, the sign-in panel, the desktop's first folder, a
 * back end that does not answer, and the global keys. The back end is a
 * mock transport. */
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { StudioError } from "../api";
import type { ServerInfo } from "../api/generated/types";
import { useProject } from "../project/projectStore";
import { backend, renderShell, resetShellStores, SERVER } from "../test/shellKit";
import { calibratedRun, DIGEST, projectView } from "../test/projectFixtures";

afterEach(() => {
  cleanup();
  resetShellStores();
});

const OPEN = (dir = "/work/cell-a") =>
  projectView(dir, [calibratedRun(dir, "run-001"), calibratedRun(dir, "run-002")]);

describe("the rail", () => {
  it("lists the seven screens in the workflow's order, the current one marked", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    const rail = screen.getByRole("navigation", { name: "Screens" });
    const names = [...rail.querySelectorAll("a")].map((a) => a.textContent?.trim());
    expect(names).toEqual([
      "Library",
      "Dataset",
      "Detect",
      "Calibrate",
      "Inspect",
      "Compare",
      "Device",
    ]);
    expect(
      screen.getByRole("link", { name: "Library" }).getAttribute("aria-current"),
    ).toBe("page");
  });

  it("disables a screen that is not built, and one that needs a project, and says why", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    const dataset = screen.getByRole("link", { name: "Dataset" });
    expect(dataset.getAttribute("aria-disabled")).toBe("true");
    expect(dataset.getAttribute("title")).toMatch(/coming in a later release/i);
    const inspect = screen.getByRole("link", { name: "Inspect" });
    expect(inspect.getAttribute("aria-disabled")).toBe("true");
    expect(inspect.getAttribute("title")).toBe("Open a project first");
    expect(
      screen.getByRole("link", { name: "Library" }).getAttribute("aria-disabled"),
    ).toBe(null);

    act(() => useProject.setState({ project: OPEN() }));
    expect(
      screen.getByRole("link", { name: "Inspect" }).getAttribute("aria-disabled"),
    ).toBe(null);
    expect(
      screen.getByRole("link", { name: "Calibrate" }).getAttribute("aria-disabled"),
    ).toBe(null);
    // Not built yet, project or not.
    expect(
      screen.getByRole("link", { name: "Compare" }).getAttribute("aria-disabled"),
    ).toBe("true");
  });

  it("keeps the project and run in the address of each link", async () => {
    backend(SERVER, { commands: { open_project: () => OPEN() } });
    renderShell("/library?project=%2Fwork%2Fcell-a&run=run-001");
    await screen.findByRole("tree", { name: "Saved runs" }).catch(() => null);
    await waitFor(() =>
      expect(
        screen.getByRole("link", { name: "Inspect" }).getAttribute("href"),
      ).toContain("project=%2Fwork%2Fcell-a"),
    );
  });
});

describe("the landmarks", () => {
  it("names the regions and offers a way past the bar and the rail to the screen", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    expect(screen.getAllByRole("banner").length).toBeGreaterThan(0);
    expect(screen.getByRole("navigation", { name: "Screens" })).toBeTruthy();
    expect(screen.getByRole("main")).toBeTruthy();
    expect(screen.getByLabelText("Status")).toBeTruthy();
    // Each screen has its one h1.
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);

    const skip = screen.getByRole("button", { name: "Skip to the screen" });
    fireEvent.click(skip);
    expect(document.activeElement?.id).toBe("screen");
  });

  it("names a screen that has no title of its own", async () => {
    backend();
    renderShell("/calibrate");
    await screen.findByText(/Calibrate works on a project/);
    expect(screen.getByRole("heading", { level: 1, name: "Calibrate" })).toBeTruthy();
  });
});

describe("the context bar", () => {
  it("names the project, its device and the run with its state, verdict and staleness", async () => {
    backend();
    renderShell("/inspect/frames");
    await screen.findByText("inspect body").catch(() => null);
    const view = OPEN();
    view.spec = {
      ...view.spec,
      device: { id: "SN-17", product: "Stereo-X1", label: "bench rig" },
    };
    view.input_digest = { ...DIGEST, config: "sha256:changed" };
    act(() => useProject.setState({ project: view }));
    expect(screen.getByText("Stereo-X1")).toBeTruthy();
    expect(screen.getByText("bench rig")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Project" }).textContent).toContain(
      "cell-a",
    );
    // No run shown yet: the selector invites a choice.
    expect(screen.getByRole("combobox", { name: "Run" }).textContent).toContain(
      "Choose a run",
    );
  });

  it("says (no device) quietly for a project without one", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    act(() => useProject.setState({ project: OPEN() }));
    expect(screen.getByText("(no device)")).toBeTruthy();
  });

  it("shows the run's pills once it is the current run", async () => {
    const view = OPEN();
    backend(SERVER, {
      commands: {
        open_project: () => ({
          ...view,
          input_digest: { ...DIGEST, config: "sha256:changed" },
        }),
      },
    });
    renderShell("/inspect/frames?project=%2Fwork%2Fcell-a&run=run-002");
    await waitFor(() => expect(screen.getByText("Recorded")).toBeTruthy());
    // The context bar's pills; the run list's cards say the same of each run.
    const pills = within(screen.getByLabelText("State of the run"));
    expect(pills.getByText("Pass")).toBeTruthy();
    expect(pills.getByText("Stale")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Run" }).textContent).toContain(
      "run-002",
    );
  });
});

describe("errors", () => {
  it("shows a project error until it is dismissed", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    act(() =>
      useProject.setState({
        error: { kind: "not_a_project", message: "/tmp is not a calibration project" },
      }),
    );
    expect(screen.getByText("Not a project folder")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("Not a project folder")).toBeNull();
  });

  it("says a screen needs a project, and links to the Library", async () => {
    backend();
    renderShell("/calibrate");
    expect(await screen.findByText(/Calibrate works on a project/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open from the Library" })).toBeTruthy();
  });

  it("says so when no back end answers, with Retry", async () => {
    backend(SERVER, {
      commands: {
        server_info: () => {
          throw new StudioError({ kind: "internal", message: "Failed to fetch" });
        },
      },
    });
    renderShell("/library");
    await screen.findByText("No back end answers");
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });
});

describe("the sign-in panel", () => {
  it("is all the page shows while the server wants its token, and logs in", async () => {
    let authorized = false;
    const mock = backend(SERVER, {
      commands: {
        server_info: () => {
          if (!authorized)
            throw new StudioError({ kind: "unauthorized", message: "token" });
          return SERVER;
        },
      },
    });
    mock.login = (token: string) => {
      authorized = token === "secret-token";
      return authorized
        ? Promise.resolve()
        : Promise.reject(new StudioError({ kind: "unauthorized", message: "no" }));
    };
    renderShell("/library");
    const field = await screen.findByLabelText("Token");
    // No rail, no bar: nothing but the panel.
    expect(screen.queryByRole("navigation", { name: "Screens" })).toBeNull();
    expect(screen.queryByText("library body")).toBeNull();
    expect(screen.getByText(/calibration-studio-server token/)).toBeTruthy();
    fireEvent.change(field, { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await screen.findByText("That is not this server's token.");
    fireEvent.change(field, { target: { value: "secret-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await screen.findByText("library body");
    expect(screen.getByRole("navigation", { name: "Screens" })).toBeTruthy();
  });
});

describe("the first run on the desktop", () => {
  it("asks for a first folder in the Library and adds the one picked", async () => {
    const desktop: ServerInfo = {
      ...SERVER,
      shell: "desktop",
      roots: [{ path: "/cache/samples", samples: true }],
      roots_editable: true,
    };
    const mock = backend(desktop, {
      nativeFolder: () => "/home/me/data",
      commands: {
        add_root: ({ path }) => ({
          ...desktop,
          roots: [...desktop.roots, { path, samples: false }],
        }),
      },
    });
    renderShell("/library");
    expect(await screen.findByText(/Add a folder that holds your projects/)).toBeTruthy();
    fireEvent.click(await screen.findByRole("button", { name: "Add folder…" }));
    await waitFor(() =>
      expect(mock.calls.some((c) => c.command === "add_root")).toBe(true),
    );
    await waitFor(() =>
      expect(screen.queryByText(/Add a folder that holds your projects/)).toBeNull(),
    );
  });
});

describe("the global keys", () => {
  it("open the shortcut list with ? and the palette with ⌘K", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    fireEvent.keyDown(window, { key: "?", code: "Slash", shiftKey: true });
    expect(
      await screen.findByRole("dialog", { name: "Keyboard shortcuts" }),
    ).toBeTruthy();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    fireEvent.keyDown(window, { key: "k", code: "KeyK", metaKey: true });
    expect(await screen.findByRole("dialog", { name: "Command palette" })).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Search commands" })).toBeTruthy();
  });

  it("switch screens with ⌘ and the digit, but not to a disabled one", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    fireEvent.keyDown(window, { key: "4", code: "Digit4", metaKey: true });
    // Calibrate needs a project: nothing happens.
    expect(screen.getByTestId("where").textContent).toBe("/library");
    act(() => useProject.setState({ project: OPEN() }));
    fireEvent.keyDown(window, { key: "4", code: "Digit4", metaKey: true });
    await waitFor(() =>
      expect(screen.getByTestId("where").textContent).toContain("/calibrate"),
    );
  });

  it("filter the palette and jump to a screen", async () => {
    backend();
    renderShell("/library");
    await screen.findByText("library body");
    act(() => useProject.setState({ project: OPEN() }));
    fireEvent.keyDown(window, { key: "k", code: "KeyK", metaKey: true });
    const input = await screen.findByRole("combobox", { name: "Search commands" });
    fireEvent.change(input, { target: { value: "calib" } });
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "CalibrateScreens",
    ]);
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() =>
      expect(screen.getByTestId("where").textContent).toContain("/calibrate"),
    );
  });
});

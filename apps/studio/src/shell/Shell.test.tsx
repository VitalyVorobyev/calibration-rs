// @vitest-environment jsdom
/** The shell: the workspace rail marks the current route, the project
 * navigator appears with an open project and lists its runs, a project
 * error is shown until dismissed; the session asks for the token when the
 * server wants it and for a first folder on the desktop's first run. The
 * back end is a mock transport. */
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../api";
import type { ServerInfo } from "../api/generated/types";
import { createMockTransport, type MockOptions } from "../api/transport/mock";
import { useProject } from "../project/projectStore";
import { useSession } from "./session";
import { Shell } from "./Shell";

const SERVER: ServerInfo = {
  version: "0.1.0",
  shell: "server",
  roots: [{ path: "/data", samples: false }],
  roots_editable: false,
  samples: true,
};

function backend(info: ServerInfo = SERVER, options: MockOptions = {}) {
  const mock = createMockTransport({
    ...options,
    commands: { server_info: () => info, recent_projects: () => [], ...options.commands },
  });
  setTransport(mock);
  return mock;
}

function renderShell() {
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: <Shell />,
        children: [{ path: "diagnose", element: <p>diagnose body</p> }],
      },
    ],
    { initialEntries: ["/diagnose"] },
  );
  return render(
    <TooltipProvider>
      <RouterProvider router={router} />
    </TooltipProvider>,
  );
}

beforeEach(() => {
  backend();
});

afterEach(() => {
  setTransport(null);
  cleanup();
  useProject.setState({ project: null, error: null });
  useSession.setState({ status: { kind: "connecting" } });
});

describe("Shell", () => {
  it("renders the workspace rail with the current route marked", () => {
    renderShell();
    expect(screen.getByText("diagnose body")).toBeTruthy();
    for (const name of [
      "Diagnose",
      "3D viewer",
      "Epipolar",
      "Depth (dense stereo)",
      "Run calibration",
    ]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
    expect(
      screen.getByRole("link", { name: "Diagnose" }).getAttribute("aria-current"),
    ).toBe("page");
    expect(screen.queryByRole("tree", { name: "Saved runs" })).toBeNull();
  });

  it("shows the project navigator with its runs while a project is open", () => {
    renderShell();
    act(() =>
      useProject.setState({
        project: {
          dir: "/work/cell-a",
          dataset_dir: "/work/cell-a",
          manifest: {},
          runs: [
            {
              name: "run-001",
              diagnostics: {
                topology: "planar_intrinsics",
                usable_views: 9,
                total_views: 10,
                duration_ms: 1,
                cache_used: false,
              },
            },
          ],
        },
      }),
    );
    expect(screen.getByRole("tree", { name: "Saved runs" })).toBeTruthy();
    expect(screen.getByRole("treeitem", { name: /run-001/ })).toBeTruthy();
    expect(screen.getByText("9/10 views")).toBeTruthy();
    // The header's project menu names the project.
    expect(screen.getAllByText("cell-a").length).toBeGreaterThan(0);
  });

  it("shows a project error until it is dismissed", () => {
    renderShell();
    act(() =>
      useProject.setState({
        error: { kind: "not_a_project", message: "/tmp is not a calibration project" },
      }),
    );
    expect(screen.getByText("Not a project folder")).toBeTruthy();
    expect(screen.getByText("/tmp is not a calibration project")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("Not a project folder")).toBeNull();
  });

  it("offers every workspace and the project commands on any back end", () => {
    renderShell();
    for (const name of [
      "Diagnose",
      "3D viewer",
      "Epipolar",
      "Depth (dense stereo)",
      "Run calibration",
    ]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
    expect(screen.getByRole("button", { name: "Open Export…" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Settings" })).toBeTruthy();
  });

  it("asks for the token when the server wants it, and logs in", async () => {
    let authorized = false;
    const mock = backend(SERVER, {
      commands: {
        server_info: () => {
          if (!authorized) {
            throw new StudioError({ kind: "unauthorized", message: "token" });
          }
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
    renderShell();
    const field = await screen.findByLabelText("Token");
    // The workspace stays hidden until the session is open.
    expect(screen.queryByText("diagnose body")).toBeNull();
    fireEvent.change(field, { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    await screen.findByText("That is not this server's token.");
    fireEvent.change(field, { target: { value: "secret-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    await screen.findByText("diagnose body");
  });

  it("asks the desktop for a first folder and adds the one picked", async () => {
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
    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: "Add folder…" }));
    await waitFor(() =>
      expect(mock.calls.some((c) => c.command === "add_root")).toBe(true),
    );
    await waitFor(() => expect(screen.queryByText("Add a folder to start")).toBeNull());
  });

  it("says so when no back end answers, and still shows the workspace", async () => {
    backend(SERVER, {
      commands: {
        server_info: () => {
          throw new StudioError({ kind: "internal", message: "Failed to fetch" });
        },
      },
    });
    renderShell();
    await screen.findByText("No back end answers");
    expect(screen.getByText("diagnose body")).toBeTruthy();
  });
});

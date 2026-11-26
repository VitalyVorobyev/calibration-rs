// @vitest-environment jsdom
/** The shell: the workspace rail marks the current route, the project
 * navigator appears with an open project and lists its runs, a project
 * error is shown until dismissed, and the browser viewer offers no desktop
 * features. The back end is a mock transport of the desktop app's kind. */
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport } from "../api";
import { createMockTransport } from "../api/transport/mock";
import { useProject } from "../project/projectStore";
import { Shell } from "./Shell";

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
  setTransport(createMockTransport());
});

afterEach(() => {
  setTransport(null);
  cleanup();
  useProject.setState({ project: null, error: null });
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

  it("offers no desktop features in the browser viewer", () => {
    setTransport(createMockTransport({ kind: "http" }));
    renderShell();
    for (const name of ["Diagnose", "3D viewer", "Epipolar", "Depth (dense stereo)"]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
    expect(screen.queryByRole("link", { name: "Run calibration" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Open Export…" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Project/ })).toBeNull();
    expect(screen.getByText("No export loaded")).toBeTruthy();
  });
});

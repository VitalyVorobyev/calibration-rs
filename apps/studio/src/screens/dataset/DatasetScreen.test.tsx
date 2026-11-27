// @vitest-environment jsdom
/** The Dataset screen over a mock back end: the description's form, the
 * library's messages under the fields they are about (and the app's own
 * warning for a pattern that matched nothing), Save refused while there are
 * errors and sent over the project's revision otherwise, and a manifest that
 * is a file of its own shown read-only. */
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
import { afterEach, describe, expect, it } from "vitest";
import type { DatasetSpec, ProjectView, ProjectViews } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { projectView } from "../../test/projectFixtures";
import { backend, resetShellStores } from "../../test/shellKit";
import { DatasetScreen } from "./index";

afterEach(() => {
  cleanup();
  resetShellStores();
});

const SCHEMA = {
  type: "object",
  properties: {
    topology: {
      type: "string",
      enum: ["planar_intrinsics", "rig_extrinsics"],
    },
    target: {
      type: "object",
      properties: { rows: { type: "integer", minimum: 0 } },
      required: ["rows"],
    },
    cameras: { type: "array", items: { type: "object" } },
  },
  required: ["topology", "target", "cameras"],
};

const REPLY: ProjectViews = {
  cameras: ["cam0"],
  views: [
    { token: "0", images: ["img/a.png"] },
    { token: "1", images: ["img/b.png"] },
  ],
  messages: [],
};

function open(view: ProjectView = projectView("/work/cell")) {
  act(() => useProject.setState({ project: view }));
}

function mount(
  handlers: { views?: (spec: unknown) => ProjectViews; saved?: unknown[] } = {},
) {
  backend(undefined, {
    commands: {
      dataset_form: () => ({ schema: SCHEMA, default: {} }),
      project_views: ({ spec }) => (handlers.views ?? (() => REPLY))(spec),
      save_project: (args) => {
        handlers.saved?.push(args);
        return { ...projectView(args.dir), revision: "sha256:r2", spec: args.spec };
      },
    },
  });
  return render(
    <TooltipProvider>
      <DatasetScreen />
    </TooltipProvider>,
  );
}

describe("the Dataset screen", () => {
  it("shows the cameras and views the library found, and what the selected cell holds", async () => {
    mount();
    open();
    const matrix = await screen.findByRole("grid", { name: "Camera by view" });
    expect(within(matrix).getByRole("columnheader", { name: "cam0" })).toBeTruthy();
    expect(screen.getByText("2 views")).toBeTruthy();
    // The camera's pattern, with what it matches.
    await waitFor(() =>
      expect(document.body.textContent).toContain("2 files · first img/a.png"),
    );
    fireEvent.click(
      within(matrix).getByRole("gridcell", { name: /view 1, camera cam0/ }),
    );
    expect(
      await within(
        screen.getByRole("complementary", { name: "Selected image" }),
      ).findByText("img/b.png"),
    ).toBeTruthy();
  });

  it("names the robot pose and laser frame the selected view pairs with", async () => {
    mount({
      views: () => ({
        ...REPLY,
        views: [
          { token: "0", images: ["img/a.png"], pose: { index: 0 }, laser_frames: [null] },
          { token: "1", images: ["img/b.png"], laser_frames: ["laser/b.png"] },
        ],
        poses: { unpaired: [{ index: 2 }] },
        laser_frames: { unpaired: [] },
      }),
    });
    const view = projectView("/work/cell");
    const manifest = {
      ...(view.spec.dataset.manifest as DatasetSpec),
      robot_poses: { path: "poses.txt", format: "rowmajor4x4" },
    } as DatasetSpec;
    open({
      ...view,
      spec: { ...view.spec, dataset: { root: ".", manifest } },
      dataset: { kind: "ok", spec: manifest },
    });
    const matrix = await screen.findByRole("grid", { name: "Camera by view" });
    const detail = screen.getByRole("complementary", { name: "Selected image" });
    fireEvent.click(
      within(matrix).getByRole("gridcell", { name: /view 0, camera cam0/ }),
    );
    expect((await within(detail).findByTestId("view-pose")).textContent).toBe(
      "row 0 of poses.txt",
    );
    expect(within(detail).getByTestId("view-laser").textContent).toBe("none for cam0");
    fireEvent.click(
      within(matrix).getByRole("gridcell", { name: /view 1, camera cam0/ }),
    );
    expect((await within(detail).findByTestId("view-laser")).textContent).toBe(
      "laser/b.png",
    );
    expect(within(detail).getByTestId("view-pose").textContent).toBe(
      "no pose pairs with this view",
    );
  });

  it("places the library's message under its field, refuses Save, and counts the error", async () => {
    mount({
      views: (spec) =>
        JSON.stringify(spec).includes('"rows":0')
          ? {
              ...REPLY,
              messages: [
                {
                  kind: "manifest",
                  field: "target",
                  message: "chessboard requires rows >= 2",
                },
              ],
            }
          : REPLY,
    });
    open();
    const rows = await screen.findByRole("spinbutton", { name: /^Rows/ });
    fireEvent.change(rows, { target: { value: "0" } });
    expect(
      await screen.findByText("chessboard requires rows >= 2", {}, { timeout: 3000 }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Save" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(screen.getByText("· 1 error")).toBeTruthy();
  });

  it("saves the edited manifest over the revision the project was opened at", async () => {
    const saved: unknown[] = [];
    mount({ saved });
    open();
    const rows = await screen.findByRole("spinbutton", { name: /^Rows/ });
    fireEvent.change(rows, { target: { value: "9" } });
    const save = screen.getByRole("button", { name: "Save" });
    await waitFor(() => expect(save.hasAttribute("disabled")).toBe(false), {
      timeout: 3000,
    });
    fireEvent.click(save);
    await waitFor(() => expect(saved).toHaveLength(1));
    const sent = saved[0] as {
      revision: string;
      spec: { dataset: { manifest: { target: { rows: number } } } };
    };
    expect(sent.revision).toBe("sha256:r1");
    expect(sent.spec.dataset.manifest.target.rows).toBe(9);
    // Saved: the draft starts again from the project the save answered.
    await waitFor(() => expect(screen.getByText("Saved")).toBeTruthy());
  });

  it("warns, without stopping Save, when a camera's pattern matches nothing", async () => {
    mount({ views: () => ({ ...REPLY, views: [] }) });
    open();
    expect(await screen.findByText("No image matches “img/*.png”.")).toBeTruthy();
    expect(await screen.findByText(/Nothing matches “img\/\*\.png”/)).toBeTruthy();
  });

  it("shows a manifest that is a file of its own read-only, with the way to copy it in", async () => {
    mount();
    const view = projectView("/work/cell");
    open({
      ...view,
      dataset_editable: false,
      spec: { ...view.spec, dataset: { manifest: "dataset.toml" } },
    });
    expect(
      await screen.findByText("Read-only: the manifest is a file of its own", {
        selector: "p",
      }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Copy into the project to edit" }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Save" })).toBeNull();
    expect(
      screen.getByRole("spinbutton", { name: /^Rows/ }).hasAttribute("disabled"),
    ).toBe(true);
  });
});

// @vitest-environment jsdom
/** The Detect screen's exclusions over a mock back end: every kind written to
 * the project's `[exclude]` with undo, a retry over a changed file, the panel
 * with notes and Restore, and nothing written in read-only mode. */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useProject } from "../../project/projectStore";
import { useJobs } from "../../session/jobs";
import { projectView } from "../../test/projectFixtures";
import {
  cleanupDetect,
  DIR,
  matrix,
  mount,
  press,
  resetDetect,
  saves,
  select,
} from "./detectKit";

beforeEach(resetDetect);
afterEach(cleanupDetect);

describe("exclusions", () => {
  it("X leaves out this camera's image, ⇧X the whole view", async () => {
    const mock = mount();
    await select(/view 0001, camera cam1/);
    press("x");
    await waitFor(() => expect(saves(mock)).toHaveLength(1));
    expect(saves(mock)[0]!.spec.exclude).toEqual({
      images: [{ camera: "cam1", image: "b/0001.png" }],
    });
    expect(saves(mock)[0]!.revision).toBe("sha256:r1");
    // The matrix marks it.
    expect(
      await within(await matrix()).findByRole("gridcell", {
        name: /view 0001, camera cam1, 20:.*Excluded/,
      }),
    ).toBeTruthy();
    press("x", { shiftKey: true });
    await waitFor(() => expect(saves(mock)).toHaveLength(2));
    expect(saves(mock)[1]!.spec.exclude).toEqual({
      images: [{ camera: "cam1", image: "b/0001.png" }],
      views: [{ view: { token: "0001" } }],
    });
    expect(await within(await matrix()).findByText("view excluded")).toBeTruthy();
  });

  it("leaves out the picked features by their board point", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    fireEvent.click(screen.getByRole("checkbox", { name: "Pick feature 1,0" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Pick feature 1,1" }));
    fireEvent.click(screen.getByRole("button", { name: "Exclude 2 features" }));
    await waitFor(() => expect(saves(mock)).toHaveLength(1));
    expect(saves(mock)[0]!.spec.exclude).toEqual({
      features: [
        {
          camera: "cam0",
          image: "a/0001.png",
          points: [
            [0.02, 0, 0],
            [0.02, 0.02, 0],
          ],
        },
      ],
    });
    // They stay in the table, marked.
    const table = screen.getByRole("table", { name: "Features of this image" });
    await waitFor(() => expect(within(table).getAllByText("excluded")).toHaveLength(2));
  });

  it("undoes the last exclusion with ⌘Z and from the tray", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    press("x");
    await waitFor(() => expect(saves(mock)).toHaveLength(1));
    const undo = screen.getByRole("button", { name: "Undo" });
    await waitFor(() => expect((undo as HTMLButtonElement).disabled).toBe(false));
    press("z", { metaKey: true });
    await waitFor(() => expect(saves(mock)).toHaveLength(2));
    expect(saves(mock)[1]!.spec.exclude).toEqual({});
    await waitFor(() => expect((undo as HTMLButtonElement).disabled).toBe(true));
  });

  it("tries once more on the file the disk now holds when it changed under the save", async () => {
    const mock = mount({ stale: 1 });
    await select(/view 0001, camera cam0/);
    press("x");
    await waitFor(() => expect(saves(mock)).toHaveLength(2));
    // The retry was over the disk's revision.
    expect(saves(mock)[1]!.revision).toBe("sha256:disk0");
    expect(mock.calls.some((c) => c.command === "open_project")).toBe(true);
    expect(useProject.getState().error).toBeNull();
    expect(useProject.getState().project?.spec.exclude).toEqual({
      images: [{ camera: "cam0", image: "a/0001.png" }],
    });
  });

  it("lists every exclusion with its note and restores them", async () => {
    const mock = mount({
      view: {
        ...projectView(DIR),
        spec: {
          ...projectView(DIR).spec,
          exclude: {
            views: [{ view: { token: "0002" }, note: "glare" }],
            images: [{ camera: "cam0", image: "a/0001.png" }],
            features: [{ camera: "cam0", image: "a/0001.png", points: [[0, 0, 0]] }],
          },
        },
      },
    });
    await matrix();
    expect(screen.getByRole("region", { name: "Exclusions" }).textContent).toContain(
      "1 view · 1 image · 1 feature",
    );
    fireEvent.click(screen.getByRole("button", { name: /Exclusions/ }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getAllByRole("row")).toHaveLength(4);
    expect(within(dialog).getByDisplayValue("glare")).toBeTruthy();
    // A note is saved when the field loses focus.
    const note = within(dialog).getByRole("textbox", {
      name: /Why image a\/0001.png is excluded/,
    });
    fireEvent.change(note, { target: { value: "blurred" } });
    fireEvent.blur(note);
    await waitFor(() => expect(saves(mock)).toHaveLength(1));
    expect(saves(mock)[0]!.spec.exclude).toMatchObject({
      images: [{ camera: "cam0", image: "a/0001.png", note: "blurred" }],
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Restore view 0002" }));
    await waitFor(() => expect(saves(mock)).toHaveLength(2));
    expect(saves(mock)[1]!.spec.exclude).not.toHaveProperty("views");
    fireEvent.click(within(dialog).getByRole("button", { name: "Restore all" }));
    await waitFor(() => expect(saves(mock)).toHaveLength(3));
    expect(saves(mock)[2]!.spec.exclude).toEqual({});
  });

  it("writes nothing in read-only mode", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    act(() => useJobs.setState({ connection: "lost" }));
    press("x");
    press("x", { shiftKey: true });
    for (const name of ["Exclude image", "Exclude view"]) {
      expect(screen.getByRole("button", { name }).hasAttribute("disabled")).toBe(true);
    }
    expect(screen.getByRole("button", { name: "Undo" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(saves(mock)).toHaveLength(0);
  });

  it("does not fire while typing in a field", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    const search = screen.getByRole("searchbox", { name: "Find a view by its token" });
    search.focus();
    fireEvent.keyDown(search, { key: "x", code: "KeyX" });
    expect(saves(mock)).toHaveLength(0);
  });
});

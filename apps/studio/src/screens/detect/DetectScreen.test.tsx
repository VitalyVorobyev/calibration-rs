// @vitest-environment jsdom
/** The Detect screen over a mock back end: the matrix of counts and why an
 * image was left out, Detect all as a job with progress and Cancel, the
 * features of the selected image, and trying other detector settings (a
 * preview writes nothing, Apply saves them into the project). */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useJobs } from "../../session/jobs";
import { projectView } from "../../test/projectFixtures";
import { overview } from "./__fixtures__/details";
import {
  cleanupDetect,
  DIR,
  matrix,
  mount,
  resetDetect,
  saves,
  select,
} from "./detectKit";

beforeEach(resetDetect);
afterEach(cleanupDetect);

describe("the matrix", () => {
  it("shows what was found in each image and why the last run left one out", async () => {
    mount();
    const grid = await matrix();
    expect(
      within(grid).getByRole("gridcell", { name: /view 0001, camera cam0, 20:/ }),
    ).toBeTruthy();
    expect(
      within(grid).getByRole("gridcell", {
        name: /view 0002, camera cam0, 14: 14 of 20 features\. Last run: Only part of the chessboard was found/,
      }),
    ).toBeTruthy();
    expect(
      within(grid).getByRole("gridcell", {
        name: /view 0002, camera cam1, 0: No grid of corners found/,
      }),
    ).toBeTruthy();
    expect(
      within(grid).getByRole("gridcell", {
        name: /view 0003, camera cam0, ·: Not detected yet/,
      }),
    ).toBeTruthy();
    expect(
      within(grid).getByRole("gridcell", {
        name: /view 0003, camera cam1, –: This camera has no image/,
      }),
    ).toBeTruthy();
  });

  it("finds the views with a problem, and those left out", async () => {
    mount({
      view: {
        ...projectView(DIR),
        spec: {
          ...projectView(DIR).spec,
          exclude: { images: [{ camera: "cam0", image: "a/0001.png" }] },
        },
      },
    });
    const grid = await matrix();
    expect(within(grid).getAllByRole("rowheader")).toHaveLength(3);
    fireEvent.click(screen.getByRole("switch", { name: "Only problems" }));
    expect(within(await matrix()).getAllByRole("rowheader")).toHaveLength(1);
    fireEvent.click(screen.getByRole("switch", { name: "Only problems" }));
    fireEvent.click(screen.getByRole("switch", { name: "Only excluded" }));
    const rows = within(await matrix()).getAllByRole("rowheader");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.textContent).toBe("0001");
    expect(
      within(await matrix()).getByRole("gridcell", {
        name: /view 0001, camera cam0, 20:.*Excluded/,
      }),
    ).toBeTruthy();
  });

  it("says nothing has been detected yet and offers to run it", async () => {
    const never = overview();
    for (const v of never.views)
      v.images = v.images.map((i) =>
        i ? { image: i.image, state: { kind: "not_detected" } } : null,
      );
    mount({ overview: never });
    expect(
      await screen.findByText(
        /Nothing has been detected with these detector settings yet/,
      ),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Detect all" })).toBeTruthy();
  });

  it("says the dataset has no views", async () => {
    mount({ overview: { cameras: ["cam0"], expected: 20, views: [] } });
    expect(await screen.findByText(/The dataset has no views/)).toBeTruthy();
  });
});

describe("Detect all", () => {
  it("is a job with progress, and Cancel stops it", async () => {
    mount({
      detect: async (control) => {
        control.emit({ camera: 0, image: "a/0001.png", done: 1, total: 5 });
        await control.cancelled;
      },
    });
    await matrix();
    fireEvent.click(screen.getByRole("button", { name: "Detect all" }));
    expect(await screen.findByText("Detecting 1/5")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "Detection progress" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByText("Detecting 1/5")).toBeNull());
    // What it detected is read again.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Detect all" })).toBeTruthy(),
    );
  });

  it("is refused in read-only mode", async () => {
    mount();
    await matrix();
    act(() => useJobs.setState({ connection: "lost" }));
    expect(
      screen.getByRole("button", { name: "Detect all" }).hasAttribute("disabled"),
    ).toBe(true);
  });
});

describe("the selected image", () => {
  it("says to select one, then lists its features", async () => {
    mount();
    expect(await screen.findByText(/Select an image in the matrix/)).toBeTruthy();
    await select(/view 0001, camera cam0/);
    expect(screen.getByRole("heading", { name: /View 0001/ })).toBeTruthy();
    expect(screen.getByText("Board found", { selector: "p" })).toBeTruthy();
  });

  it("detects an image that has no cached detection only when asked", async () => {
    const mock = mount();
    fireEvent.click(
      within(await matrix()).getByRole("gridcell", { name: /view 0003, camera cam0/ }),
    );
    expect(await screen.findByText(/No detection to show/)).toBeTruthy();
    expect(mock.calls.some((c) => c.command === "detection_detail")).toBe(false);
  });

  it("lists what was rejected in the Rejected tab", async () => {
    mount();
    await select(/view 0001, camera cam0/);
    fireEvent.click(screen.getByRole("tab", { name: /Rejected/ }));
    expect(
      await screen.findByText("Corner candidate not taken by any labelled grid."),
    ).toBeTruthy();
  });
});

describe("trying detector settings", () => {
  async function editMinFeatures(value: string) {
    fireEvent.click(await screen.findByRole("tab", { name: "Detector" }));
    const field = await screen.findByRole("spinbutton", {
      name: /Min features per view/,
    });
    fireEvent.change(field, { target: { value } });
    fireEvent.blur(field);
  }

  it("previews on this image, says how it differs, and writes nothing", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    await editMinFeatures("12");
    fireEvent.click(await screen.findByRole("button", { name: "Preview on this image" }));
    expect(await screen.findByText("Preview (unsaved settings)")).toBeTruthy();
    expect(
      await screen.findByText(/\+1 feature against the saved settings/),
    ).toBeTruthy();
    const call = mock.calls.find((c) => c.command === "detect_preview");
    expect(call?.args).toMatchObject({
      dir: DIR,
      camera: 0,
      image: "a/0001.png",
      detector: { min_features_per_view: 12 },
    });
    expect(saves(mock)).toHaveLength(0);
    // The saved detection can be shown again.
    fireEvent.click(screen.getByRole("radio", { name: "Saved" }));
  });

  it("applies the settings into the project's manifest, then offers to detect all again", async () => {
    const mock = mount();
    await select(/view 0001, camera cam0/);
    await editMinFeatures("12");
    const apply = screen.getByRole("button", { name: "Apply to project" });
    await waitFor(() => expect((apply as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(apply);
    await screen.findByText("Detector settings saved");
    expect(saves(mock)).toHaveLength(1);
    expect(saves(mock)[0]!.revision).toBe("sha256:r1");
    expect(saves(mock)[0]!.spec.dataset.manifest.detector).toEqual({
      min_features_per_view: 12,
    });
    // The next run of every image reads the cache again.
    fireEvent.click(
      screen.getByRole("button", { name: "Detect all with the new settings" }),
    );
    await waitFor(() =>
      expect(mock.calls.some((c) => c.command === "detect_all")).toBe(true),
    );
  });

  it("will not apply while the manifest is a file of its own, but still previews", async () => {
    mount({
      view: {
        ...projectView(DIR),
        dataset_editable: false,
        spec: { version: 2, dataset: { manifest: "dataset.toml" } },
      },
    });
    await select(/view 0001, camera cam0/);
    fireEvent.click(await screen.findByRole("tab", { name: "Detector" }));
    expect(
      await screen.findByText("Read-only: the manifest is a file of its own"),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Apply to project" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(
      screen
        .getByRole("button", { name: "Preview on this image" })
        .hasAttribute("disabled"),
    ).toBe(false);
  });

  it("discards the edit and the preview", async () => {
    mount();
    await select(/view 0001, camera cam0/);
    await editMinFeatures("12");
    fireEvent.click(await screen.findByRole("button", { name: "Preview on this image" }));
    await screen.findByText("Preview (unsaved settings)");
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    await waitFor(() =>
      expect(screen.queryByText("Preview (unsaved settings)")).toBeNull(),
    );
    expect(
      screen.getByRole("button", { name: "Apply to project" }).hasAttribute("disabled"),
    ).toBe(true);
  });
});

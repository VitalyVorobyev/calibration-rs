// @vitest-environment jsdom
/** The map tabs of Inspect over a mock back end: the distortion, projection σ
 * and residual maps with the records and statistics under the last, what they
 * state and select, and the words for a run with no σ or no residual records. */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StudioError } from "../../api";
import { backend, mount, show, installInspectHooks } from "../../test/inspectKit";
import { useSelection } from "../../session/selection";
import { distortionReply } from "../../test/planeFixtures";
import { DistortionTab } from "./maps/DistortionTab";
import { SigmaTab } from "./maps/SigmaTab";
import { ResidualsTab } from "./ResidualsTab";

installInspectHooks();

describe("Distortion", () => {
  it("states what the lens does and offers a tilt layer only when there is one", async () => {
    backend();
    show();
    mount(<DistortionTab />);
    expect(
      await screen.findByText(/The lens displaces a pixel by up to 3\.61 px/),
    ).toBeTruthy();
    expect(screen.queryByRole("radio", { name: "Tilt" })).toBeNull();
    expect(screen.getByRole("radio", { name: "Lens" })).toBeTruthy();
    expect(screen.getByText(/64 × 48|4 × 3 nodes of 10 px/)).toBeTruthy();
  });

  it("offers the tilt layer for a tilted sensor", async () => {
    backend({ distortion_map: () => distortionReply(true) });
    show();
    mount(<DistortionTab />);
    expect(await screen.findByRole("radio", { name: "Tilt" })).toBeTruthy();
  });

  it("says when the map cannot be made", async () => {
    backend({
      distortion_map: () => {
        throw new StudioError({ kind: "invalid", message: "no camera" });
      },
    });
    show();
    mount(<DistortionTab />);
    expect(await screen.findByText(/Could not make this map: no camera/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
    // The tab's controls stay, so the person can still switch.
    expect(screen.getByRole("radio", { name: "Lens" })).toBeTruthy();
  });
});

describe("Projection σ", () => {
  it("says where a ray lands is known to", async () => {
    backend();
    show();
    mount(<SigmaTab />);
    expect(await screen.findByText(/is known to/)).toBeTruthy();
  });

  it("gives the reason there is no σ map", async () => {
    backend({
      projection_sigma_map: () => ({
        status: "unavailable",
        camera: 0,
        reason: { kind: "fixed" },
      }),
    });
    show();
    mount(<SigmaTab />);
    expect(
      await screen.findByText("No projection uncertainty for this camera"),
    ).toBeTruthy();
    expect(screen.getByText("held fixed, not estimated")).toBeTruthy();
  });
});

describe("Residuals", () => {
  it("shows the map, the records worst first and the statistics", async () => {
    backend();
    show();
    mount(<ResidualsTab />);
    expect(await screen.findByText(/The systematic error peaks at/)).toBeTruthy();
    const records = await screen.findByRole("table", { name: "Residual records" });
    const rows = within(records).getAllByRole("row").slice(1);
    // |r|: 1 (view 1), 0.5, 0.1.
    expect(rows[0]!.textContent).toContain("1.000");
    expect(rows[2]!.textContent).toContain("0.100");
    expect(
      await screen.findByRole("table", { name: "Reprojection statistics" }),
    ).toBeTruthy();
  });

  it("selects a record's view and feature, which the frame viewer follows", async () => {
    backend();
    show();
    mount(<ResidualsTab />);
    const records = await screen.findByRole("table", { name: "Residual records" });
    fireEvent.click(within(records).getAllByRole("row")[2]!);
    expect(useSelection.getState().selection).toEqual({ camera: 0, view: 0, feature: 5 });
    fireEvent.click(within(records).getAllByRole("row")[1]!);
    expect(useSelection.getState().selection).toEqual({ camera: 0, view: 1, feature: 6 });
  });

  it("selects a view from the statistics", async () => {
    backend();
    show();
    mount(<ResidualsTab />);
    const stats = await screen.findByRole("table", { name: "Reprojection statistics" });
    fireEvent.click(within(stats).getByText("view 1"));
    expect(useSelection.getState().selection.view).toBe(1);
  });

  it("asks the map for the chosen view only", async () => {
    const mock = backend();
    show();
    useSelection.getState().selectView(1);
    mount(<ResidualsTab />);
    await screen.findByText(/The systematic error peaks at/);
    fireEvent.click(screen.getByRole("combobox", { name: "Views" }));
    fireEvent.click(screen.getByRole("option", { name: /View 1/ }));
    await waitFor(() =>
      expect(
        mock.calls.some(
          (c) =>
            c.command === "residual_map" &&
            JSON.stringify((c.args as { options: unknown }).options) === '{"views":[1]}',
        ),
      ).toBe(true),
    );
  });

  it("opens for a run without residual records", async () => {
    backend({
      residual_map: () => {
        throw new StudioError({ kind: "invalid", message: "no per-feature residuals" });
      },
      residual_records: () => ({
        index: [],
        view: [],
        feature: [],
        target_xyz_m: [],
        observed_px: [],
        projected_px: [],
        residual_px: [],
      }),
      residual_stats: () => ({ cameras: [], cells: [], views: [] }),
    });
    show();
    mount(<ResidualsTab />);
    expect(
      await screen.findByText(
        /This run has no per-feature residuals for camera 0, so there is no residual map/,
      ),
    ).toBeTruthy();
    expect(
      await screen.findByText(/This run holds no per-feature residuals for camera 0/),
    ).toBeTruthy();
    expect(
      await screen.findByText(/The run has no per-feature residuals to summarise/),
    ).toBeTruthy();
  });
});

// @vitest-environment jsdom
/** The Compare screen's fit tables and its projection-difference map over a
 * mock back end: each run's error side by side, the summary of every pair,
 * the map of the picked pair, and the words for a pair with nothing to draw
 * or no map. */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MAP, IDS, installCompareHooks, mount, tick } from "./__fixtures__/compareKit";
import { comparison } from "./__fixtures__/compareFixtures";

installCompareHooks();

describe("fit quality", () => {
  it("sets each run's RMS side by side, and says when the residuals do not compare", async () => {
    mount();
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Fit quality" }));
    expect(
      await screen.findByRole("table", { name: "Fit quality of cam0" }),
    ).toBeTruthy();
    expect(screen.getByRole("table", { name: "Fit quality of cam1" })).toBeTruthy();
    expect(
      screen.getByText(
        /Residuals compare like with like only between runs of the same observations/,
      ),
    ).toBeTruthy();
    const runsTable = screen.getByRole("table", { name: /quality and uncertainty/ });
    expect(within(runsTable).getAllByText(/^available/)[0]).toBeTruthy();
  });

  it("gives the reason a run has no covariance", async () => {
    const c = comparison();
    c.runs[1]!.covariance = { status: "unavailable", reason: { kind: "no_redundancy" } };
    mount({ compare_runs: () => c });
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Fit quality" }));
    expect(
      await screen.findByText(/no redundancy: as many measurements as unknowns/),
    ).toBeTruthy();
  });
});

describe("projection difference", () => {
  it("lists every run and camera, and draws the picked pair's map with what it says", async () => {
    mount();
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Projection difference" }));
    const table = await screen.findByRole("table", {
      name: /Projection difference of each run and camera/,
    });
    expect(within(table).getAllByRole("row").length).toBeGreaterThan(3);
    expect(
      await screen.findByText(
        /run-002 projects within 0\.120 px of run-001 over the covered region/,
      ),
    ).toBeTruthy();
    expect(
      screen.getByText(/3 of 12 nodes lie outside what the features of either run cover/),
    ).toBeTruthy();
    expect(screen.getByRole("radio", { name: "Rotation" })).toBeTruthy();
    // The fitted channel's switch uses the map's own words.
    expect(screen.getByRole("switch", { name: "Fade extrapolated" })).toBeTruthy();
  });

  it("says there is nothing to draw when the projections agree to rounding error", async () => {
    mount({
      projection_difference_map: () => ({
        ...MAP,
        max_valid_px: 3e-14,
        max_fit_px: 2e-14,
      }),
    });
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Projection difference" }));
    expect(
      await screen.findByText("Nothing to draw: the projections are the same"),
    ).toBeTruthy();
    expect(screen.queryByRole("application")).toBeNull();
  });

  it("says why a pair has no projection difference", async () => {
    mount({
      projection_difference_map: () => ({
        status: "unavailable",
        camera: "cam0",
        reason: "not_central",
      }),
    });
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Projection difference" }));
    expect(
      await screen.findByText(
        /No projection difference for cam0 between run-001 and run-002/,
      ),
    ).toBeTruthy();
    expect(screen.getByText(/not a central projection/)).toBeTruthy();
    // And in the table, the pair whose summary says so.
    const table = screen.getByRole("table", {
      name: /Projection difference of each run/,
    });
    expect(within(table).getByText(/too few image positions in common/)).toBeTruthy();
  });

  it("asks for the other alignment when it is chosen", async () => {
    const { mock } = mount();
    tick(IDS);
    fireEvent.click(await screen.findByRole("tab", { name: "Projection difference" }));
    await screen.findByText(/run-002 projects within/);
    fireEvent.click(screen.getByRole("radio", { name: "None" }));
    await waitFor(() =>
      expect(
        mock.calls.some(
          (c) =>
            c.command === "projection_difference_map" &&
            (c.args as { compensation: string }).compensation === "none",
        ),
      ).toBe(true),
    );
  });
});

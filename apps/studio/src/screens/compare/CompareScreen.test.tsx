// @vitest-environment jsdom
/** The Compare screen over a mock back end: the set of runs with its
 * reference, how the runs relate before any number, the parameter tables
 * with their tests and the reason of each test that is not made, keeping a
 * run (star and label), and the states of a comparison. The fit tables and
 * the projection-difference map are in `CompareViews.test.tsx`. */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StudioError } from "../../api";
import type { RunComparison } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { useCompareSet } from "../../session/compareSet";
import { projectView } from "../../test/projectFixtures";
import { DIR, comparison } from "./__fixtures__/compareFixtures";
import { IDS, installCompareHooks, mount, runs, tick } from "./__fixtures__/compareKit";

installCompareHooks();

describe("the set", () => {
  it("asks for runs when none is chosen, and for a second when one is", async () => {
    mount();
    expect(await screen.findByText(/Tick two or more runs in the run list/)).toBeTruthy();
    tick(["run-002"]);
    expect(await screen.findByText(/One run is chosen/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Star run-002" })).toBeTruthy();
  });

  it("opens on the runs the address names, in its order", async () => {
    const { router } = mount({}, `/compare?project=${DIR}&runs=run-002,run-001`);
    const set = await screen.findByRole("list", { name: "Runs compared" });
    expect(within(set).getAllByRole("listitem")[0]!.textContent).toContain("run-002");
    expect(within(set).getByText("Reference")).toBeTruthy();
    expect(useCompareSet.getState().ids).toEqual(["run-002", "run-001"]);
    expect(new URLSearchParams(router.state.location.search).get("runs")).toBe(
      "run-002,run-001",
    );
  });

  it("compares against the first run: making another the reference reorders the set", async () => {
    const { mock, router } = mount();
    tick(IDS);
    await screen.findByRole("list", { name: "Runs compared" });
    const tiles = screen.getAllByRole("listitem");
    const second = tiles.find(
      (t) =>
        t.textContent?.includes("run-002") &&
        within(t).queryByRole("button", { name: "Make reference" }),
    )!;
    fireEvent.click(within(second).getByRole("button", { name: "Make reference" }));
    await waitFor(() =>
      expect(useCompareSet.getState().ids).toEqual(["run-002", "run-001", "run-003"]),
    );
    await waitFor(() =>
      expect(
        mock.calls.some(
          (c) =>
            c.command === "compare_runs" &&
            (c.args as { runs: string[] }).runs[0] === "run-002",
        ),
      ).toBe(true),
    );
    expect(new URLSearchParams(router.state.location.search).get("runs")).toBe(
      "run-002,run-001,run-003",
    );
  });

  it("moves a run later and removes one", async () => {
    mount();
    tick(IDS);
    await screen.findByText(/How the runs relate to run-001/);
    fireEvent.click(screen.getByRole("button", { name: "Move run-001 later" }));
    expect(useCompareSet.getState().ids).toEqual(["run-002", "run-001", "run-003"]);
    fireEvent.click(
      screen.getByRole("button", { name: "Remove run-003 from the comparison" }),
    );
    expect(useCompareSet.getState().ids).toEqual(["run-002", "run-001"]);
  });
});

describe("how the runs relate", () => {
  it("says it in words before any number, with what changed between the runs", async () => {
    mount();
    tick(IDS);
    const panel = await screen.findByRole("region", { name: "How the runs relate" });
    expect(within(panel).getByText("Independent captures")).toBeTruthy();
    expect(within(panel).getByText("Same data")).toBeTruthy();
    expect(within(panel).getByText(/share no observation/)).toBeTruthy();
    expect(within(panel).getByText(/8 images and 3 200 features in common/)).toBeTruthy();
    expect(within(panel).getByText(/camera model differs for cam1/)).toBeTruthy();
    expect(within(panel).getByText("Settings that differ")).toBeTruthy();
    expect(within(panel).getByText("solver.max_iters")).toBeTruthy();
  });

  it("says a refused relation and what was done instead", async () => {
    const c: RunComparison = comparison();
    c.differences[0]!.relation_refused = {
      stated: { kind: "independent" },
      reason: "shared_features",
    };
    c.differences[0]!.relation = { kind: "overlapping" };
    mount({ compare_runs: () => c });
    tick(IDS);
    expect(await screen.findByText("Stated relation refused")).toBeTruthy();
    expect(
      screen.getByText(/the runs share features\. The pair is compared as overlapping/),
    ).toBeTruthy();
  });

  it("states what the data supports, and that nothing ranks the runs", async () => {
    mount();
    tick(IDS);
    const verdict = await screen.findByRole("region", { name: "What the data supports" });
    expect(within(verdict).getByText(/camera models differ/)).toBeTruthy();
    expect(within(verdict).getByText(/Nothing here ranks the runs/)).toBeTruthy();
  });
});

describe("parameters", () => {
  it("shows a test's outcome, and for a test not made the reason in words", async () => {
    mount();
    tick(IDS);
    const cam0 = await screen.findByRole("table", {
      name: "Parameters of cam0 across the runs",
    });
    const fy = within(cam0).getByRole("row", { name: /^fy/ });
    expect(within(fy).getByText("significant")).toBeTruthy();
    expect(within(fy).getByText(/z \+2\.83 · p 0\.005/)).toBeTruthy();
    const fx = within(cam0).getByRole("row", { name: /^fx/ });
    expect(within(fx).getByText("consistent")).toBeTruthy();
    // run-003 shares the data: no test, its reason, never a z.
    expect(within(fx).getByText("not tested")).toBeTruthy();
    expect(within(fx).getByText("shared data")).toBeTruthy();
    const k3 = within(cam0).getByRole("row", { name: /^k3/ });
    expect(within(k3).getAllByText("not tested").length).toBeGreaterThan(0);
    expect(within(k3).getByText("not estimated")).toBeTruthy();
  });

  it("reads the joint test of a camera first, and says a missing one", async () => {
    mount();
    tick(IDS);
    const cam0 = await screen.findByRole("region", { name: "Parameters of cam0" });
    expect(
      within(cam0).getByText(
        /χ² 9\.1 on 8 dof, p 0\.330: the runs agree at the 5 % level/,
      ),
    ).toBeTruthy();
    expect(
      within(cam0).getByText(/no joint test \(the runs share observations/),
    ).toBeTruthy();
    const cam1 = screen.getByRole("region", { name: "Parameters of cam1" });
    expect(within(cam1).getByText(/the runs disagree at the 5 % level/)).toBeTruthy();
    expect(
      within(cam1).getByText(
        /differs from the reference's, so its parameters are not compared/,
      ),
    ).toBeTruthy();
  });

  it("keeps only the differences that matter on request", async () => {
    mount();
    tick(IDS);
    const cam0 = await screen.findByRole("table", {
      name: "Parameters of cam0 across the runs",
    });
    expect(within(cam0).getAllByRole("row")).toHaveLength(2 + 3);
    fireEvent.click(screen.getByRole("switch", { name: "Only differences that matter" }));
    await waitFor(() => {
      const t = screen.getByRole("table", { name: "Parameters of cam0 across the runs" });
      // Two header rows and the one row whose test rejects.
      expect(within(t).getAllByRole("row")).toHaveLength(3);
      expect(within(t).getByText("fy")).toBeTruthy();
    });
    expect(screen.getByText(/1 of 3 parameters/)).toBeTruthy();
  });

  it("shows the rig poses with their test", async () => {
    mount();
    tick(IDS);
    const rig = await screen.findByRole("table", { name: "Rig poses" });
    expect(within(rig).getByText("cam1")).toBeTruthy();
    expect(within(rig).getByText(/χ² 3\.2 · 6 dof · p 0\.780/)).toBeTruthy();
    expect(within(rig).getByText("0.4 mm")).toBeTruthy();
  });
});

describe("keeping a run", () => {
  it("stars this run and unstars the others of the set", async () => {
    const { mock } = mount({}, `/compare?project=${DIR}`, ["run-001"]);
    tick(["run-001", "run-002"]);
    await screen.findByText(/How the runs relate to run-001/);
    const tile = screen
      .getAllByRole("listitem")
      .find((t) => within(t).queryByRole("button", { name: "Star run-002" }))!;
    fireEvent.click(within(tile).getByRole("button", { name: "Keep" }));
    await waitFor(() => {
      const notes = mock.calls.filter((c) => c.command === "set_run_notes");
      expect(
        notes.map((c) => [
          (c.args as { id: string }).id,
          (c.args as { notes: { starred: boolean } }).notes.starred,
        ]),
      ).toEqual([
        ["run-002", true],
        ["run-001", false],
      ]);
    });
  });

  it("saves a label typed in the tile", async () => {
    const { mock } = mount();
    tick(["run-001", "run-002"]);
    const label = await screen.findByRole("textbox", { name: "Label of run-002" });
    fireEvent.change(label, { target: { value: "after recalibration" } });
    fireEvent.blur(label);
    await waitFor(() =>
      expect(
        mock.calls.some(
          (c) =>
            c.command === "set_run_notes" &&
            (c.args as { notes: { label?: string } }).notes.label ===
              "after recalibration",
        ),
      ).toBe(true),
    );
  });
});

describe("states", () => {
  it("says why the runs could not be compared, and tries again", async () => {
    let fail = true;
    const { mock } = mount({
      compare_runs: () => {
        if (fail)
          throw new StudioError({ kind: "invalid", message: "the runs share no camera" });
        return comparison();
      },
    });
    tick(IDS);
    expect(await screen.findByText("The runs could not be compared")).toBeTruthy();
    expect(screen.getByText("the runs share no camera")).toBeTruthy();
    fail = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/How the runs relate to run-001/)).toBeTruthy();
    expect(mock.calls.filter((c) => c.command === "compare_runs").length).toBeGreaterThan(
      1,
    );
  });

  it("shows a busy status while the comparison is read", async () => {
    mount({ compare_runs: () => new Promise<RunComparison>(() => undefined) });
    tick(IDS);
    expect(
      await screen.findByRole("status", { name: "Comparing the runs" }),
    ).toBeTruthy();
  });

  it("names the quality thresholds of the project in the request", async () => {
    const { mock } = mount();
    act(() =>
      useProject.setState({
        project: {
          ...projectView(DIR, runs()),
          spec: { ...projectView(DIR).spec, quality: { coverage_warn_below: 0.4 } },
        },
      }),
    );
    tick(IDS);
    await screen.findByText(/How the runs relate to run-001/);
    const call = mock.calls.find((c) => c.command === "compare_runs");
    expect((call!.args as { options: unknown }).options).toEqual({
      thresholds: { coverage_warn_below: 0.4 },
    });
  });
});

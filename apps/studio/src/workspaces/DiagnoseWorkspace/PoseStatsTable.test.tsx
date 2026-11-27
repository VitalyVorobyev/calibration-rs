// @vitest-environment jsdom
/** Component test for the multi-pose residual stats table: renders the
 * back end's per-pose statistics (`residual_stats`' `views`), sorts on
 * header click, and jumps to a pose on row click. */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ViewStats } from "../../api/generated/types";
import { PoseStatsTable } from "./PoseStatsTable";

function stat(pose: number, mean: number): ViewStats {
  return {
    view: pose,
    stats: { count: 1, mean, rms: mean, max: mean },
    unprojected: 0,
  };
}

// pose 0 → mean 1, pose 1 → mean 5, pose 2 → mean 3.
const RESIDUALS = [stat(0, 1), stat(1, 5), stat(2, 3)];

/** Pose values of the body rows in render order. */
function bodyPoseOrder(): string[] {
  const rows = screen.getAllByRole("row").slice(1); // drop header
  return rows.map(
    (row) => within(row).getAllByRole("cell")[0]!.textContent?.trim() ?? "",
  );
}

/** The body row whose pose cell reads `pose`. */
function rowForPose(pose: string): HTMLElement {
  const row = screen
    .getAllByRole("row")
    .slice(1)
    .find((r) => within(r).getAllByRole("cell")[0]!.textContent?.trim() === pose);
  if (!row) throw new Error(`no row for pose ${pose}`);
  return row;
}

afterEach(cleanup);

describe("PoseStatsTable", () => {
  it("renders one row per pose, worst mean first by default", () => {
    render(<PoseStatsTable views={RESIDUALS} selectedPose={0} onSelectPose={() => {}} />);
    // Default sort is mean descending → pose 1 (5) then 2 (3) then 0 (1).
    expect(bodyPoseOrder()).toEqual(["1", "2", "0"]);
  });

  it("re-sorts by pose ascending when the pose header is clicked", () => {
    render(<PoseStatsTable views={RESIDUALS} selectedPose={0} onSelectPose={() => {}} />);
    const header = screen.getByRole("columnheader", { name: /pose/i });
    fireEvent.click(within(header).getByRole("button"));
    expect(bodyPoseOrder()).toEqual(["0", "1", "2"]);
  });

  it("jumps to the clicked pose", () => {
    const onSelectPose = vi.fn();
    render(
      <PoseStatsTable views={RESIDUALS} selectedPose={0} onSelectPose={onSelectPose} />,
    );
    fireEvent.click(rowForPose("2"));
    expect(onSelectPose).toHaveBeenCalledWith(2);
  });

  it("marks the selected pose's row as current", () => {
    render(<PoseStatsTable views={RESIDUALS} selectedPose={2} onSelectPose={() => {}} />);
    expect(rowForPose("2").getAttribute("aria-current")).toBe("true");
    expect(rowForPose("1").getAttribute("aria-current")).toBeNull();
  });

  it("renders a placeholder when there are no residuals", () => {
    render(<PoseStatsTable views={[]} selectedPose={0} onSelectPose={() => {}} />);
    expect(screen.getByText(/No per-feature residuals/)).toBeTruthy();
  });
});

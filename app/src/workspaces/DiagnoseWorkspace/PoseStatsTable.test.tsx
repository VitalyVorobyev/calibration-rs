// @vitest-environment jsdom
/** Component test for the multi-pose residual stats table:
 * renders per-pose rows from mocked residuals, sorts on header click, and
 * jumps to a pose on row click. The aggregation itself is covered by
 * `lib/residualStats.test.ts`; this asserts the wiring. */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TargetFeatureResidual } from "../../types";
import { PoseStatsTable } from "./PoseStatsTable";

function res(pose: number, error_px: number): TargetFeatureResidual {
  return {
    pose,
    camera: 0,
    feature: 0,
    target_xyz_m: [0, 0, 0],
    observed_px: [0, 0],
    projected_px: [0, 0],
    error_px,
  };
}

// pose 0 → mean 1, pose 1 → mean 5, pose 2 → mean 3.
const RESIDUALS = [res(0, 1), res(1, 5), res(2, 3)];

/** Pose values of the body rows in render order. */
function bodyPoseOrder(): string[] {
  const rows = screen.getAllByRole("row").slice(1); // drop header
  return rows.map((row) => within(row).getAllByRole("cell")[0].textContent?.trim() ?? "");
}

afterEach(cleanup);

describe("PoseStatsTable", () => {
  it("renders one row per pose, worst mean first by default", () => {
    render(
      <PoseStatsTable residuals={RESIDUALS} selectedPose={0} onSelectPose={() => {}} />,
    );
    // Default sort is mean descending → pose 1 (5) then 2 (3) then 0 (1).
    expect(bodyPoseOrder()).toEqual(["1", "2", "0"]);
  });

  it("re-sorts by pose ascending when the pose header is clicked", () => {
    render(
      <PoseStatsTable residuals={RESIDUALS} selectedPose={0} onSelectPose={() => {}} />,
    );
    fireEvent.click(screen.getByRole("columnheader", { name: /pose/i }));
    expect(bodyPoseOrder()).toEqual(["0", "1", "2"]);
  });

  it("jumps to the clicked pose", () => {
    const onSelectPose = vi.fn();
    render(
      <PoseStatsTable
        residuals={RESIDUALS}
        selectedPose={0}
        onSelectPose={onSelectPose}
      />,
    );
    fireEvent.click(screen.getByTitle("Jump to pose 2"));
    expect(onSelectPose).toHaveBeenCalledWith(2);
  });

  it("renders a placeholder when there are no residuals", () => {
    render(<PoseStatsTable residuals={[]} selectedPose={0} onSelectPose={() => {}} />);
    expect(screen.getByText(/No per-feature residuals/)).toBeTruthy();
  });
});

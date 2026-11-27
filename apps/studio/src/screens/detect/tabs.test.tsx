// @vitest-environment jsdom
/** The Features and Rejected tabs on the detail of each target kind: what
 * the table and the strip say, the warnings a detection earns, and the
 * rejected candidates in words. */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  charucoDetail,
  chessboardDetail,
  puzzleboardDetail,
  ringgridDetail,
} from "./__fixtures__/details";
import { FeaturesTab } from "./FeaturesTab";
import { RejectedTab } from "./RejectedTab";
import { rejectedItems } from "./overlayModel";
import type { DetectionDetail } from "../../api/generated/types";

afterEach(cleanup);

function features(
  detail: DetectionDetail,
  over: Partial<React.ComponentProps<typeof FeaturesTab>> = {},
) {
  const props = {
    detail,
    expected: 4,
    excluded: new Set<number>(),
    picked: new Set<number>(),
    hovered: null,
    onPick: vi.fn(),
    onHover: vi.fn(),
    onExclude: vi.fn(),
    readOnly: false,
    unmatched: 0,
    ...over,
  };
  render(
    <TooltipProvider>
      <FeaturesTab {...props} />
    </TooltipProvider>,
  );
  return props;
}

describe("the Features tab", () => {
  it("reads a chessboard: the strip of facts and a row per corner", () => {
    features(chessboardDetail());
    const strip = screen.getByRole("list");
    expect(strip.textContent).toContain("4 of 4");
    expect(strip.textContent).toContain("rejected1");
    expect(strip.textContent).toContain("cell20.0 px");
    expect(strip.textContent).toContain("noise σ1.50");
    expect(strip.textContent).toContain("leave-one-out0.040 px");
    expect(screen.getByText("Board found")).toBeTruthy();
    const table = screen.getByRole("table", { name: "Features of this image" });
    expect(within(table).getAllByRole("row")).toHaveLength(5);
    expect(within(table).getByText("1,0")).toBeTruthy();
    expect(within(table).getAllByText("accepted")).toHaveLength(4);
  });

  it("marks the excluded features and presses a row to pick it", () => {
    const props = features(chessboardDetail(), { excluded: new Set([2]) });
    expect(screen.getAllByText("excluded")).toHaveLength(1);
    fireEvent.click(screen.getByText("1,1"));
    expect(props.onPick).toHaveBeenCalledWith(3, false);
    fireEvent.click(screen.getByRole("checkbox", { name: "Pick feature 1,0" }));
    expect(props.onPick).toHaveBeenCalledWith(1, true);
  });

  it("mirrors the pointer over a row to the overlay", () => {
    const props = features(chessboardDetail());
    fireEvent.pointerEnter(screen.getByText("0,1").closest("tr")!);
    expect(props.onHover).toHaveBeenCalledWith(2);
  });

  it("excludes the picked features, or restores them when they all were out", () => {
    const props = features(chessboardDetail(), { picked: new Set([0, 1]) });
    fireEvent.click(screen.getByRole("button", { name: "Exclude 2 features" }));
    expect(props.onExclude).toHaveBeenCalled();
    cleanup();
    features(chessboardDetail(), {
      picked: new Set([0]),
      excluded: new Set([0]),
    });
    expect(screen.getByRole("button", { name: "Restore 1 feature" })).toBeTruthy();
  });

  it("disables excluding in read-only mode", () => {
    features(chessboardDetail(), { picked: new Set([0]), readOnly: true });
    expect(
      screen.getByRole("button", { name: "Exclude 1 feature" }).hasAttribute("disabled"),
    ).toBe(true);
  });

  it("warns of excluded points that match no feature", () => {
    features(chessboardDetail(), { unmatched: 2 });
    expect(screen.getByText(/2 excluded points match no feature/)).toBeTruthy();
  });

  it("reads a ChArUco board with the corner ids", () => {
    features(charucoDetail(), { expected: 6 });
    expect(screen.getByRole("list").textContent).toContain("2 of 6");
    expect(screen.getAllByText("1,1")).toHaveLength(1);
  });

  it("reads a PuzzleBoard: master ids, the mismatch, and the decode summary", () => {
    features(puzzleboardDetail(), { expected: null });
    expect(screen.getByRole("columnheader", { name: /Id/ })).toBeTruthy();
    expect(screen.getByText("4021")).toBeTruthy();
    expect(screen.getAllByText("master mismatch")).toHaveLength(2);
    expect(screen.getByText(/bit error 10.0%/)).toBeTruthy();
    expect(screen.queryByText("The labelling is likely wrong")).toBeNull();
  });

  it("says a PuzzleBoard's labelling is likely wrong when its dots dissent", () => {
    features(puzzleboardDetail(0.4), { expected: null });
    expect(screen.getByText("The labelling is likely wrong")).toBeTruthy();
  });

  it("reads a ring grid: code, confidence, centre estimator, and a view not anchored", () => {
    features(ringgridDetail(false), { expected: 2 });
    expect(screen.getByRole("columnheader", { name: /Code/ })).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: /Conf\./ })).toBeTruthy();
    expect(screen.getByText("photometric")).toBeTruthy();
    expect(screen.getByText("outer ellipse")).toBeTruthy();
    expect(screen.getByText("0.93")).toBeTruthy();
    expect(screen.getByText("Labels are relative")).toBeTruthy();
  });

  it("does not warn of an anchored ring grid", () => {
    features(ringgridDetail(true), { expected: 2 });
    expect(screen.queryByText("Labels are relative")).toBeNull();
  });

  it("says what the detector made of an image it doubts", () => {
    const detail = chessboardDetail();
    detail.outcome = { status: "ambiguous", repeated_pixels: 3, repeated_points: 0 };
    features(detail);
    expect(screen.getByText(/Ambiguous: 3 features repeat a pixel/)).toBeTruthy();
  });
});

describe("the Rejected tab", () => {
  const show = (detail: DetectionDetail, leftOut?: "too_few_features") => {
    const onFocus = vi.fn();
    render(
      <RejectedTab
        detail={detail}
        items={rejectedItems(detail)}
        leftOut={leftOut}
        onFocus={onFocus}
      />,
    );
    return onFocus;
  };

  it("groups the candidates by reason and says it in words", () => {
    show(ringgridDetail());
    expect(screen.getByRole("region", { name: "Not anchored" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Outside the board" })).toBeTruthy();
    expect(
      screen.getByText("Ring has no board position (view not anchored)."),
    ).toBeTruthy();
    expect(screen.getByText("conf. 0.60")).toBeTruthy();
  });

  it("shows a candidate on the image when it is pressed", () => {
    const onFocus = show(chessboardDetail());
    fireEvent.click(screen.getByRole("button", { name: /300.0, 40.0/ }));
    expect(onFocus).toHaveBeenCalledWith(expect.objectContaining({ id: "r0", x: 300 }));
    expect(screen.getByText("response 0.31")).toBeTruthy();
  });

  it("lists the markers a ChArUco board decoded as another", () => {
    show(charucoDetail());
    expect(screen.getByRole("region", { name: "Marker misread" })).toBeTruthy();
    expect(screen.getByText("Marker decodes as 17, board expects 23")).toBeTruthy();
  });

  it("puts the detector's verdict and the run's above the candidates", () => {
    const detail = chessboardDetail();
    detail.outcome = { status: "not_found", reason: "No grid of corners found" };
    show(detail, "too_few_features");
    expect(screen.getByText("No grid of corners found")).toBeTruthy();
    expect(screen.getByText("Too few features to use")).toBeTruthy();
  });

  it("says when nothing was rejected", () => {
    const detail = charucoDetail();
    if (detail.target.kind === "charuco") detail.target.rejected_markers = [];
    show(detail);
    expect(screen.getByText(/Nothing was rejected in this image/)).toBeTruthy();
  });
});

import { describe, expect, it } from "vitest";
import type { ImageDetection } from "../../api/generated/types";
import { overview } from "./__fixtures__/details";
import {
  cellStatus,
  filterRows,
  leftOutSentence,
  neverDetected,
  NO_DETECT_FILTER,
  outcomeSentence,
  overviewStats,
  rejectionSentence,
} from "./detectModel";

const img = (state: ImageDetection["state"], left_out?: ImageDetection["left_out"]) =>
  ({ image: "a.png", state, ...(left_out ? { left_out } : {}) }) as ImageDetection;
const detected = (features: number) => img({ kind: "detected", features });

describe("cellStatus", () => {
  it("is fine when the whole board was found", () => {
    const s = cellStatus(detected(20), 20, false);
    expect(s).toMatchObject({ tone: "normal", count: 20, text: "20", fraction: 1 });
    expect(s.sentence).toBe("20 of 20 features");
  });

  it("warns on fewer than expected, with the share for the bar", () => {
    const s = cellStatus(detected(14), 20, false);
    expect(s.tone).toBe("warn");
    expect(s.fraction).toBeCloseTo(0.7);
  });

  it("has no expected count for a PuzzleBoard: any count is fine", () => {
    expect(cellStatus(detected(9), null, false)).toMatchObject({
      tone: "normal",
      fraction: null,
    });
  });

  it("is a defect when nothing was found or the detector says so", () => {
    expect(cellStatus(detected(0), 20, false).tone).toBe("defect");
    const none = cellStatus(
      img({
        kind: "detected",
        features: 3,
        outcome: { status: "not_found", reason: "No grid" },
      }),
      20,
      false,
    );
    expect(none).toMatchObject({ tone: "defect", sentence: "No grid" });
  });

  it("warns when the detector doubted the board", () => {
    const s = cellStatus(
      img({
        kind: "detected",
        features: 20,
        outcome: { status: "ambiguous", repeated_pixels: 2, repeated_points: 1 },
      }),
      20,
      false,
    );
    expect(s.tone).toBe("warn");
    expect(s.sentence).toContain("Ambiguous");
  });

  it("carries the last run's verdict on an image it left out", () => {
    const s = cellStatus(
      img({ kind: "detected", features: 20 }, "turned_chessboard"),
      20,
      false,
    );
    expect(s.tone).toBe("warn");
    expect(s.sentence).toContain("Last run: The chessboard is turned against the others");
  });

  it("is pending without a cached detection, a defect when it failed", () => {
    expect(cellStatus(img({ kind: "not_detected" }), 20, false)).toMatchObject({
      tone: "pending",
      text: "·",
    });
    const failed = cellStatus(
      img({ kind: "failed", error: { kind: "io", message: "unreadable image" } }),
      20,
      false,
    );
    expect(failed).toMatchObject({ tone: "defect", sentence: "unreadable image" });
  });

  it("says a camera has no image, and marks an excluded image whatever it found", () => {
    expect(cellStatus(null, 20, false).tone).toBe("missing");
    const out = cellStatus(detected(20), 20, true);
    expect(out.tone).toBe("excluded");
    expect(out.count).toBe(20);
    expect(out.sentence).toContain("Excluded");
  });
});

describe("sentences", () => {
  it("say the detector's outcome and what the run did", () => {
    expect(outcomeSentence({ status: "detected" })).toBe("Board found");
    expect(outcomeSentence({ status: "not_found", reason: "" })).toBe("No board found");
    expect(outcomeSentence({ status: "filtered" })).toMatch(/own checks/);
    expect(leftOutSentence("too_few_features")).toBe("Too few features to use");
    expect(leftOutSentence("excluded")).toBe("Excluded from the calibration");
  });

  it("say why a candidate was left out", () => {
    expect(rejectionSentence({ kind: "unlabelled", strength: 1 })).toBe(
      "Corner candidate not taken by any labelled grid",
    );
    expect(rejectionSentence({ kind: "outside_board" })).toBe(
      "Labelled outside the declared board",
    );
    expect(rejectionSentence({ kind: "unanchored", confidence: 0.5 })).toBe(
      "Ring has no board position (view not anchored)",
    );
    expect(rejectionSentence({ kind: "uncorrected_center", confidence: 0.5 })).toBe(
      "Photometric fit fell back to the ellipse centre",
    );
  });
});

describe("the overview", () => {
  const data = overview();
  const status = (view: (typeof data.views)[number], camera: number) =>
    cellStatus(view.images[camera] ?? null, data.expected ?? null, false);

  it("counts images by what the cache holds", () => {
    const stats = overviewStats(data);
    expect(stats).toEqual({ images: 5, analysed: 4, pending: 1, failed: 0, notFound: 1 });
    expect(neverDetected(stats)).toBe(false);
    expect(neverDetected({ ...stats, analysed: 0, pending: 5, notFound: 0 })).toBe(true);
  });

  it("finds the views with a problem", () => {
    const rows = filterRows(
      data.views,
      { ...NO_DETECT_FILTER, problemsOnly: true },
      status,
    );
    expect(rows.map((r) => r.view.token)).toEqual(["0002"]);
  });

  it("finds the views whose worst count is low, and by token", () => {
    expect(
      filterRows(data.views, { ...NO_DETECT_FILTER, countBelow: 15 }, status).map(
        (r) => r.index,
      ),
    ).toEqual([1]);
    expect(
      filterRows(data.views, { ...NO_DETECT_FILTER, token: "0003" }, status).map(
        (r) => r.view.token,
      ),
    ).toEqual(["0003"]);
  });

  it("finds the excluded ones", () => {
    const out = (v: (typeof data.views)[number], c: number) =>
      cellStatus(v.images[c] ?? null, 20, v.token === "0001" && c === 1);
    expect(
      filterRows(data.views, { ...NO_DETECT_FILTER, excludedOnly: true }, out).map(
        (r) => r.view.token,
      ),
    ).toEqual(["0001"]);
  });
});

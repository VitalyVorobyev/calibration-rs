/** What the Detect screen says about detections, as pure rules: a cell of
 * the overview as a status, the detector's outcome and a run's verdict on
 * an image as sentences, the reason a candidate was left out in words, and
 * the filters that find a bad image among hundreds. */
import type {
  DetectionOutcome,
  DetectionOverview,
  ImageDetection,
  PointRejection,
  RejectReason,
  ViewDetections,
} from "../../api/generated/types";
import { formatCount } from "../../lib/format";

/** How a cell looks: `normal` is fine, `warn` is fewer than expected or
 * something the detector or a run doubted, `defect` is not found or
 * failed, `pending` has no cached detection yet, `excluded` is left out of
 * the calibration, `missing` has no image. */
export type CellTone = "normal" | "warn" | "defect" | "pending" | "excluded" | "missing";

export interface CellStatus {
  tone: CellTone;
  /** Detected features; `null` without a detection. */
  count: number | null;
  /** The cell's text: `48`, `0`, `·`. */
  text: string;
  /** What a tooltip and a screen reader say: the outcome in words. */
  sentence: string;
  /** The share of the expected count found, in `[0, 1]`; `null` when not known. */
  fraction: number | null;
}

/** The detector's own outcome in a sentence. */
export function outcomeSentence(outcome: DetectionOutcome | null | undefined): string {
  if (!outcome) return "Detected";
  switch (outcome.status) {
    case "detected":
      return "Board found";
    case "not_found":
      return outcome.reason.trim() === "" ? "No board found" : outcome.reason;
    case "ambiguous":
      return `Ambiguous: ${formatCount(outcome.repeated_pixels)} features repeat a pixel and ${formatCount(outcome.repeated_points)} repeat a board point`;
    case "filtered":
      return "Left out by the detector's own checks";
  }
}

/** Why a run left an image out of its calibration, in a sentence. */
export function leftOutSentence(reason: RejectReason): string {
  switch (reason) {
    case "too_few_features":
      return "Too few features to use";
    case "partial_chessboard":
      return "Only part of the chessboard was found";
    case "turned_chessboard":
      return "The chessboard is turned against the others";
    case "ambiguous_chessboard_turn":
      return "The chessboard's turn is ambiguous";
    case "excluded":
      return "Excluded from the calibration";
  }
}

/** Why a point candidate was not kept, in a sentence. */
export function rejectionSentence(reason: PointRejection): string {
  switch (reason.kind) {
    case "unlabelled":
      return "Corner candidate not taken by any labelled grid";
    case "outside_board":
      return "Labelled outside the declared board";
    case "unanchored":
      return "Ring has no board position (view not anchored)";
    case "uncorrected_center":
      return "Photometric fit fell back to the ellipse centre";
  }
}

/** The groups of the Rejected tab: a title for each kind of rejection. */
export function rejectionTitle(kind: PointRejection["kind"]): string {
  switch (kind) {
    case "unlabelled":
      return "Not in the grid";
    case "outside_board":
      return "Outside the board";
    case "unanchored":
      return "Not anchored";
    case "uncorrected_center":
      return "Centre fell back";
  }
}

/** The status of one camera's image of a view. `expected` is the board's
 * full count (`null` for a PuzzleBoard); `excluded` is whether the project
 * leaves the image out. */
export function cellStatus(
  image: ImageDetection | null,
  expected: number | null,
  excluded: boolean,
): CellStatus {
  if (image === null) {
    return {
      tone: "missing",
      count: null,
      text: "–",
      sentence: "This camera has no image for this view",
      fraction: null,
    };
  }
  const leftOut = image.left_out ? leftOutSentence(image.left_out) : null;
  const state = image.state;
  let status: CellStatus;
  if (state.kind === "not_detected") {
    status = {
      tone: "pending",
      count: null,
      text: "·",
      sentence: "Not detected yet with the project's detector settings",
      fraction: null,
    };
  } else if (state.kind === "failed") {
    status = {
      tone: "defect",
      count: null,
      text: "!",
      sentence: state.error.message,
      fraction: null,
    };
  } else {
    const count = state.features;
    const outcome = state.outcome ?? null;
    const found = outcome === null || outcome.status === "detected";
    const fraction = expected && expected > 0 ? Math.min(1, count / expected) : null;
    const short = expected !== null && expected > 0 && count < expected;
    const tone: CellTone =
      count === 0 || outcome?.status === "not_found"
        ? "defect"
        : !found || short || leftOut !== null
          ? "warn"
          : "normal";
    const base =
      outcome !== null && outcome.status !== "detected"
        ? outcomeSentence(outcome)
        : expected
          ? `${formatCount(count)} of ${formatCount(expected)} features`
          : `${formatCount(count)} features`;
    status = { tone, count, text: formatCount(count), sentence: base, fraction };
  }
  if (leftOut !== null && state.kind !== "failed") {
    status = { ...status, sentence: `${status.sentence}. Last run: ${leftOut}` };
  }
  return excluded
    ? { ...status, tone: "excluded", sentence: `${status.sentence}. Excluded` }
    : status;
}

/** What narrows the rows of the overview matrix. */
export interface DetectFilter {
  /** Only views with an image that is not fine. */
  problemsOnly: boolean;
  /** Only views the project excludes (wholly or in one image). */
  excludedOnly: boolean;
  /** Only views whose worst count is below this; `null` for no limit. */
  countBelow: number | null;
  /** Only views whose token contains this text (case-insensitive). */
  token: string;
}

export const NO_DETECT_FILTER: DetectFilter = {
  problemsOnly: false,
  excludedOnly: false,
  countBelow: null,
  token: "",
};

export function isFiltering(filter: DetectFilter): boolean {
  return (
    filter.problemsOnly ||
    filter.excludedOnly ||
    filter.countBelow !== null ||
    filter.token.trim() !== ""
  );
}

/** The views that pass `filter`, with their position in `views`, in order.
 * `status(view, camera)` is the cell's status. */
export function filterRows(
  views: readonly ViewDetections[],
  filter: DetectFilter,
  status: (view: ViewDetections, camera: number) => CellStatus,
): { view: ViewDetections; index: number }[] {
  const needle = filter.token.trim().toLowerCase();
  return views.flatMap((view, index) => {
    if (needle !== "" && !view.token.toLowerCase().includes(needle)) return [];
    const cells = view.images.map((_, c) => status(view, c));
    if (
      filter.problemsOnly &&
      !cells.some((s) => s.tone === "warn" || s.tone === "defect")
    )
      return [];
    if (filter.excludedOnly && !cells.some((s) => s.tone === "excluded")) return [];
    if (filter.countBelow !== null) {
      const counts = cells.flatMap((s) => (s.count === null ? [] : [s.count]));
      if (counts.length === 0 || Math.min(...counts) >= filter.countBelow) return [];
    }
    return [{ view, index }];
  });
}

export interface OverviewStats {
  /** Images the dataset has. */
  images: number;
  /** Images with a cached detection. */
  analysed: number;
  /** Images with no cached detection. */
  pending: number;
  /** Images that failed to read or detect. */
  failed: number;
  /** Images the detector found no board in. */
  notFound: number;
}

export function overviewStats(overview: DetectionOverview): OverviewStats {
  const stats: OverviewStats = {
    images: 0,
    analysed: 0,
    pending: 0,
    failed: 0,
    notFound: 0,
  };
  for (const view of overview.views) {
    for (const image of view.images) {
      if (image === null) continue;
      stats.images += 1;
      if (image.state.kind === "not_detected") stats.pending += 1;
      else if (image.state.kind === "failed") stats.failed += 1;
      else {
        stats.analysed += 1;
        if (image.state.features === 0 || image.state.outcome?.status === "not_found")
          stats.notFound += 1;
      }
    }
  }
  return stats;
}

/** Whether the project has never been detected (no image has a cache). */
export function neverDetected(stats: OverviewStats): boolean {
  return stats.images > 0 && stats.analysed === 0 && stats.failed === 0;
}

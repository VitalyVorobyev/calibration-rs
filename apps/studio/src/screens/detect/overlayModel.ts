/** A detection as the overlays draw it: `DetectionDetail` (the library's,
 * one entry per feature plus the target's per-kind detail) turned into
 * `TargetDetection`s for `@vitavision/overlays`, one function per target
 * kind, and the rows the Features and Rejected tabs list.
 *
 * Ids: a feature is its index in `detail.features` (a number, so a hover
 * or a press on the overlay maps straight back to its row); a marker is
 * `m<n>`, an edge bit `e<n>`, an origin dot `o<n>`, a rejected candidate
 * `r<n>` and a rejected marker `x<n>`. */
import type {
  DetectionDetail,
  EllipseDetail,
  PointRejection,
  RejectedMarker,
  RejectedPoint,
  RingCenter,
} from "../../api/generated/types";
import type {
  TargetCorner,
  TargetDetection,
  TargetEdgeBit,
  TargetEllipse,
  TargetId,
  TargetMarker,
  TargetRing,
} from "@vitavision/overlays";
import { rejectionSentence } from "./detectModel";

export type TargetKindName = DetectionDetail["target"]["kind"];

/** The sides of a feature a table and a readout show. */
export interface FeatureRow {
  /** The feature's index in the detection. */
  index: number;
  /** The id the detector gave it (a marker-board corner, a ring code); `null` without. */
  id: number | null;
  /** The board cell `i,j`, when it has one. */
  grid: [number, number] | null;
  x: number;
  y: number;
  /** A corner's score (corner kinds). */
  score: number | null;
  /** The centre estimator (ring kinds). */
  center: RingCenter | null;
  /** The detector's confidence in a ring centre. */
  confidence: number | null;
  /** The 1σ of the position along its least certain direction, px. */
  sigmaPx: number | null;
  /** A PuzzleBoard corner is one end of an edge whose bit disagrees with the master pattern. */
  masterMismatch: boolean;
  world: [number, number, number];
}

/** √ of the larger eigenvalue of `[xx, xy, yy]`; `null` when it is not a covariance. */
export function sigmaOf(cov: readonly [number, number, number] | null | undefined) {
  if (!cov) return null;
  const [xx, xy, yy] = cov;
  const mean = (xx + yy) / 2;
  const radius = Math.hypot((xx - yy) / 2, xy);
  const largest = mean + radius;
  return Number.isFinite(largest) && largest >= 0 ? Math.sqrt(largest) : null;
}

const NO_FEATURES: ReadonlySet<number> = new Set();

const key = (p: readonly [number, number]) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`;

/** The features of `detail` as table rows. */
export function featureRows(detail: DetectionDetail): FeatureRow[] {
  const target = detail.target;
  // PuzzleBoard corners whose edge bit disagrees with the master pattern.
  const mismatched = new Set<string>();
  if (target.kind === "puzzleboard") {
    for (const bit of target.edge_bits ?? []) {
      if (!bit.matches_master) {
        for (const end of bit.endpoints_px) mismatched.add(key(end));
      }
    }
  }
  return detail.features.map((feature, index) => {
    const corner =
      target.kind === "ringgrid" ? undefined : (target.corners[index] ?? undefined);
    const ring =
      target.kind === "ringgrid" ? (target.rings[index] ?? undefined) : undefined;
    const cov = ring?.center_cov_px ?? feature.cov_px ?? null;
    return {
      index,
      id: corner?.id ?? ring?.id ?? null,
      grid: corner?.grid ?? ring?.grid ?? null,
      x: feature.image_xy[0],
      y: feature.image_xy[1],
      score: corner?.score ?? null,
      center: ring?.center ?? null,
      confidence: ring?.confidence ?? null,
      sigmaPx: sigmaOf(cov),
      masterMismatch: mismatched.has(key(feature.image_xy)),
      world: feature.world_xyz,
    };
  });
}

/** The feature's label on the overlay and in tables: its board cell, or
 * for a PuzzleBoard its master id, or a ring's code. */
export function featureLabel(kind: TargetKindName, row: FeatureRow): string {
  if (kind === "puzzleboard" && row.id !== null) return String(row.id);
  if (kind === "ringgrid") {
    if (row.id !== null) return String(row.id);
  }
  if (row.grid) return `${row.grid[0]},${row.grid[1]}`;
  return String(row.index);
}

function ellipse(e: EllipseDetail): TargetEllipse {
  return { rx: e.semi_axes_px[0], ry: e.semi_axes_px[1], angle: e.angle_rad };
}

function cornersOf(
  kind: TargetKindName,
  detail: DetectionDetail,
  rows: readonly FeatureRow[],
  axes: boolean,
  excluded: ReadonlySet<number>,
): TargetCorner[] {
  return rows.map((row) => {
    const corner =
      detail.target.kind === "ringgrid"
        ? undefined
        : (detail.target.corners[row.index] ?? undefined);
    const feature = detail.features[row.index]!;
    return {
      id: row.index,
      x: row.x,
      y: row.y,
      ...(row.grid ? { i: row.grid[0], j: row.grid[1] } : {}),
      ...(row.score !== null ? { score: row.score } : {}),
      ...(axes && corner?.axes_rad
        ? { angle: corner.axes_rad[0], angle2: corner.axes_rad[1] }
        : {}),
      ...(feature.cov_px ? { cov: feature.cov_px } : {}),
      label: featureLabel(kind, row),
      ...(excluded.has(row.index) ? { excluded: true } : {}),
      ...(row.masterMismatch ? { flagged: true } : {}),
    };
  });
}

/** The accepted part of a detection, for `TargetOverlay`. `axes` adds the
 * chessboard corners' edge directions; `excluded` holds the indices of the
 * features the project leaves out (drawn dimmed and crossed). A PuzzleBoard
 * corner off the master pattern is `flagged`. */
export function acceptedDetection(
  detail: DetectionDetail,
  options: { axes: boolean; excluded?: ReadonlySet<number> },
): TargetDetection {
  const excluded = options.excluded ?? NO_FEATURES;
  const target = detail.target;
  const rows = featureRows(detail);
  switch (target.kind) {
    case "chessboard":
      return {
        kind: "chessboard",
        corners: cornersOf("chessboard", detail, rows, options.axes, excluded),
      };
    case "charuco":
      return {
        kind: "charuco",
        corners: cornersOf("charuco", detail, rows, false, excluded),
        markers: target.markers.map((m, n): TargetMarker => ({
          id: `m${n}`,
          corners: m.quad_px.flat(),
          label: String(m.id),
        })),
      };
    case "puzzleboard":
      return {
        kind: "puzzleboard",
        corners: cornersOf("puzzleboard", detail, rows, false, excluded),
        edgeBits: (target.edge_bits ?? []).map((bit, n): TargetEdgeBit => {
          const [a, b] = bit.endpoints_px;
          return {
            id: `e${n}`,
            x: (a[0] + b[0]) / 2,
            y: (a[1] + b[1]) / 2,
            radius: Math.hypot(a[0] - b[0], a[1] - b[1]) / 4,
            bit: bit.bit === 1 ? 1 : 0,
            confidence: bit.confidence,
          };
        }),
      };
    case "ringgrid": {
      const rings = target.rings.map((ring, index): TargetRing => {
        const row = rows[index]!;
        const outer = ring.outer ?? ring.inner;
        return {
          id: index,
          x: row.x,
          y: row.y,
          outer: outer ? ellipse(outer) : { rx: 2, ry: 2, angle: 0 },
          ...(ring.outer && ring.inner ? { inner: ellipse(ring.inner) } : {}),
          label: featureLabel("ringgrid", row),
          ...(excluded.has(index) ? { excluded: true } : {}),
        };
      });
      return {
        kind: "ringgrid",
        rings,
        circles: (target.origin_dots_px ?? []).map((p, n) => ({
          id: `o${n}`,
          x: p[0],
          y: p[1],
          polarity: "white" as const,
          label: "origin",
        })),
      };
    }
  }
}

/** One thing the rejected overlay draws and the Rejected tab lists. */
export interface RejectedItem {
  /** The overlay id: `r<n>` for a point, `x<n>` for a marker. */
  id: TargetId;
  kind: "point" | "marker";
  x: number;
  y: number;
  /** The reason, in words. */
  sentence: string;
  /** The grouping key: a point's rejection kind, or `marker`. */
  group: PointRejection["kind"] | "marker";
  /** What the candidate was, when it had a label. */
  label: string | null;
  point?: RejectedPoint;
  marker?: RejectedMarker;
}

function centreOf(quad: RejectedMarker["quad_px"]): [number, number] {
  const x = quad.reduce((s, p) => s + p[0], 0) / 4;
  const y = quad.reduce((s, p) => s + p[1], 0) / 4;
  return [x, y];
}

/** The candidates the detector left out: its points and, for ChArUco,
 * the marker cells that decoded as another marker. */
export function rejectedItems(detail: DetectionDetail): RejectedItem[] {
  const points = (detail.rejected ?? []).map((point, n): RejectedItem => ({
    id: `r${n}`,
    kind: "point",
    x: point.image_xy[0],
    y: point.image_xy[1],
    sentence: rejectionSentence(point.reason),
    group: point.reason.kind,
    label: point.grid
      ? `${point.grid[0]},${point.grid[1]}`
      : point.id !== null && point.id !== undefined
        ? `id ${point.id}`
        : null,
    point,
  }));
  const target = detail.target;
  const markers =
    target.kind === "charuco"
      ? (target.rejected_markers ?? []).map((marker, n): RejectedItem => {
          const [x, y] = centreOf(marker.quad_px);
          return {
            id: `x${n}`,
            kind: "marker",
            x,
            y,
            sentence: `Marker decodes as ${marker.decoded_id}, board expects ${marker.expected_id}`,
            group: "marker",
            label: `${marker.decoded_id} ≠ ${marker.expected_id}`,
            marker,
          };
        })
      : [];
  return [...points, ...markers];
}

/** The rejected part of a detection, for a dimmed `TargetOverlay`. */
export function rejectedDetection(items: readonly RejectedItem[]): TargetDetection {
  return {
    kind: "corners",
    corners: items
      .filter((i) => i.kind === "point")
      .map((i): TargetCorner => ({
        id: i.id,
        x: i.x,
        y: i.y,
        label: "",
      })),
    markers: items
      .filter((i) => i.kind === "marker" && i.marker)
      .map((i): TargetMarker => ({
        id: i.id,
        corners: i.marker!.quad_px.flat(),
        label: i.label ?? "",
      })),
  };
}

/** What differs between two detections of the same image (the saved one
 * and a preview), for the diff strip. Features are the same when their
 * target point matches (within a micrometre) */
export interface DetectionDiff {
  added: number;
  removed: number;
  /** Features present in both whose board label changed. */
  relabelled: number;
}

export function diffDetections(
  saved: DetectionDetail,
  preview: DetectionDetail,
): DetectionDiff {
  const same = (a: number[], b: number[]) =>
    a.every((v, i) => Math.abs(v - b[i]!) <= 1e-6);
  const labelOf = (d: DetectionDetail, i: number) =>
    d.target.kind === "ringgrid" ? d.target.rings[i]?.grid : d.target.corners[i]?.grid;
  let relabelled = 0;
  let matched = 0;
  preview.features.forEach((f, pi) => {
    const si = saved.features.findIndex(
      (s) => Math.hypot(s.image_xy[0] - f.image_xy[0], s.image_xy[1] - f.image_xy[1]) < 2,
    );
    if (si < 0) return;
    matched += 1;
    const a = labelOf(saved, si);
    const b = labelOf(preview, pi);
    if (
      (a && b && !same([...a], [...b])) ||
      !same(saved.features[si]!.world_xyz, f.world_xyz)
    )
      relabelled += 1;
  });
  return {
    added: preview.features.length - matched,
    removed: saved.features.length - matched,
    relabelled,
  };
}

/** The diff as a sentence: "+3 features, −1, 2 labels changed". */
export function describeDiff(diff: DetectionDiff): string {
  if (diff.added === 0 && diff.removed === 0 && diff.relabelled === 0)
    return "Same features as the saved settings";
  const noun = (n: number) => (n === 1 ? "feature" : "features");
  const parts: string[] = [];
  if (diff.added > 0) parts.push(`+${diff.added} ${noun(diff.added)}`);
  if (diff.removed > 0) parts.push(`−${diff.removed} ${noun(diff.removed)}`);
  if (diff.relabelled > 0)
    parts.push(
      `${diff.relabelled} ${diff.relabelled === 1 ? "label" : "labels"} changed`,
    );
  return `${parts.join(", ")} against the saved settings`;
}

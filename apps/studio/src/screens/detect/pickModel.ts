/** Boxing features on the stage: which of them lie under a rectangle. */
export interface PickPoint {
  /** The feature's index in the detection. */
  index: number;
  x: number;
  y: number;
}

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** The indices of `points` inside `box` (either corner first). */
export function pointsInBox(points: readonly PickPoint[], box: Box): number[] {
  const [xa, xb] = [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)];
  const [ya, yb] = [Math.min(box.y0, box.y1), Math.max(box.y0, box.y1)];
  return points
    .filter((p) => p.x >= xa && p.x <= xb && p.y >= ya && p.y <= yb)
    .map((p) => p.index);
}

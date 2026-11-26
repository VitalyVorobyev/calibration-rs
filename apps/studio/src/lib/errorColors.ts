/** Color scale for target reprojection error in pixels. */
export function colorForError(err: number): string {
  if (err < 1) return "#1abc9c";
  if (err < 2) return "#2ecc71";
  if (err < 5) return "#f1c40f";
  if (err < 10) return "#e67e22";
  return "#e74c3c";
}

/** Color scale for laser point-to-plane distance in millimeters.
 * Thresholds follow the device norm (<0.2 mm plane σ is healthy);
 * same palette as `colorForError` so the two legends read alike. */
export function colorForLaserError(mm: number): string {
  if (mm < 0.2) return "#1abc9c";
  if (mm < 0.5) return "#2ecc71";
  if (mm < 1.0) return "#f1c40f";
  if (mm < 2.0) return "#e67e22";
  return "#e74c3c";
}

import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  FrameKey,
  TargetFeatureResidual,
  ViewportTransform,
} from "../types";
import { IDENTITY_TRANSFORM } from "../types";

interface FrameCanvasProps {
  frame: FrameKey;
  /** All target residuals for the loaded export; the canvas filters
   * down to those matching `frame.{pose, camera}` before drawing. */
  residuals: TargetFeatureResidual[];
  /** Render-time transform applied via `ctx.setTransform`. Defaults to
   * identity. Future commits introduce wheel zoom / drag pan against
   * this prop. */
  transform?: ViewportTransform;
  /** Surfaced when the underlying image fails to decode. */
  onError?: (msg: string) => void;
}

/** Cap on rendered arrow length so a freak diverged feature does not
 * dominate the canvas. */
const ARROW_LENGTH_CAP_PX = 40;
/** Multiplier applied to the residual vector so sub-pixel errors are
 * actually visible on a 640×480 canvas. */
const ARROW_GAIN = 30;

export function FrameCanvas({
  frame,
  residuals,
  transform = IDENTITY_TRANSFORM,
  onError,
}: FrameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setImgUrl(null);
    invoke<string>("load_image", { path: frame.abs_path })
      .then((dataUrl) => {
        if (!cancelled) setImgUrl(dataUrl);
      })
      .catch((e) => {
        if (!cancelled) onError?.(`Could not load image: ${e}`);
      });
    return () => {
      cancelled = true;
    };
  }, [frame.abs_path, onError]);

  useEffect(() => {
    if (!imgUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const roi = frame.roi;
      const sx = roi?.x ?? 0;
      const sy = roi?.y ?? 0;
      const sw = roi?.w ?? img.naturalWidth;
      const sh = roi?.h ?? img.naturalHeight;
      canvas.width = sw;
      canvas.height = sh;
      ctx.imageSmoothingEnabled = false;
      ctx.setTransform(
        transform.scale,
        0,
        0,
        transform.scale,
        transform.tx,
        transform.ty,
      );
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      // Per ImageManifest convention, residual pixel coords are already
      // ROI-local — the canvas was just blitted from `[sx, sx+sw) ×
      // [sy, sy+sh)` to `(0, 0)`, so we draw residuals in canvas-pixel
      // space directly.
      drawResidualArrows(ctx, residuals, frame);
    };
    img.onerror = () => onError?.("Image failed to decode.");
    img.src = imgUrl;
  }, [frame, residuals, imgUrl, transform, onError]);

  return (
    <canvas
      ref={canvasRef}
      className="max-h-full max-w-full border border-border [image-rendering:pixelated]"
    />
  );
}

function drawResidualArrows(
  ctx: CanvasRenderingContext2D,
  all: TargetFeatureResidual[],
  frame: FrameKey,
) {
  const arrows = all.filter(
    (r) => r.pose === frame.pose && r.camera === frame.camera,
  );
  for (const r of arrows) {
    if (!r.projected_px) continue;
    const ox = r.observed_px[0];
    const oy = r.observed_px[1];
    const dx0 = r.projected_px[0] - r.observed_px[0];
    const dy0 = r.projected_px[1] - r.observed_px[1];
    const mag = Math.hypot(dx0, dy0);
    if (mag < 1e-6) continue;
    const gain = Math.min(ARROW_GAIN, ARROW_LENGTH_CAP_PX / Math.max(mag, 1e-6));
    const dx = dx0 * gain;
    const dy = dy0 * gain;
    ctx.strokeStyle = colorForError(r.error_px ?? mag);
    ctx.lineWidth = 1.5;
    drawArrow(ctx, ox, oy, ox + dx, oy + dy);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath();
    ctx.arc(ox, oy, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 4;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineTo(
    x2 - head * Math.cos(angle - Math.PI / 6),
    y2 - head * Math.sin(angle - Math.PI / 6),
  );
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - head * Math.cos(angle + Math.PI / 6),
    y2 - head * Math.sin(angle + Math.PI / 6),
  );
  ctx.stroke();
}

/** Cool→warm ramp matching the histogram bucket edges (1, 2, 5, 10 px)
 * already used for `target_hist_per_camera`. Keeps the visual scale
 * consistent across UI surfaces if a histogram view is added later. */
export function colorForError(err: number): string {
  if (err < 1) return "#1abc9c";
  if (err < 2) return "#2ecc71";
  if (err < 5) return "#f1c40f";
  if (err < 10) return "#e67e22";
  return "#e74c3c";
}

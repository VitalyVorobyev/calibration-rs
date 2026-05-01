import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  FrameKey,
  TargetFeatureResidual,
  ViewportTransform,
} from "../types";

interface FrameCanvasProps {
  frame: FrameKey;
  /** All target residuals for the loaded export; the canvas filters
   * down to those matching `frame.{pose, camera}` before drawing. */
  residuals: TargetFeatureResidual[];
  /** Surfaced when the underlying image fails to decode. */
  onError?: (msg: string) => void;
}

/** Imperative handle the toolbar uses to drive zoom/fit without
 * lifting transform state into the parent. We will lift it later
 * (compare mode wants linked transforms across two canvases), but
 * the imperative shape keeps the single-canvas case simple. */
export interface FrameCanvasHandle {
  /** Set transform so the ROI fills the container with letterbox. */
  fit(): void;
  /** Set transform so 1 image-pixel = 1 display-pixel, centred. */
  reset1to1(): void;
  /** Multiply current scale by `factor`, anchored at the canvas
   * centre. Clamped to [0.25, 16]. */
  zoomBy(factor: number): void;
}

const ARROW_LENGTH_CAP_PX = 40;
const ARROW_GAIN = 30;
const SCALE_MIN = 0.25;
const SCALE_MAX = 16;

export const FrameCanvas = forwardRef<FrameCanvasHandle, FrameCanvasProps>(
  function FrameCanvas({ frame, residuals, onError }, ref) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [container, setContainer] = useState({ w: 0, h: 0 });
    const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
    const [transform, setTransform] = useState<ViewportTransform>({
      scale: 1,
      tx: 0,
      ty: 0,
    });

    // ROI helpers — the image is the manifest source PNG; we crop down
    // to the camera tile via roi when present.
    const roi = frame.roi;

    // Load + decode the image.
    useEffect(() => {
      let cancelled = false;
      setImgEl(null);
      invoke<string>("load_image", { path: frame.abs_path })
        .then((dataUrl) => {
          if (cancelled) return;
          const img = new Image();
          img.onload = () => {
            if (!cancelled) setImgEl(img);
          };
          img.onerror = () => {
            if (!cancelled) onError?.("Image failed to decode.");
          };
          img.src = dataUrl;
        })
        .catch((e) => {
          if (!cancelled) onError?.(`Could not load image: ${e}`);
        });
      return () => {
        cancelled = true;
      };
    }, [frame.abs_path, onError]);

    // Track container size so the canvas can fill it at 1:1 device
    // pixels (no CSS scaling — keeps zoomed pixels crisp).
    useLayoutEffect(() => {
      const el = containerRef.current;
      if (!el) return;
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          setContainer({ w: Math.max(1, width | 0), h: Math.max(1, height | 0) });
        }
      });
      observer.observe(el);
      return () => observer.disconnect();
    }, []);

    // Compute the canonical "fit" transform: scale ROI uniformly into
    // the container, centred.
    const computeFit = useCallback((): ViewportTransform => {
      const sw = roi?.w ?? imgEl?.naturalWidth ?? 0;
      const sh = roi?.h ?? imgEl?.naturalHeight ?? 0;
      if (sw === 0 || sh === 0 || container.w === 0 || container.h === 0) {
        return { scale: 1, tx: 0, ty: 0 };
      }
      const scale = Math.min(container.w / sw, container.h / sh);
      const tx = (container.w - scale * sw) / 2;
      const ty = (container.h - scale * sh) / 2;
      return { scale, tx, ty };
    }, [roi, imgEl, container]);

    // Reset to fit whenever the frame or the image/container sizes
    // change. The dependency on `imgEl` ensures we wait until the
    // image is decoded; without it the first fit happens against the
    // wrong (zero) image dimensions.
    useEffect(() => {
      if (!imgEl || container.w === 0 || container.h === 0) return;
      setTransform(computeFit());
    }, [frame.abs_path, imgEl, container, computeFit]);

    // Imperative API for the toolbar.
    useImperativeHandle(
      ref,
      () => ({
        fit: () => setTransform(computeFit()),
        reset1to1: () => {
          const sw = roi?.w ?? imgEl?.naturalWidth ?? 0;
          const sh = roi?.h ?? imgEl?.naturalHeight ?? 0;
          setTransform({
            scale: 1,
            tx: (container.w - sw) / 2,
            ty: (container.h - sh) / 2,
          });
        },
        zoomBy: (factor) =>
          setTransform((t) => zoomAround(t, factor, container.w / 2, container.h / 2)),
      }),
      [computeFit, roi, imgEl, container],
    );

    // Wheel-to-zoom anchored at the cursor.
    const handleWheel = useCallback(
      (e: React.WheelEvent<HTMLCanvasElement>) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
        setTransform((t) => zoomAround(t, factor, cx, cy));
      },
      [],
    );

    // Drag-to-pan with the primary button.
    const dragRef = useRef<{
      startX: number;
      startY: number;
      tx0: number;
      ty0: number;
    } | null>(null);
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return;
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        tx0: transform.tx,
        ty0: transform.ty,
      };
    };
    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const d = dragRef.current;
      if (!d) return;
      setTransform((t) => ({
        ...t,
        tx: d.tx0 + (e.clientX - d.startX),
        ty: d.ty0 + (e.clientY - d.startY),
      }));
    };
    const handleMouseUp = () => {
      dragRef.current = null;
    };

    // Render: clear, apply transform, blit ROI, draw arrows.
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !imgEl || container.w === 0) return;
      canvas.width = container.w;
      canvas.height = container.h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.setTransform(
        transform.scale,
        0,
        0,
        transform.scale,
        transform.tx,
        transform.ty,
      );
      const sx = roi?.x ?? 0;
      const sy = roi?.y ?? 0;
      const sw = roi?.w ?? imgEl.naturalWidth;
      const sh = roi?.h ?? imgEl.naturalHeight;
      ctx.drawImage(imgEl, sx, sy, sw, sh, 0, 0, sw, sh);
      // Per ImageManifest convention residual coords are already in the
      // ROI-local frame, so we draw them directly in canvas-px space —
      // the transform takes care of zoom/pan.
      drawResidualArrows(ctx, residuals, frame, transform.scale);
    }, [imgEl, container, transform, roi, residuals, frame]);

    return (
      <div
        ref={containerRef}
        className="relative h-full w-full overflow-hidden rounded-md bg-bg-soft"
      >
        <canvas
          ref={canvasRef}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="block h-full w-full cursor-grab [image-rendering:pixelated] active:cursor-grabbing"
        />
      </div>
    );
  },
);

function zoomAround(
  t: ViewportTransform,
  factor: number,
  cx: number,
  cy: number,
): ViewportTransform {
  const next = clampScale(t.scale * factor);
  const real = next / t.scale;
  return {
    scale: next,
    tx: cx - (cx - t.tx) * real,
    ty: cy - (cy - t.ty) * real,
  };
}

function clampScale(s: number): number {
  return Math.min(SCALE_MAX, Math.max(SCALE_MIN, s));
}

function drawResidualArrows(
  ctx: CanvasRenderingContext2D,
  all: TargetFeatureResidual[],
  frame: FrameKey,
  scale: number,
) {
  const arrows = all.filter(
    (r) => r.pose === frame.pose && r.camera === frame.camera,
  );
  // Draw stroke widths in canvas-px (scale-invariant), so arrows stay
  // crisp at any zoom level. We pre-scale lineWidth + arrowhead size
  // by 1/scale because the transform expands them otherwise.
  const inv = 1 / scale;
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
    ctx.lineWidth = 1.5 * inv;
    drawArrow(ctx, ox, oy, ox + dx, oy + dy, inv);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath();
    ctx.arc(ox, oy, 1.6 * inv, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  inv: number,
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 4 * inv;
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
 * consistent across UI surfaces. */
export function colorForError(err: number): string {
  if (err < 1) return "#1abc9c";
  if (err < 2) return "#2ecc71";
  if (err < 5) return "#f1c40f";
  if (err < 10) return "#e67e22";
  return "#e74c3c";
}

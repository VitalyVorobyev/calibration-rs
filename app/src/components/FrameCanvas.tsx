import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type {
  FrameKey,
  TargetFeatureResidual,
  ViewportTransform,
} from "../types";
import { IDENTITY_TRANSFORM } from "../types";

interface FrameCanvasProps {
  frame: FrameKey;
  /** Target residuals for the loaded export; filtered down to
   * `frame.{pose, camera}` before drawing. */
  residuals: TargetFeatureResidual[];
  /** Decoded image element. `null` while loading. */
  image: HTMLImageElement | null;
  /** Controlled transform. When provided, the canvas treats it as
   * authoritative and emits all updates via `onTransformChange`.
   * When `undefined` the canvas keeps an internal transform — the
   * single-pane mode used by `DiagnoseWorkspace`. Compare mode passes
   * a shared transform from above. */
  transform?: ViewportTransform;
  onTransformChange?: (t: ViewportTransform) => void;
  /** Surfaced when something fails (the parent owns the error UI). */
  onError?: (msg: string) => void;
  /** Called on `mousemove` with image-pixel coordinates (ROI-local,
   * matching the residual frame). `null` when the cursor leaves the
   * image area. */
  onCursor?: (cursor: { x: number; y: number } | null) => void;
  /** Called on a discrete left-click (mousedown → mouseup with little
   * cursor movement, distinguishing it from a pan-drag) at the given
   * image-pixel coordinates. Used by the epipolar workspace to pick a
   * pane-A pixel for the overlay request. */
  onPick?: (pixel: { x: number; y: number }) => void;
  /** Visual ring drawn around the canvas when this pane is the
   * keyboard-active pane in compare mode. */
  active?: boolean;
}

/** Pixel-distance threshold below which a mousedown→mouseup pair is a
 * click (firing `onPick`) and above which it's a pan-drag. */
const CLICK_DRAG_THRESHOLD_PX = 4;

/** Imperative handle the toolbar uses to drive zoom/fit. */
export interface FrameCanvasHandle {
  fit(): void;
  reset1to1(): void;
  zoomBy(factor: number): void;
}

const ARROW_LENGTH_CAP_PX = 40;
const ARROW_GAIN = 30;
const SCALE_MIN = 0.25;
const SCALE_MAX = 16;

export const FrameCanvas = forwardRef<FrameCanvasHandle, FrameCanvasProps>(
  function FrameCanvas(
    {
      frame,
      residuals,
      image,
      transform: controlled,
      onTransformChange,
      onError,
      onCursor,
      onPick,
      active,
    },
    ref,
  ) {
    void onError;
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [container, setContainer] = useState({ w: 0, h: 0 });
    const [internal, setInternal] = useState<ViewportTransform>(IDENTITY_TRANSFORM);
    const isControlled = controlled !== undefined;
    const transform = controlled ?? internal;

    const setTransform = useCallback(
      (next: ViewportTransform | ((prev: ViewportTransform) => ViewportTransform)) => {
        const value =
          typeof next === "function"
            ? (next as (p: ViewportTransform) => ViewportTransform)(transform)
            : next;
        if (isControlled) {
          onTransformChange?.(value);
        } else {
          setInternal(value);
          onTransformChange?.(value);
        }
      },
      [isControlled, onTransformChange, transform],
    );

    const roi = frame.roi;

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

    const computeFit = useCallback((): ViewportTransform => {
      const sw = roi?.w ?? image?.naturalWidth ?? 0;
      const sh = roi?.h ?? image?.naturalHeight ?? 0;
      if (sw === 0 || sh === 0 || container.w === 0 || container.h === 0) {
        return IDENTITY_TRANSFORM;
      }
      const scale = Math.min(container.w / sw, container.h / sh);
      const tx = (container.w - scale * sw) / 2;
      const ty = (container.h - scale * sh) / 2;
      return { scale, tx, ty };
    }, [roi, image, container]);

    // Auto-fit on frame change is owned by the canvas only in
    // uncontrolled mode. When the parent controls the transform
    // (compare mode with linked panes), it decides when to re-fit.
    useEffect(() => {
      if (isControlled) return;
      if (!image || container.w === 0 || container.h === 0) return;
      setInternal(computeFit());
    }, [frame.abs_path, image, container, computeFit, isControlled]);

    useImperativeHandle(
      ref,
      () => ({
        fit: () => setTransform(computeFit()),
        reset1to1: () => {
          const sw = roi?.w ?? image?.naturalWidth ?? 0;
          const sh = roi?.h ?? image?.naturalHeight ?? 0;
          setTransform({
            scale: 1,
            tx: (container.w - sw) / 2,
            ty: (container.h - sh) / 2,
          });
        },
        zoomBy: (factor) =>
          setTransform((t) =>
            zoomAround(t, factor, container.w / 2, container.h / 2),
          ),
      }),
      [computeFit, roi, image, container, setTransform],
    );

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
      [setTransform],
    );

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
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const ix = (cx - transform.tx) / transform.scale;
        const iy = (cy - transform.ty) / transform.scale;
        const sw = roi?.w ?? image?.naturalWidth ?? 0;
        const sh = roi?.h ?? image?.naturalHeight ?? 0;
        if (ix >= 0 && iy >= 0 && ix < sw && iy < sh) {
          onCursor?.({ x: ix, y: iy });
        } else {
          onCursor?.(null);
        }
      }
      const d = dragRef.current;
      if (d) {
        setTransform((t) => ({
          ...t,
          tx: d.tx0 + (e.clientX - d.startX),
          ty: d.ty0 + (e.clientY - d.startY),
        }));
      }
    };
    const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const d = dragRef.current;
      dragRef.current = null;
      // Distinguish click from pan-drag by total cursor movement.
      // Below the threshold we treat it as a pick; above it the user
      // was dragging and shouldn't accidentally select a feature.
      if (!d || !onPick) return;
      const dx = e.clientX - d.startX;
      const dy = e.clientY - d.startY;
      if (Math.hypot(dx, dy) > CLICK_DRAG_THRESHOLD_PX) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const ix = (cx - transform.tx) / transform.scale;
      const iy = (cy - transform.ty) / transform.scale;
      const sw = roi?.w ?? image?.naturalWidth ?? 0;
      const sh = roi?.h ?? image?.naturalHeight ?? 0;
      if (ix >= 0 && iy >= 0 && ix < sw && iy < sh) {
        onPick({ x: ix, y: iy });
      }
    };
    const handleMouseLeave = () => {
      dragRef.current = null;
      onCursor?.(null);
    };

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !image || container.w === 0) return;
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
      const sw = roi?.w ?? image.naturalWidth;
      const sh = roi?.h ?? image.naturalHeight;
      ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
      drawResidualArrows(ctx, residuals, frame, transform.scale);
    }, [image, container, transform, roi, residuals, frame]);

    return (
      <div
        ref={containerRef}
        className={`relative h-full w-full overflow-hidden rounded-md bg-bg-soft transition-shadow ${
          active ? "ring-1 ring-brand" : ""
        }`}
      >
        <canvas
          ref={canvasRef}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
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

export function colorForError(err: number): string {
  if (err < 1) return "#1abc9c";
  if (err < 2) return "#2ecc71";
  if (err < 5) return "#f1c40f";
  if (err < 10) return "#e67e22";
  return "#e74c3c";
}

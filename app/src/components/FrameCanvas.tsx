import {
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type Ref,
  type RefObject,
} from "react";
import {
  ImageLayer,
  ImageStage,
  imageViewBox,
  useScreenPx,
  useStage,
  type StageContext,
  type StageView,
} from "@vitavision/stage2d";
import type {
  FrameKey,
  LaserFeatureResidual,
  TargetFeatureResidual,
  ViewportTransform,
} from "../types";
import {
  buildLaserOverlay,
  buildResidualArrows,
  LASER_LINE_STROKE,
  LASER_UNRESOLVED_FILL,
  OBSERVED_DOT_FILL,
  STROKE_PX,
} from "./frameOverlayPaths";

interface FrameCanvasProps {
  frame: FrameKey;
  /** Target residuals for the loaded export; filtered down to
   * `frame.{pose, camera}` before drawing. */
  residuals: TargetFeatureResidual[];
  /** Laser residuals to plot instead of target arrows — pass them when
   * `frame` is a laser-kind frame (with `residuals={[]}`). Observed
   * pixels are colored by point-to-plane distance; the projected laser
   * line is drawn from the first record that carries endpoints. */
  laserResiduals?: LaserFeatureResidual[] | undefined;
  /** Decoded image element. `null` while loading. */
  image: HTMLImageElement | null;
  /** Controlled transform. When provided (`null` included: "open at fit,
   * then report it"), the viewer treats it as authoritative and emits all
   * updates via `onTransformChange`. When `undefined` it keeps an internal
   * transform and re-fits whenever the frame, image or ROI changes — the
   * single-pane mode used by `DiagnoseWorkspace`. Compare mode passes a
   * shared transform from above. */
  transform?: ViewportTransform | null | undefined;
  onTransformChange?: ((t: ViewportTransform) => void) | undefined;
  /** Surfaced when something fails (the parent owns the error UI). */
  onError?: ((msg: string) => void) | undefined;
  /** Called while the pointer moves over the image with ROI-local
   * image-pixel coordinates (pixel *centres* at integers, the convention
   * of the residuals). `null` when the cursor leaves the image area. */
  onCursor?: (cursor: { x: number; y: number } | null) => void;
  /** Called on a discrete click (a press that never became a pan-drag) at
   * the given ROI-local image-pixel coordinates (centre convention). Used
   * by the epipolar workspace to pick a pane-A pixel. */
  onPick?: ((pixel: { x: number; y: number }) => void) | undefined;
  /** Visual ring drawn around the viewer when this pane is the
   * keyboard-active pane in compare mode. */
  active?: boolean;
  /** Receives the imperative zoom/fit handle. */
  ref?: Ref<FrameCanvasHandle>;
  /** Extra stage layers, drawn in the frame's ROI-local image
   * coordinates above the residuals (e.g. the epipolar overlay). */
  children?: ReactNode;
}

/** Imperative handle the toolbar uses to drive zoom/fit. */
export interface FrameCanvasHandle {
  fit(): void;
  reset1to1(): void;
  zoomBy(factor: number): void;
}

const LABEL = "Frame viewer";

/** Mirrors the enclosing stage's context into a ref, so the imperative
 * handle and the click handler (both outside the stage) can use it. */
function StageBridge({ ctxRef }: { ctxRef: RefObject<StageContext | null> }) {
  const ctx = useStage();
  useLayoutEffect(() => {
    ctxRef.current = ctx;
  });
  useLayoutEffect(
    () => () => {
      ctxRef.current = null;
    },
    [ctxRef],
  );
  return null;
}

/** Residual arrows and the laser overlay as batched SVG paths. */
function ResidualLayer({
  frame,
  residuals,
  laserResiduals,
}: {
  frame: FrameKey;
  residuals: TargetFeatureResidual[];
  laserResiduals: LaserFeatureResidual[] | undefined;
}) {
  const { image } = useStage();
  const unit = useScreenPx()(1);
  const { pose, camera } = frame;
  const arrows = useMemo(
    () => buildResidualArrows(residuals, { pose, camera }, unit),
    [residuals, pose, camera, unit],
  );
  const laser = useMemo(
    () =>
      laserResiduals ? buildLaserOverlay(laserResiduals, { pose, camera }, unit) : null,
    [laserResiduals, pose, camera, unit],
  );
  const stroke = STROKE_PX * unit;
  return (
    <svg
      viewBox={imageViewBox(image)}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      aria-hidden="true"
    >
      {arrows.arrows.map((a) => (
        <path key={a.color} d={a.d} fill="none" stroke={a.color} strokeWidth={stroke} />
      ))}
      {arrows.dots && <path d={arrows.dots} fill={OBSERVED_DOT_FILL} />}
      {laser?.line && (
        <path
          d={laser.line}
          fill="none"
          stroke={LASER_LINE_STROKE}
          strokeWidth={stroke}
        />
      )}
      {laser?.dots.map((d) => (
        <path key={d.color} d={d.d} fill={d.color} />
      ))}
      {laser?.unresolved && <path d={laser.unresolved} fill={LASER_UNRESOLVED_FILL} />}
    </svg>
  );
}

export function FrameCanvas({
  frame,
  residuals,
  laserResiduals,
  image,
  transform: controlled,
  onTransformChange,
  onError,
  onCursor,
  onPick,
  active,
  ref,
  children,
}: FrameCanvasProps) {
  const ctxRef = useRef<StageContext | null>(null);
  const isControlled = controlled !== undefined;
  const [internal, setInternal] = useState<StageView | null>(null);
  // While controlled, mirror the parent's view so that handing control back
  // (unlinking compare panes) continues from what is on screen: an already
  // measured stage never re-opens a `null` view on its own.
  if (isControlled && controlled && internal !== controlled) setInternal(controlled);
  const view = isControlled ? controlled : internal;

  const roi = frame.roi;
  const width = roi?.w ?? image?.naturalWidth ?? 0;
  const height = roi?.h ?? image?.naturalHeight ?? 0;

  // Uncontrolled: a new frame, image or ROI re-opens at fit. Done during
  // render, keyed on the inputs, so the first painted frame is already fitted.
  const fitKey = `${frame.abs_path}|${roi?.x},${roi?.y},${roi?.w},${roi?.h}`;
  const [fitInputs, setFitInputs] = useState({ fitKey, image });
  if (fitInputs.fitKey !== fitKey || fitInputs.image !== image) {
    setFitInputs({ fitKey, image });
    if (!isControlled) setInternal(null);
  }

  useImperativeHandle(
    ref,
    () => ({
      fit: () => ctxRef.current?.fit(),
      reset1to1: () => {
        const ctx = ctxRef.current;
        if (!ctx) return;
        ctx.setView({
          scale: 1,
          tx: (ctx.box.width - ctx.image.width) / 2,
          ty: (ctx.box.height - ctx.image.height) / 2,
        });
      },
      zoomBy: (factor) => {
        const ctx = ctxRef.current;
        if (ctx) ctx.zoomTo(ctx.view.scale * factor);
      },
    }),
    [],
  );

  const ringClass = active ? "ring-1 ring-signal" : "";

  if (!image || width === 0 || height === 0) {
    return (
      <div
        role="application"
        aria-label={LABEL}
        className={`h-full w-full rounded-control border border-line bg-raised ${ringClass}`}
      />
    );
  }

  const inside = (p: { x: number; y: number }) =>
    p.x >= -0.5 && p.y >= -0.5 && p.x < width - 0.5 && p.y < height - 0.5;

  const handleClick = (e: ReactPointerEvent<HTMLDivElement>) => {
    const ctx = ctxRef.current;
    if (!onPick || !ctx) return;
    const p = ctx.toImage({ x: e.clientX, y: e.clientY });
    if (inside(p)) onPick({ x: p.x, y: p.y });
  };

  return (
    <ImageStage
      image={{ width, height }}
      view={view}
      onView={(next) => {
        if (!isControlled) setInternal(next);
        onTransformChange?.(next);
      }}
      initialView="fit"
      // The workspaces own the keyboard (f, 1, +, -, arrows step poses); the
      // stage's own shortcuts would zoom twice.
      shortcuts={false}
      label={LABEL}
      className={`bg-raised ${ringClass}`}
      onHover={(p) => onCursor?.(p && inside(p) ? { x: p.x, y: p.y } : null)}
      onBackgroundClick={handleClick}
    >
      <StageBridge ctxRef={ctxRef} />
      {/* The ROI crop: the full image offset by the ROI origin, clipped to the stage. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div
          className="absolute"
          style={{
            left: -(roi?.x ?? 0),
            top: -(roi?.y ?? 0),
            width: image.naturalWidth,
            height: image.naturalHeight,
          }}
        >
          {/* Pixelated from 1:1 up, as the old canvas drew it: residuals are read
              against individual pixels. */}
          <ImageLayer
            src={image.src}
            alt=""
            pixelatedAbove={1}
            onError={() => onError?.(`Could not display ${frame.abs_path}`)}
          />
        </div>
      </div>
      <ResidualLayer
        frame={frame}
        residuals={residuals}
        laserResiduals={laserResiduals}
      />
      {children}
    </ImageStage>
  );
}
